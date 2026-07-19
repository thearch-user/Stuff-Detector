import threading

import cv2


class _AsyncPredictor:
    """
    Runs detector.predict() in a background thread so the camera loop never
    blocks on CNN inference.

    The display loop hands over the newest frame and immediately reads back
    the most recent completed prediction. Frames that arrive while the
    detector is busy are simply replaced by newer ones -- only the latest
    frame is ever analyzed, so the predictor can never fall behind.
    """

    def __init__(self, detector):
        self._detector = detector
        self._condition = threading.Condition()
        self._pending_frame = None      # newest frame awaiting inference
        self._result = ("...", 0.0)     # last completed (label, confidence)
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Offer a new frame for analysis (overwrites any queued frame)."""
        with self._condition:
            self._pending_frame = frame
            self._condition.notify()

    def latest(self):
        """Return the most recent (label, confidence) without blocking."""
        with self._condition:
            return self._result

    def stop(self):
        with self._condition:
            self._running = False
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _worker(self):
        while True:
            with self._condition:
                # Sleep until a new frame arrives or we are told to stop.
                while self._running and self._pending_frame is None:
                    self._condition.wait()
                if not self._running:
                    return
                frame = self._pending_frame
                self._pending_frame = None

            # Inference happens OUTSIDE the lock so submit()/latest()
            # from the display loop never wait on the CNN.
            result = self._detector.predict(frame)

            with self._condition:
                self._result = result


def Camera(detector=None):
    cap = cv2.VideoCapture(0)

    # Keep the driver's internal queue short so we always read a fresh
    # frame instead of a stale buffered one (reduces perceived latency).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    predictor = _AsyncPredictor(detector) if detector is not None else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if predictor is not None:
                # Hand the frame to the background thread (non-blocking)
                # and draw whatever prediction finished most recently.
                predictor.submit(frame)
                label, confidence = predictor.latest()
                text = f"{label} ({confidence:.2f})"
                color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv2.imshow("camera", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        if predictor is not None:
            predictor.stop()
        cap.release()
        cv2.destroyAllWindows()
