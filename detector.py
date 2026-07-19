"""
detector.py
===========

Stuff-Detector: a lightweight, few-shot object recognizer built on top of a
pretrained MobileNetV3-Large backbone.

Overview
--------
Instead of training a classifier from scratch (which would require thousands
of labeled images per class), this module takes a *metric learning* shortcut:

1.  A pretrained CNN (MobileNetV3-Large, trained on ImageNet) is loaded
    WITHOUT its classification head. With ``pooling="avg"`` the network
    outputs a single fixed-length feature vector (an "embedding") per image.

2.  Every reference image found in ``data/images`` is passed through the
    network once at startup. The resulting embedding is L2-normalized and
    stored in a matrix, one row per known object.

3.  At prediction time, the incoming camera frame is embedded the same way.
    Because all vectors are unit-length, a simple dot product between the
    frame embedding and each stored reference embedding yields the *cosine
    similarity* -- a score in roughly [-1, 1] where higher means
    "more visually similar".

4.  The reference with the highest similarity wins. If even the best score
    falls below ``UNKNOWN_THRESHOLD``, the detector refuses to guess and
    reports the object as ``"unknown"`` instead of returning a bad match.

This approach is often called a "nearest-neighbor classifier over deep
features" and it works surprisingly well for small, personal datasets:
adding a new object is as easy as dropping a single JPEG into the image
folder and restarting the app.

Public API
----------
The public surface intentionally stays tiny and stable:

``StuffDetector(image_dir=IMAGE_DIR)``
    Builds the model and learns every reference image in ``image_dir``.

``StuffDetector.predict(frame) -> (label, confidence)``
    Classifies a single BGR frame (as produced by OpenCV / ``cv2``).

``IMAGE_DIR`` / ``UNKNOWN_THRESHOLD``
    Module-level defaults, importable by other parts of the project.

Everything else in this file (helpers, dataclasses, the CLI at the bottom)
is supporting machinery and may change freely between versions.

Usage
-----
Typical usage from the rest of the project::

    from detector import StuffDetector

    detector = StuffDetector()
    label, confidence = detector.predict(frame)

The module can also be executed directly for a quick smoke test::

    python detector.py path/to/some_image.jpg
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Standard library imports
# --------------------------------------------------------------------------
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Third-party imports
# --------------------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf


# ==========================================================================
# Logging
# ==========================================================================
#
# A module-level logger keeps console output tidy and lets the host
# application silence or redirect detector chatter without touching this
# file. If the application never configures logging, the basicConfig call
# below ensures messages still reach the console (matching the behaviour of
# the original print() based implementation).
# ==========================================================================

logger = logging.getLogger("stuff_detector")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    # Don't double-print if the root logger is also configured.
    logger.propagate = False


# ==========================================================================
# Module-level constants (public)
# ==========================================================================

#: Directory that holds one reference JPEG per known object.
#: The filename (minus extension) becomes the object's label.
IMAGE_DIR = Path("data/images")

#: Cosine similarity below this value means the detector is not confident
#: enough and will answer "unknown" instead of the best match.
#:
#: Tuning notes:
#:   * Raise it  -> fewer false positives, more "unknown" answers.
#:   * Lower it  -> more eager matching, higher risk of wrong labels.
#: 0.55 was found to be a good balance for MobileNetV3-Large features.
UNKNOWN_THRESHOLD = 0.55

#: Label returned when no reference clears the similarity threshold.
UNKNOWN_LABEL = "unknown"


# ==========================================================================
# Module-level constants (internal defaults)
# ==========================================================================

#: Input resolution expected by the backbone network (height, width).
#: MobileNetV3 was trained on 224x224 crops, so we resize everything to
#: exactly this size before inference.
_INPUT_SIZE: Tuple[int, int] = (224, 224)

#: File extensions considered when scanning the reference directory.
#: Kept as JPEG-only to mirror the project's original behaviour.
_REFERENCE_EXTENSIONS: Tuple[str, ...] = (".jpg",)

#: Numeric type used for all embedding math.
_DTYPE = np.float32


# ==========================================================================
# Configuration dataclass
# ==========================================================================


@dataclass(frozen=True)
class DetectorConfig:
    """
    Bundle of tunable knobs for :class:`StuffDetector`.

    The defaults reproduce the classic behaviour of the project exactly, so
    ``StuffDetector()`` works the same as it always has. Power users can
    build a custom config and pass it in::

        cfg = DetectorConfig(unknown_threshold=0.65)
        detector = StuffDetector(config=cfg)

    Attributes
    ----------
    image_dir:
        Folder containing the reference images.
    unknown_threshold:
        Minimum cosine similarity required to accept a match.
    input_size:
        (height, width) fed to the CNN backbone.
    extensions:
        Filename extensions scanned inside ``image_dir``.
    """

    image_dir: Path = field(default_factory=lambda: Path(IMAGE_DIR))
    unknown_threshold: float = UNKNOWN_THRESHOLD
    input_size: Tuple[int, int] = _INPUT_SIZE
    extensions: Tuple[str, ...] = _REFERENCE_EXTENSIONS

    def __post_init__(self) -> None:
        # Defensive validation: catch bad values immediately at construction
        # time instead of failing mysteriously deep inside TensorFlow later.
        if not (-1.0 <= self.unknown_threshold <= 1.0):
            raise ValueError(
                "unknown_threshold must lie within [-1, 1] because it is "
                f"compared against a cosine similarity, got "
                f"{self.unknown_threshold!r}"
            )

        height, width = self.input_size
        if height <= 0 or width <= 0:
            raise ValueError(
                f"input_size must contain positive dimensions, "
                f"got {self.input_size!r}"
            )

        if not self.extensions:
            raise ValueError("extensions must not be empty")


# ==========================================================================
# Result dataclass
# ==========================================================================


@dataclass(frozen=True)
class Prediction:
    """
    Rich result object returned by :meth:`StuffDetector.predict_detailed`.

    The plain :meth:`StuffDetector.predict` method still returns a simple
    ``(label, confidence)`` tuple for backwards compatibility; this class
    exists for callers that want the full picture (e.g. debugging overlays
    or a future web dashboard).

    Attributes
    ----------
    label:
        Final answer after threshold filtering -- may be ``"unknown"``.
    confidence:
        Cosine similarity of the single best reference match.
    best_match:
        Name of the closest reference, *before* threshold filtering.
        Useful to see what the detector "almost" said.
    is_known:
        True when ``confidence`` cleared the unknown threshold.
    ranking:
        Every reference label paired with its similarity, sorted from most
        to least similar. Handy for top-k displays.
    elapsed_ms:
        Wall-clock inference time in milliseconds (embedding + matching).
    """

    label: str
    confidence: float
    best_match: str
    is_known: bool
    ranking: Tuple[Tuple[str, float], ...]
    elapsed_ms: float

    # -- convenience helpers ------------------------------------------------

    def as_tuple(self) -> Tuple[str, float]:
        """Collapse to the classic ``(label, confidence)`` pair."""
        return self.label, self.confidence

    def top(self, k: int = 3) -> Tuple[Tuple[str, float], ...]:
        """Return the ``k`` most similar references as (label, score)."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        return self.ranking[:k]

    def __str__(self) -> str:  # pragma: no cover - cosmetic only
        marker = "" if self.is_known else " (below threshold)"
        return (
            f"{self.label} @ {self.confidence:.3f}{marker} "
            f"[{self.elapsed_ms:.1f} ms]"
        )


# ==========================================================================
# Image helper functions
# ==========================================================================
#
# These small, pure functions keep the StuffDetector class itself focused on
# orchestration. Each helper does exactly one image-related job and can be
# unit-tested in isolation.
# ==========================================================================


def _is_valid_frame(frame: object) -> bool:
    """
    Cheap sanity check that ``frame`` looks like a decoded OpenCV image.

    OpenCV represents images as ``numpy.ndarray`` with shape
    ``(height, width, channels)``. We accept anything with two spatial
    dimensions and reject empty arrays, ``None``, and non-array inputs.
    """
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    if frame.ndim not in (2, 3):
        return False
    return True


def _ensure_three_channels(frame: np.ndarray) -> np.ndarray:
    """
    Normalize channel layouts so downstream code can assume 3-channel BGR.

    Handles the two edge cases a webcam or image file can realistically
    produce:

    * grayscale (H, W)            -> replicated into 3 channels
    * BGRA with alpha (H, W, 4)   -> alpha channel dropped
    * regular BGR (H, W, 3)       -> passed through untouched
    """
    if frame.ndim == 2:
        # Single-channel grayscale: expand to BGR by replication.
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    channels = frame.shape[2]
    if channels == 4:
        # Discard the alpha channel (e.g. PNGs with transparency).
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if channels == 3:
        return frame

    raise ValueError(
        f"unsupported channel count: expected 1, 3 or 4 channels, "
        f"got shape {frame.shape}"
    )


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV's default BGR channel order to RGB.

    MobileNetV3 (like nearly every ImageNet model) expects RGB input, while
    ``cv2.imread`` and ``cv2.VideoCapture`` hand back BGR. Forgetting this
    swap is one of the most common silent accuracy killers, so it lives in
    its own clearly-named function.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize_for_model(
    frame: np.ndarray, size: Tuple[int, int] = _INPUT_SIZE
) -> np.ndarray:
    """
    Resize ``frame`` to the exact resolution the backbone expects.

    Notes
    -----
    * ``size`` is (height, width) but ``cv2.resize`` wants (width, height),
      hence the reversed indexing below.
    * A simple stretch (no aspect-ratio preserving letterbox) matches the
      original implementation. Since both the reference images and the live
      frames go through the *same* stretch, the distortion largely cancels
      out in the similarity comparison.
    """
    height, width = size
    return cv2.resize(frame, (width, height)).astype(_DTYPE)


def _make_batch(image: np.ndarray) -> np.ndarray:
    """
    Wrap a single preprocessed image into a batch of one.

    Keras models always consume 4-D tensors of shape
    ``(batch, height, width, channels)`` -- even for a single image.

    MobileNetV3 is special among Keras applications: it contains its own
    ``Rescaling`` / normalization layers *inside* the model graph, so the
    raw 0-255 float pixels can be fed directly without calling a separate
    ``preprocess_input`` function.
    """
    return np.expand_dims(image, axis=0)


def _read_image(path: Path) -> Optional[np.ndarray]:
    """
    Load an image from disk, returning ``None`` on failure.

    ``cv2.imread`` never raises -- it silently returns ``None`` for missing
    files, unsupported formats, or corrupt data. This wrapper preserves that
    contract but funnels every disk read through one place so future
    improvements (EXIF rotation, caching, ...) have a single home.
    """
    image = cv2.imread(str(path))
    if image is None:
        return None
    return image


# ==========================================================================
# Embedding math helpers
# ==========================================================================


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    Scale ``vector`` to unit length.

    After normalization, the dot product between two vectors equals their
    cosine similarity, which turns the whole matching step into one cheap
    matrix-vector multiplication.

    A tiny epsilon guards against a (theoretically possible) all-zero
    embedding, which would otherwise trigger a division-by-zero warning.
    """
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        # Degenerate embedding -- return as-is rather than producing NaNs.
        return vector.astype(_DTYPE)
    return (vector / norm).astype(_DTYPE)


def _cosine_similarities(
    reference_matrix: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between ``query`` and every reference row.

    Parameters
    ----------
    reference_matrix:
        Array of shape ``(num_references, embedding_dim)`` whose rows are
        already L2-normalized.
    query:
        L2-normalized vector of shape ``(embedding_dim,)``.

    Returns
    -------
    np.ndarray
        Vector of shape ``(num_references,)`` -- one similarity per class.
    """
    # Because every vector is unit-length, cosine similarity reduces to a
    # plain dot product. NumPy's `@` operator broadcasts this across all
    # reference rows in a single vectorized call.
    return reference_matrix @ query


def _rank_labels(
    labels: Sequence[str], similarities: np.ndarray
) -> Tuple[Tuple[str, float], ...]:
    """
    Pair each label with its similarity and sort best-first.

    Returns an immutable tuple so callers can safely cache or share the
    ranking without accidental mutation.
    """
    order = np.argsort(similarities)[::-1]  # descending
    return tuple((labels[i], float(similarities[i])) for i in order)


# ==========================================================================
# Label helpers
# ==========================================================================


def _label_from_path(path: Path) -> str:
    """
    Derive a human-readable label from a reference image filename.

    Examples
    --------
    ``keyboard.jpg``     -> ``keyboard``
    ``PI_5.jpg.jpg``     -> ``PI_5``   (double-extension typo tolerated)

    The extra ``removesuffix(".jpg")`` call handles files that were
    accidentally saved with a doubled extension -- a real occurrence in this
    project's dataset that we deliberately keep supporting.
    """
    return path.stem.removesuffix(".jpg")


def _discover_reference_paths(
    image_dir: Path, extensions: Iterable[str] = _REFERENCE_EXTENSIONS
) -> List[Path]:
    """
    Find every candidate reference image inside ``image_dir``.

    Results are sorted alphabetically so the learned class order is
    deterministic across runs and machines -- important for reproducible
    debugging sessions.
    """
    paths: List[Path] = []
    for extension in extensions:
        paths.extend(image_dir.glob(f"*{extension}"))
    return sorted(paths)


# ==========================================================================
# The detector
# ==========================================================================


class StuffDetector:
    """
    Few-shot object recognizer backed by MobileNetV3-Large embeddings.

    Lifecycle
    ---------
    1. ``__init__`` builds the CNN backbone and immediately "learns" every
       reference image in the configured directory (one forward pass each).
    2. ``predict`` / ``predict_detailed`` embed incoming frames and match
       them against the stored references via cosine similarity.
    3. Optional extras -- ``add_reference``, ``reload_references``,
       ``summary`` -- support interactive experimentation without restarts.

    Thread-safety
    -------------
    The class performs no locking. TensorFlow inference itself is
    thread-safe, but ``add_reference`` / ``reload_references`` mutate shared
    state, so call those from a single thread only.

    Examples
    --------
    >>> detector = StuffDetector()
    >>> label, confidence = detector.predict(frame)
    >>> print(label, confidence)
    keyboard 0.83
    """

    # ----------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------

    def __init__(
        self,
        image_dir: Path | str = IMAGE_DIR,
        config: Optional[DetectorConfig] = None,
    ) -> None:
        """
        Build the backbone network and learn all reference images.

        Parameters
        ----------
        image_dir:
            Folder of reference JPEGs. Ignored when ``config`` is provided
            (the config's own ``image_dir`` wins in that case).
        config:
            Optional :class:`DetectorConfig` for advanced tuning.

        Raises
        ------
        RuntimeError
            If the reference directory contains no readable images -- the
            detector would be useless, so we fail fast and loudly.
        """
        if config is None:
            # The common path: replicate historical behaviour using the
            # module defaults, honouring a custom image_dir if given.
            config = DetectorConfig(image_dir=Path(image_dir))

        self.config: DetectorConfig = config

        # -- backbone ------------------------------------------------------
        # Pretrained CNN without the classifier head -> feature embeddings.
        # * include_top=False strips the 1000-class ImageNet softmax.
        # * pooling="avg" applies global average pooling, collapsing the
        #   final feature maps into a single flat vector per image.
        self.model = self._build_model(config.input_size)

        # -- reference database ---------------------------------------------
        # labels[i] corresponds to embeddings[i]; embeddings is a 2-D matrix
        # with one L2-normalized row per known object.
        self.labels: List[str]
        self.embeddings: np.ndarray
        self.labels, self.embeddings = self._load_references(config.image_dir)

    @staticmethod
    def _build_model(input_size: Tuple[int, int]) -> tf.keras.Model:
        """
        Instantiate the MobileNetV3-Large feature extractor.

        Why MobileNetV3-Large?
        * Small and fast enough for real-time webcam inference on CPU.
        * ImageNet pretraining gives it a rich general-purpose notion of
          shape, texture and colour -- exactly what similarity matching
          needs.
        * Built-in preprocessing layers mean raw 0-255 pixels can be fed
          straight in, removing a whole class of normalization bugs.
        """
        height, width = input_size
        return tf.keras.applications.MobileNetV3Large(
            include_top=False,
            pooling="avg",
            input_shape=(height, width, 3),
        )

    # ----------------------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------------------

    def _embed(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Turn one BGR image into a unit-length feature vector.

        The full preprocessing pipeline, step by step:

        1. channel cleanup  -- tolerate grayscale / BGRA inputs
        2. BGR -> RGB       -- match the backbone's training data
        3. resize           -- stretch to the model's fixed input size
        4. batch of one     -- Keras models want a 4-D tensor
        5. forward pass     -- ``training=False`` disables any train-only
                               behaviour (dropout, batch-norm updates)
        6. L2-normalize     -- so dot products become cosine similarities
        """
        if not _is_valid_frame(bgr_image):
            raise ValueError(
                "expected a non-empty numpy image array, "
                f"got {type(bgr_image).__name__}"
            )

        prepared = _ensure_three_channels(bgr_image)
        rgb = _bgr_to_rgb(prepared)
        resized = _resize_for_model(rgb, self.config.input_size)
        batch = _make_batch(resized)  # MobileNetV3 preprocesses internally

        embedding = self.model(batch, training=False).numpy()[0]
        return _l2_normalize(embedding)

    # ----------------------------------------------------------------------
    # Reference loading
    # ----------------------------------------------------------------------

    def _load_references(
        self, image_dir: Path
    ) -> Tuple[List[str], np.ndarray]:
        """
        Scan ``image_dir`` and embed every readable reference image.

        Unreadable files (corrupt, wrong format, permission issues) are
        skipped with a warning rather than aborting the whole startup --
        one bad photo shouldn't take the detector down.

        Returns
        -------
        (labels, embeddings)
            Parallel structures: ``labels[i]`` names the object whose
            embedding lives in row ``i`` of the matrix.

        Raises
        ------
        RuntimeError
            If not a single image could be loaded.
        """
        labels: List[str] = []
        embeddings: List[np.ndarray] = []

        paths = _discover_reference_paths(image_dir, self.config.extensions)

        for path in paths:
            image = _read_image(path)
            if image is None:
                logger.warning("skipping unreadable image: %s", path)
                continue

            label = _label_from_path(path)  # handles PI_5.jpg.jpg
            labels.append(label)
            embeddings.append(self._embed(image))
            logger.info("learned: %s", label)

        if not embeddings:
            raise RuntimeError(f"no reference images found in {image_dir}")

        # np.stack turns the list of 1-D vectors into a (N, dim) matrix so
        # prediction can compare against all classes in one multiplication.
        return labels, np.stack(embeddings)

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Return ``(label, confidence)`` for the object in the frame.

        This is the stable, minimal API used by the camera loop. The label
        is ``"unknown"`` whenever the best cosine similarity fails to reach
        the configured threshold.
        """
        embedding = self._embed(frame)
        similarities = _cosine_similarities(self.embeddings, embedding)

        best = int(similarities.argmax())
        confidence = float(similarities[best])

        # Threshold gate: better to admit ignorance than to mislabel.
        if confidence >= self.config.unknown_threshold:
            label = self.labels[best]
        else:
            label = UNKNOWN_LABEL

        return label, confidence

    def predict_detailed(self, frame: np.ndarray) -> Prediction:
        """
        Like :meth:`predict`, but returns a rich :class:`Prediction` with
        the full similarity ranking and timing information.

        Intended for debugging overlays, logging, and future UI features;
        the hot path in the camera loop should keep using ``predict``.
        """
        started = time.perf_counter()

        embedding = self._embed(frame)
        similarities = _cosine_similarities(self.embeddings, embedding)

        best = int(similarities.argmax())
        confidence = float(similarities[best])
        best_match = self.labels[best]
        is_known = confidence >= self.config.unknown_threshold

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return Prediction(
            label=best_match if is_known else UNKNOWN_LABEL,
            confidence=confidence,
            best_match=best_match,
            is_known=is_known,
            ranking=_rank_labels(self.labels, similarities),
            elapsed_ms=elapsed_ms,
        )

    def predict_top_k(
        self, frame: np.ndarray, k: int = 3
    ) -> Tuple[Tuple[str, float], ...]:
        """
        Return the ``k`` most similar reference labels with their scores.

        No threshold filtering is applied here -- callers see the raw
        ranking and can decide for themselves what to trust.
        """
        return self.predict_detailed(frame).top(k)

    # ----------------------------------------------------------------------
    # Runtime dataset management (optional extras)
    # ----------------------------------------------------------------------

    def add_reference(self, label: str, bgr_image: np.ndarray) -> None:
        """
        Teach the detector a new object at runtime without restarting.

        Parameters
        ----------
        label:
            Name to report when this object is recognized.
        bgr_image:
            A representative BGR photo of the object.

        Notes
        -----
        The new reference lives only in memory; save the image into the
        reference folder yourself if you want it to survive a restart.
        """
        label = label.strip()
        if not label:
            raise ValueError("label must be a non-empty string")

        embedding = self._embed(bgr_image)

        # Append to both parallel structures atomically enough for a
        # single-threaded caller: labels first, then the matrix row.
        self.labels.append(label)
        self.embeddings = np.vstack([self.embeddings, embedding])

        logger.info("learned: %s", label)

    def reload_references(self) -> int:
        """
        Re-scan the reference directory from scratch.

        Useful after dropping new photos into ``data/images`` while the
        application is running. Returns the number of classes now known.
        """
        self.labels, self.embeddings = self._load_references(
            self.config.image_dir
        )
        return len(self.labels)

    # ----------------------------------------------------------------------
    # Introspection helpers
    # ----------------------------------------------------------------------

    def known_labels(self) -> Tuple[str, ...]:
        """Return every label currently in the reference database."""
        return tuple(self.labels)

    def summary(self) -> str:
        """
        Produce a short human-readable status report.

        Handy for a startup banner or a diagnostics endpoint.
        """
        height, width = self.config.input_size
        lines = [
            "StuffDetector summary",
            "---------------------",
            f"backbone        : MobileNetV3-Large ({height}x{width} input)",
            f"embedding dim   : {self.embeddings.shape[1]}",
            f"known objects   : {len(self.labels)}",
            f"threshold       : {self.config.unknown_threshold:.2f}",
            f"reference dir   : {self.config.image_dir}",
            "labels          : " + ", ".join(self.labels),
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------------------
    # Dunder conveniences
    # ----------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of known reference objects."""
        return len(self.labels)

    def __contains__(self, label: object) -> bool:
        """``"keyboard" in detector`` -> is that label known?"""
        return label in self.labels

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"{type(self).__name__}("
            f"classes={len(self.labels)}, "
            f"threshold={self.config.unknown_threshold}, "
            f"image_dir={str(self.config.image_dir)!r})"
        )


# ==========================================================================
# Command-line smoke test
# ==========================================================================
#
# Running this file directly performs a quick end-to-end check without
# needing the webcam:
#
#     python detector.py                     -> load references, print summary
#     python detector.py photo.jpg [...]     -> classify one or more images
#
# This block is guarded by __name__ so importing the module (as main.py and
# the backend do) never triggers any of it.
# ==========================================================================


def _run_cli(argv: Sequence[str]) -> int:
    """
    Tiny CLI entry point. Returns a process exit code (0 = success).

    Kept as a function (rather than inline under ``__main__``) so it can be
    invoked from tests with a fake ``argv``.
    """
    # Step 1: build the detector. This also validates the dataset folder.
    try:
        detector = StuffDetector()
    except RuntimeError as error:
        logger.error("failed to start detector: %s", error)
        return 1

    # Step 2: always show the summary banner.
    logger.info("")
    logger.info(detector.summary())
    logger.info("")

    # Step 3: classify any image paths supplied on the command line.
    exit_code = 0
    for raw_path in argv:
        path = Path(raw_path)
        image = _read_image(path)

        if image is None:
            logger.error("could not read image: %s", path)
            exit_code = 1
            continue

        result = detector.predict_detailed(image)
        logger.info("%s -> %s", path.name, result)

        # Show the runners-up too; useful when tuning the threshold.
        for rank, (label, score) in enumerate(result.top(3), start=1):
            logger.info("    #%d %-20s %.3f", rank, label, score)

    return exit_code


if __name__ == "__main__":
    sys.exit(_run_cli(sys.argv[1:]))
