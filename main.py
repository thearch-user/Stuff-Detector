from camera import Camera
from detector import StuffDetector

if __name__ == "__main__":
    detector = StuffDetector()
    Camera(detector)
