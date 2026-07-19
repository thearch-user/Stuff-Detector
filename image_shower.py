import cv2
from image_loader import load_image

def image_shower(image):

    image = image = load_image(image)
    print(f"image: {image.shape}")

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    load_image(image)
