import cv2 as cv


def resize(img, new_height):
    h, w = img.shape[:2]
    new_width = int(round(w*new_height/h))
    return cv.resize(img, dsize=(new_width, new_height))
