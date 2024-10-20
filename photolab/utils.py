import os

import cv2
import cv2 as cv


def resize(img, new_height):
    h, w = img.shape[:2]
    new_width = int(round(w*new_height/h))
    return cv.resize(img, dsize=(new_width, new_height))


def easy_save(img, filename):
    dir = os.path.dirname(filename)
    filename_no_ext, ext = os.path.splitext(filename)

    os.makedirs(dir, exist_ok=True)

    n = 1
    while os.path.exists(filename):
        filename = f"{filename_no_ext}-{n:03}{ext}"
        n += 1

    cv2.imwrite(filename, img)
