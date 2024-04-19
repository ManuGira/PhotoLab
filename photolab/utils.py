import os

import cv2 as cv
import numpy as np


def color_hex2rgb(hexa):
    # hexa must be in format #123456 in hexadecimal
    r, g, b = hexa[1:3], hexa[3:5], hexa[5:7]
    r, g, b = [int(c, 16) for c in [r, g, b]]
    return r, g, b



def sec_to_hms(sec):
    h = int(sec / 60 / 60)
    m = int(sec / 60 - h * 60)
    s = int(sec - h * 60 * 60 - m * 60)
    return h, m, s


def pth(*args):
    out = "."
    for arg in args:
        if isinstance(arg, list):
            arg = os.path.join(*arg)
        out = os.path.join(out, arg)
    out = os.path.normpath(out)
    out = os.path.join(".", out)
    return out

def insert_text_before_file_extension(filename, text):
    spl = filename.split(".")
    name = '.'.join(spl[:-1])
    ext = spl[-1]
    return f"{name}{text}.{ext}"

def export_to_png(name, data):
    folder = "gallery"
    if not os.path.exists(folder):
        os.mkdir(folder)
    file = os.path.join(folder, f"{name}.png")
    cv.imwrite(file, data)


def convert_to_show(img, reshape=None):
    img = np.array(img).copy()
    if reshape is not None:
        img = np.reshape(img, newshape=reshape)

    # map False,True to 0,255
    if img.dtype == np.bool:
        img8 = np.zeros_like(img, dtype=np.uint8)
        img8[img] = 255
        img = img8

    # force color image
    if len(img.shape) == 3 and not img.shape[2]==3:
        img = img[:, :, 0]
        img.shape = img.shape[0:2]
    if not len(img.shape) == 3:
        img.shape += (1,)
        img = np.repeat(img, repeats=3, axis=2)

    return img


def draw_poly(img, pts, fill=False, color=127, reshape=None):
    img = convert_to_show(img, reshape=reshape)

    pts = np.squeeze(pts)
    if len(pts.shape) == 2:
        pts = [pts]
    pts = np.array(np.round(pts), dtype=np.int)

    color = np.squeeze(color)
    if len(color.shape) == 0:
        color = (int(color),)*3

    if fill:
        img = cv.fillPoly(img, pts, color)
    else:
        img = cv.polylines(img, pts, True, color=color)
    return img


def resize(img, new_height):
    h, w = img.shape[:2]
    new_width = int(round(w*new_height/h))
    return cv.resize(img, dsize=(new_width, new_height))


# def compute_gradient(img):
#     dx, dy = cv.spatialGradient(img, ksize=3)
#     dx = dx.astype(float)/255
#     dy = dy.astype(float)/255
#     gradient = np.sqrt(dx**2+dy**2)
#     return gradient

def compute_gradient(img):
    H, W = img.shape
    gradient_xy = np.zeros(shape=(H, W, 2), dtype=img.dtype)
    gradient_xy[:, :, 0] = cv.filter2D(img.astype(float), cv.CV_64F, np.array([[-1, 1]]))
    gradient_xy[:, :, 1] = cv.filter2D(img.astype(float), cv.CV_64F, np.array([-1, 1]))

    # compute gradient magnitude
    grad_magn = np.sqrt(gradient_xy[:, :, 0] ** 2 + gradient_xy[:, :, 1] ** 2)
    return grad_magn

# def compute_gradient(hits, normalizing_threshold=64):
#     """
#
#     :param hits: (M x N) np array of integer
#     :param normalizing_threshold: normalizing and clipping threshold. Lower threshold will increase gradient effects.
#     :return: (M x N) np array of type float64
#     """
#     H, W = hits.shape
#     gradient_xy = np.zeros(shape=(H, W, 2), dtype=hits.dtype)
#     gradient_xy[:, :, 0] = cv.filter2D(hits.astype(float), cv.CV_64F, np.array([[-1, 1]]))
#     gradient_xy[:, :, 1] = cv.filter2D(hits.astype(float), cv.CV_64F, np.array([-1, 1]))
#
#     # compute gradient magnitude
#     grad_magn = np.sqrt(gradient_xy[:, :, 0] ** 2 + gradient_xy[:, :, 1] ** 2)
#     grad_magn.shape += (1,)
#     grad_magn = np.repeat(grad_magn, repeats=2, axis=2)
#
#     # clip gradient magnitude to threshold
#     mask = grad_magn > normalizing_threshold
#     gradient_xy[mask] = normalizing_threshold * gradient_xy[mask] / grad_magn[mask]
#     gradient_xy = gradient_xy.astype(np.float64) / normalizing_threshold
#
#     return gradient_xy

def imshow(img, name="utils", ms=0, reshape=None):
    toshow = convert_to_show(img, reshape=reshape)
    cv.imshow(name, toshow)
    cv.waitKey(ms)


def invert_cdf(cdf, size=None):
    """
    acompute the reciprocal of the cdf
    min and max of cdf must lay in [0, 1]
    cdf is a sorted array in increasing order
    """
    if size is None:
        size = len(cdf)

    xss = np.linspace(0, 1, size)
    # xs.shape += (1,)
    s0 = len(cdf)
    xs = np.linspace(0, 1, s0)
    ys = cdf.copy()
    # ys.shape += (1,)

    # cdf_inv0 = np.concatenate((ys, xs), axis=1)
    out = []
    xr = 0
    k = 1
    x1, y1, x0, y0 = 0, 0, 0, 0
    for yr in xss:
        while y1 <= yr and k < s0:
            x0, y0 = x1, y1
            x1, y1 = xs[k], ys[k]
            k += 1

        dy = y1-y0
        dx = x1-x0
        t = 1 if dy == 0 else (yr-y0)/dy

        xr = x0 + t*dx
        out.append(xr)

    return out
