import enum

import numpy as np
import cv2
from photolab import color_spaces as cs

"""
Correction filters for Protanopia color blindness as described in the paper
"Smartphone Based Image Color Correction for Color Blindness" by Lamiaa A. Elrefaei.
https://www.researchgate.net/profile/Lamiaa-Elrefaei/publication/326626897_Smartphone_Based_Image_Color_Correction_for_Color_Blindness/links/5b636af70f7e9b00b2a23f2e/Smartphone-Based-Image-Color-Correction-for-Color-Blindness.pdf
"""

Grayscale_matrix = np.array([
    [0.114, 0.587, 0.299],
    [0.114, 0.587, 0.299],
    [0.114, 0.587, 0.299],
])

BGR_to_LMS_matrix = np.array([
    # BGR to LMS matrix
    [4.11935, 43.5161, 17.884],
    [3.86714, 27.1554, 3.45565],
    [1.46709, 0.184309, 0.0299566],
])

LMS_Protanopia_matrix = np.array([
    [0, 2.02344, -2.52581],
    [0, 1, 0],
    [0, 0, 1],
])
BGR_Protanopia_matrix = np.linalg.inv(BGR_to_LMS_matrix) @ LMS_Protanopia_matrix @ BGR_to_LMS_matrix

LMS_Deuteranopia_matrix = np.array([
    [1, 0, 0],
    [0.49421, 0, 1.24827],
    [0, 0, 1],
])
BGR_Deuteranopia_matrix = np.linalg.inv(BGR_to_LMS_matrix) @ LMS_Deuteranopia_matrix @ BGR_to_LMS_matrix

LMS_Tritanopia_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [-0.395913, 0.801109, 0],
])
BGR_Tritanopia_matrix = np.linalg.inv(BGR_to_LMS_matrix) @ LMS_Tritanopia_matrix @ BGR_to_LMS_matrix


def compute_prot_LMS_correction_matrix(shift: float, strength: float):
    """
    :param shift: float in [-1.0, 1.0]
    :param strength: float typically in [0.0, 1.0]
    :return: 3x3 matrix
    """
    rad = (shift) * np.pi + np.pi / 4
    f1 = np.cos(rad)
    f2 = np.sin(rad)
    Protanopia_shift_matrix = np.array([
        [0, 0, 0],
        [f1, 1, 0],
        [f2, 0, 1],
    ])
    print(shift, strength)
    Prot_LMS_correction_matrix = np.identity(3) + strength * Protanopia_shift_matrix @ (np.identity(3) - BGR_Protanopia_matrix)
    return Prot_LMS_correction_matrix


def compute_deuter_LMS_correction_matrix(shift: float, strength: float):
    """
    :param shift: float in [-1.0, 1.0]
    :param strength: float typically in [0.0, 1.0]
    :return: 3x3 matrix
    """
    rad = (shift) * np.pi + np.pi / 4
    f1 = np.cos(rad)
    f2 = np.sin(rad)
    Deuteranopia_shift_matrix = np.array([
        [1, f1, 0],
        [0, 0, 0],
        [0, f2, 1],
    ])
    print(shift, strength)
    Deuter_LMS_correction_matrix = np.identity(3) + strength * Deuteranopia_shift_matrix @ (np.identity(3) - BGR_Deuteranopia_matrix)
    return Deuter_LMS_correction_matrix


def compute_trit_LMS_correction_matrix(shift: float, strength: float):
    """
    :param shift: float in [-1.0, 1.0]
    :param strength: float typically in [0.0, 1.0]
    :return: 3x3 matrix
    """
    rad = (shift) * np.pi + np.pi / 4
    f1 = np.cos(rad)
    f2 = np.sin(rad)
    Tritanopia_shift_matrix = np.array([
        [1, 0, f1],
        [0, 1, f2],
        [0, 0, 0],
    ])
    print(shift, strength)
    Trit_LMS_correction_matrix = np.identity(3) + strength * Tritanopia_shift_matrix @ (np.identity(3) - BGR_Tritanopia_matrix)
    return Trit_LMS_correction_matrix


class ColorFilter(enum.IntEnum):
    NONE = enum.auto()
    GRAYSCALE = enum.auto()
    SIMULATE_PROTANOPIA = enum.auto()
    SIMULATE_DEUTERANOPIA = enum.auto()
    SIMULATE_TRITANOPIA = enum.auto()
    PROTANOPIA_LMS_CORRECTION = enum.auto()
    DEUTERANOPIA_LMS_CORRECTION = enum.auto()
    TRITANOPIA_LMS_CORRECTION = enum.auto()


COLOR_FILTER_MATRIX = {
    ColorFilter.NONE: np.identity(3),
    ColorFilter.GRAYSCALE: Grayscale_matrix,
    ColorFilter.SIMULATE_PROTANOPIA: BGR_Protanopia_matrix,
    ColorFilter.SIMULATE_DEUTERANOPIA: BGR_Deuteranopia_matrix,
    ColorFilter.SIMULATE_TRITANOPIA: BGR_Tritanopia_matrix,
}


def compute_correction_matrix(color_space, shift, strength):
    match color_space:
        case ColorFilter.NONE:
            return np.identity(3)
        case ColorFilter.PROTANOPIA_LMS_CORRECTION:
            return compute_prot_LMS_correction_matrix(shift, strength)
        case ColorFilter.DEUTERANOPIA_LMS_CORRECTION:
            return compute_deuter_LMS_correction_matrix(shift, strength)
        case ColorFilter.TRITANOPIA_LMS_CORRECTION:
            return compute_trit_LMS_correction_matrix(shift, strength)


def BGR_to_LMS(bgr):
    assert bgr.dtype == np.uint8
    lms = apply_color_matrix(bgr.astype(float) / 255, BGR_to_LMS_matrix)
    assert lms.dtype == float
    return lms


def LMS_to_BGR(lms):
    assert lms.dtype == np.float
    bgr = apply_color_matrix(lms, np.linalg.inv(BGR_to_LMS_matrix)) * 255
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    assert bgr.dtype == np.uint8
    return bgr


def apply_color_simulation_filter(bgr, color_filter: ColorFilter):
    assert bgr.dtype == np.uint8
    matrix = COLOR_FILTER_MATRIX[color_filter]
    res = cs.apply_color_matrix(bgr.astype(float), matrix)
    res = np.clip(res, 0, 255).astype(np.uint8)
    assert res.dtype == np.uint8
    return res


def apply_color_correction_filter(bgr, color_filter: ColorFilter, shift=0.5, strength=1.0):
    assert bgr.dtype == np.uint8
    matrix = compute_correction_matrix(color_filter, shift, strength)
    res = cs.apply_color_matrix(bgr.astype(float), matrix)
    res = np.clip(res, 0, 255).astype(np.uint8)
    assert res.dtype == np.uint8
    return res
