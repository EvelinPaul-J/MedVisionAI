"""
image_quality.py
Image Quality Gate — runs BEFORE any DR analysis.

In production this role is filled by a small trained CNN
(good-quality vs poor-quality classifier). Here we implement the
same *contract* (image -> quality report) using classical CV
metrics, so the trained model can be swapped in later without
touching any other file — see `assess_quality()`.
"""
import cv2
import numpy as np


def _blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness_score(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def _retina_field_coverage(bgr: np.ndarray) -> float:
    """Rough estimate of how much of the frame the retinal disc fills,
    using a circular Hough-like brightness mask instead of full Hough
    (faster, good enough for a quality gate)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    coverage = float(np.count_nonzero(mask)) / mask.size
    return coverage


def assess_quality(image_path: str) -> dict:
    bgr = cv2.imread(image_path)
    if bgr is None:
        return {
            "status": "invalid",
            "overall_score": 0,
            "focus": 0, "brightness": 0, "field_coverage": 0,
            "message": "Image could not be read. Please recapture."
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blur = _blur_score(gray)
    brightness = _brightness_score(gray)
    coverage = _retina_field_coverage(bgr)

    # Normalize each metric to a 0-100 "score" band (thresholds tuned
    # for typical fundus-camera output; recalibrate against your own
    # capture device during pilot testing).
    focus_score = float(np.clip(blur / 150.0 * 100, 0, 100))
    brightness_score = float(np.clip(100 - abs(brightness - 120) / 120 * 100, 0, 100))
    coverage_score = float(np.clip(coverage * 130, 0, 100))

    overall = round((focus_score * 0.4 + brightness_score * 0.3 + coverage_score * 0.3), 1)

    if overall >= 65:
        status = "acceptable"
        message = "Image quality sufficient for AI screening."
    elif overall >= 45:
        status = "borderline"
        message = "Image quality is borderline. Analysis will proceed, but recapturing is recommended."
    else:
        status = "poor"
        message = "Image quality insufficient. Please reposition the device and recapture."

    return {
        "status": status,
        "overall_score": overall,
        "focus": round(focus_score, 1),
        "brightness": round(brightness_score, 1),
        "field_coverage": round(coverage_score, 1),
        "message": message,
    }
