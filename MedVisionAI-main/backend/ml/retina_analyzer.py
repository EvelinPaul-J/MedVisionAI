"""DR analysis engine with trained-model ensemble + safe fallback.

If trained four-model weights exist in backend/models, the ensemble prediction
is included. If weights are absent or incompatible, the app falls back to the
existing classical-CV evidence pipeline so the demo remains runnable.
"""
import cv2
import numpy as np
import json
import os

from ml.deep_models import ensemble_predict

LESION_COLORS = {
    "microaneurysm": (0, 140, 255),
    "hemorrhage": (0, 0, 255),
    "exudate": (0, 255, 255),
}


def _retina_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask


def _find_blobs(mask: np.ndarray, min_area: int, max_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(c)
            blobs.append({"bbox": (x, y, w, h), "area": area})
    return blobs


def _detect_red_lesions(bgr, field_mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 60, 20]), np.array([10, 255, 150])
    lower2, upper2 = np.array([160, 60, 20]), np.array([180, 255, 150])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    red_mask = cv2.bitwise_and(red_mask, field_mask)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _find_blobs(red_mask, 3, 5000), red_mask


def _detect_yellow_lesions(bgr, field_mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 50, 90]), np.array([45, 255, 255]))
    yellow_mask = cv2.bitwise_and(yellow_mask, field_mask)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _find_blobs(yellow_mask, 8, 12000), yellow_mask


def _severity(red_count, yellow_count):
    score = red_count * 1.0 + yellow_count * 0.8
    if score >= 18:
        return "Severe", "urgent"
    if score >= 10:
        return "Moderate", "priority"
    if score >= 4:
        return "Mild", "priority"
    return "No obvious DR evidence", "routine"


def analyze_retina(image_path: str, heatmap_path: str):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError("Uploaded image could not be decoded as an image")

    field = _retina_mask(bgr)
    red_blobs, red_mask = _detect_red_lesions(bgr, field)
    yellow_blobs, yellow_mask = _detect_yellow_lesions(bgr, field)

    severity, risk = _severity(len(red_blobs), len(yellow_blobs))
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    deep = ensemble_predict(image_path, model_dir)

    if deep:
        if deep["prediction"] == "No_DR":
            severity = "No DR (model prediction)"
            risk = "routine"
            recommendation = "No diabetic retinopathy was predicted by the trained classifier ensemble. Clinical review is still recommended when indicated."
        else:
            severity = "DR suspected (model prediction)"
            risk = "priority"
            recommendation = "The trained classifier ensemble predicted diabetic retinopathy. Refer for clinical evaluation; this is an AI-assisted screening result, not a diagnosis."
        confidence = deep["confidence"]
        evidence = [
            f"Ensemble prediction: {deep['prediction']}.",
            f"DR probability: {deep['dr_probability']}%.",
            f"No-DR probability: {deep['no_dr_probability']}%.",
            f"Models used: {', '.join(deep['models_used'])}.",
        ]
    else:
        confidence = min(99.0, 60.0 + min(35.0, (len(red_blobs) + len(yellow_blobs)) * 1.5))
        recommendation = (
            "AI-assisted screening indicates no obvious lesion evidence; maintain routine eye screening."
            if risk == "routine" else
            "Potential retinal lesion evidence detected; clinical eye evaluation is recommended."
        )
        evidence = [
            f"Red lesion-like regions detected: {len(red_blobs)}.",
            f"Yellow lesion-like regions detected: {len(yellow_blobs)}.",
            "No trained deep-learning weights were available, so this result uses the classical-CV fallback.",
        ]

    overlay = bgr.copy()
    for blob in red_blobs:
        x, y, w, h = blob["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), LESION_COLORS["hemorrhage"], 2)
    for blob in yellow_blobs:
        x, y, w, h = blob["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), LESION_COLORS["exudate"], 2)
    heat = cv2.applyColorMap(cv2.addWeighted(red_mask, 0.6, yellow_mask, 0.6, 0), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.72, heat, 0.28, 0)
    cv2.imwrite(heatmap_path, overlay)

    lesion_summary = {
        "red_lesions": len(red_blobs),
        "yellow_lesions": len(yellow_blobs),
        "evidence": evidence,
        "model_mode": "trained_ensemble" if deep else "classical_fallback",
    }
    return {
        "dr_severity": severity,
        "risk_level": risk,
        "confidence": confidence,
        "lesion_summary": json.dumps(lesion_summary),
        "recommendation": recommendation,
        "ensemble": deep,
    }
