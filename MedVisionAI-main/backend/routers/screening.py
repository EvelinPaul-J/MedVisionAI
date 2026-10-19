import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from database import get_db
from schemas import DoctorReview
from ml.image_quality import assess_quality
from ml.retina_analyzer import analyze_retina

router = APIRouter(prefix="/api/screening", tags=["screening"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_and_analyze(patient_id: str = Form(...), file: UploadFile = File(...)):
    with get_db() as conn:
        exists = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    if not exists:
        raise HTTPException(404, "Patient not found. Register the patient first.")

    # Validate the uploaded media before writing it to disk. This also makes
    # uploads exported by a fundus-camera app more predictable.
    allowed_types = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    content_type = (file.content_type or "").lower()
    ext = allowed_types.get(content_type, ".jpg")
    if content_type not in allowed_types:
        # Some browsers/device drivers send an empty or generic content type.
        # We still save it and let OpenCV/Pillow validate the actual image.
        if content_type and not content_type.startswith("image/"):
            raise HTTPException(415, "Unsupported file. Please choose a JPG, PNG, or WEBP fundus image.")

    screening_id = "S" + uuid.uuid4().hex[:8].upper()
    original_path = os.path.join(UPLOAD_DIR, f"{screening_id}_original{ext}")
    heatmap_path = os.path.join(UPLOAD_DIR, f"{screening_id}_heatmap.jpg")

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 1 — Image Quality Gate
    quality = assess_quality(original_path)

    # A valid image should continue through the demo pipeline even when its
    # quality score is low. Only an unreadable/non-image file stops analysis.
    if quality.get("status") == "invalid":
        with get_db() as conn:
            conn.execute(
                """INSERT INTO screenings
                   (screening_id, patient_id, original_image_path, quality_score, quality_status,
                    dr_severity, risk_level, recommendation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (screening_id, patient_id, original_path, 0, "invalid",
                 "Inconclusive", "recapture", quality["message"])
            )
        return {
            "screening_id": screening_id,
            "quality": quality,
            "analysis_skipped": True,
            "message": quality["message"],
        }

    # Step 2 — Whole Retina AI Analysis (see ml/retina_analyzer.py)
    result = analyze_retina(original_path, heatmap_path)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO screenings
               (screening_id, patient_id, original_image_path, heatmap_image_path,
                quality_score, quality_status, dr_severity, risk_level, confidence,
                lesion_summary, recommendation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (screening_id, patient_id, original_path, heatmap_path,
             quality["overall_score"], quality["status"], result["dr_severity"],
             result["risk_level"], result["confidence"], result["lesion_summary"],
             result["recommendation"])
        )

    return {
        "screening_id": screening_id,
        "quality": quality,
        "analysis_skipped": False,
        "result": result,
    }


@router.get("/pending/sync")
def pending_sync():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM screenings WHERE synced = 0 ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

@router.get("/{screening_id}")
def get_screening(screening_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM screenings WHERE screening_id = ?", (screening_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Screening not found")
        return dict(row)


@router.get("")
def list_screenings(status: str = None):
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM screenings WHERE doctor_status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM screenings ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


@router.post("/{screening_id}/review")
def review_screening(screening_id: str, review: DoctorReview):
    with get_db() as conn:
        exists = conn.execute("SELECT 1 FROM screenings WHERE screening_id = ?", (screening_id,)).fetchone()
        if not exists:
            raise HTTPException(404, "Screening not found")
        conn.execute(
            "UPDATE screenings SET doctor_status = ?, doctor_notes = ? WHERE screening_id = ?",
            (review.doctor_status, review.doctor_notes, screening_id)
        )
    return {"status": "updated"}


@router.post("/{screening_id}/sync")
def sync_screening(screening_id: str):
    """Marks a locally-stored screening as synced.
    In a real deployment this endpoint would push the record to a
    central server when connectivity is detected; here it just flips
    the local flag so the offline-sync UI has something real to show."""
    with get_db() as conn:
        exists = conn.execute("SELECT 1 FROM screenings WHERE screening_id = ?", (screening_id,)).fetchone()
        if not exists:
            raise HTTPException(404, "Screening not found")
        conn.execute("UPDATE screenings SET synced = 1 WHERE screening_id = ?", (screening_id,))
    return {"status": "synced"}


@router.get("/pending/sync")
def pending_sync():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM screenings WHERE synced = 0 ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
