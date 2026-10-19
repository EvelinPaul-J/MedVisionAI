import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from database import get_db
from ml.report_generator import generate_report_pdf

router = APIRouter(prefix="/api/reports", tags=["reports"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "uploads")


@router.get("/{screening_id}")
def get_report(screening_id: str):
    with get_db() as conn:
        screening = conn.execute(
            "SELECT * FROM screenings WHERE screening_id = ?", (screening_id,)
        ).fetchone()
        if not screening:
            raise HTTPException(404, "Screening not found")
        screening = dict(screening)
        patient = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (screening["patient_id"],)
        ).fetchone()
        patient = dict(patient) if patient else {}

    out_path = os.path.join(REPORT_DIR, f"{screening_id}_report.pdf")
    generate_report_pdf(out_path, patient, screening)
    return FileResponse(out_path, media_type="application/pdf",
                         filename=f"RetinaReach_Report_{screening_id}.pdf")
