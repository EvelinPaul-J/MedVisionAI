import uuid
from fastapi import APIRouter, HTTPException
from database import get_db
from schemas import PatientCreate

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("")
def create_patient(patient: PatientCreate):
    patient_id = "P" + uuid.uuid4().hex[:8].upper()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO patients
               (patient_id, name, age, sex, diabetes_type, diabetes_duration_years, hba1c)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (patient_id, patient.name, patient.age, patient.sex,
             patient.diabetes_type, patient.diabetes_duration_years, patient.hba1c)
        )
    return {"patient_id": patient_id}


@router.get("")
def list_patients():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


@router.get("/{patient_id}")
def get_patient(patient_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Patient not found")
        screenings = conn.execute(
            "SELECT * FROM screenings WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)
        ).fetchall()
        result = dict(row)
        result["screenings"] = [dict(s) for s in screenings]
        return result
