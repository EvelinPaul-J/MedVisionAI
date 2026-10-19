from pydantic import BaseModel
from typing import Optional


class PatientCreate(BaseModel):
    name: str
    age: int
    sex: str
    diabetes_type: str
    diabetes_duration_years: Optional[float] = 0
    hba1c: Optional[float] = None


class DoctorReview(BaseModel):
    doctor_status: str   # confirmed / rejected / needs_review
    doctor_notes: Optional[str] = ""
