"""
database.py
Offline-first SQLite storage for RetinaReach AI.
No internet / external DB server required — everything runs locally.
"""
import sqlite3
import os
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "retinareach.db")


def init_db():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            sex TEXT,
            diabetes_type TEXT,
            diabetes_duration_years REAL,
            hba1c REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            screening_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            original_image_path TEXT,
            heatmap_image_path TEXT,
            quality_score REAL,
            quality_status TEXT,
            dr_severity TEXT,
            risk_level TEXT,
            confidence REAL,
            lesion_summary TEXT,       -- JSON string
            recommendation TEXT,
            doctor_status TEXT DEFAULT 'pending',   -- pending / confirmed / rejected / needs_review
            doctor_notes TEXT,
            synced INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    """)

    conn.commit()
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
