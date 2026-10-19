"""
main.py
RetinaReach AI — offline-first rural diabetic retinopathy screening app.
Run with:  uvicorn main:app --reload --port 8000
Then open: http://127.0.0.1:8000
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import init_db
from routers import patients, screening, reports

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

app = FastAPI(title="RetinaReach AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(patients.router)
app.include_router(screening.router)
app.include_router(reports.router)

# Serve uploaded/generated images (original + heatmap) so the frontend can show them
app.mount("/media", StaticFiles(directory=os.path.join(BASE_DIR, "uploads")), name="media")

# Serve the plain HTML/CSS/JS frontend — no build step needed
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/{page_name}.html")
def serve_page(page_name: str):
    path = os.path.join(FRONTEND_DIR, f"{page_name}.html")
    if os.path.exists(path):
        return FileResponse(path)
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "offline-capable"}
