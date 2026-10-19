"""
report_generator.py
Generates an AI-Assisted Retinal Screening Report as a PDF, entirely
offline (reportlab does not need internet access).
"""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_report_pdf(out_path: str, patient: dict, screening: dict):
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, spaceAfter=2)
    sub_style = ParagraphStyle("SubX", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=4)
    body = styles["Normal"]

    story = []
    story.append(Paragraph("RetinaReach AI", title_style))
    story.append(Paragraph("AI-Assisted Retinal Screening Report", sub_style))
    story.append(Spacer(1, 8))

    lesion_summary = json.loads(screening.get("lesion_summary") or "{}")

    patient_table = Table([
        ["Patient ID", patient.get("patient_id", "-"), "Age / Sex", f"{patient.get('age','-')} / {patient.get('sex','-')}"],
        ["Diabetes Type", patient.get("diabetes_type", "-"), "Duration (yrs)", str(patient.get("diabetes_duration_years", "-"))],
        ["Screening ID", screening.get("screening_id", "-"), "Date", screening.get("created_at", "-")],
    ], colWidths=[85, 130, 85, 130])
    patient_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3fb")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef3fb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Image Quality", h2))
    story.append(Paragraph(f"Status: {screening.get('quality_status','-')} (score {screening.get('quality_score','-')}/100)", body))

    story.append(Paragraph("AI Findings (Evidence)", h2))
    for line in lesion_summary.get("evidence", []):
        story.append(Paragraph(f"&bull; {line}", body))

    story.append(Paragraph("AI Screening Assessment", h2))
    assess_table = Table([
        ["DR Assessment", screening.get("dr_severity", "-")],
        ["Risk / Referral Priority", screening.get("risk_level", "-").upper()],
        ["Model Confidence", f"{screening.get('confidence','-')}%"],
    ], colWidths=[170, 260])
    assess_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3fb")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(assess_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Explainability", h2))
    if screening.get("heatmap_image_path"):
        try:
            story.append(RLImage(screening["heatmap_image_path"], width=180, height=180))
        except Exception:
            pass
    story.append(Paragraph("Highlighted regions indicate areas that most influenced the AI assessment "
                            "(microaneurysm-like = orange, hemorrhage-like = red, exudate-like = yellow).", body))

    story.append(Paragraph("Recommendation", h2))
    story.append(Paragraph(screening.get("recommendation", "-"), body))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "This is an AI-assisted screening output, not a clinical diagnosis. "
        "It is intended to support, not replace, ophthalmologist evaluation.",
        ParagraphStyle("Disclaimer", parent=styles["Italic"], fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    return out_path
