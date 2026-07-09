import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from moodmetrics.config import Config


def build_metric_table(metrics: dict[str, object]) -> Table:
    rows = [
        ["Métrique", "Valeur"],
        ["Échantillons train", metrics["train_samples"]],
        ["Échantillons test", metrics["test_samples"]],
        ["Accuracy", metrics["accuracy"]],
        ["Precision", metrics["precision"]],
        ["Recall", metrics["recall"]],
        ["F1-score", metrics["f1_score"]],
    ]
    table = Table(rows, colWidths=[7 * cm, 5 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_confusion_table(metrics: dict[str, object]) -> Table:
    values = metrics["confusion_matrix"]["values"]
    rows = [
        ["", "Prédit absent", "Prédit présent"],
        ["Réel absent", values[0][0], values[0][1]],
        ["Réel présent", values[1][0], values[1][1]],
    ]
    table = Table(rows, colWidths=[4 * cm, 4 * cm, 4 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (1, 1), (-1, -1), colors.HexColor("#eff6ff")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(2 * cm, 1.2 * cm, "MoodMetrics - Rapport d'évaluation")
    canvas.drawRightString(19 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(evaluation_path: str, output_path: str) -> Path:
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MoodMetricsTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=18,
    )
    section_style = ParagraphStyle(
        "MoodMetricsSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "MoodMetricsBody",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=10,
    )

    story = [
        Paragraph("MoodMetrics - Rapport d'évaluation", title_style),
        Paragraph(
            "Synthèse automatique des performances des classifieurs positif et négatif.",
            body_style,
        ),
        Paragraph(
            f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} avec "
            f"{evaluation['samples']} tweets annotés.",
            body_style,
        ),
    ]

    sections = [("positive", "Sentiment positif"), ("negative", "Sentiment négatif")]
    for index, (label, title) in enumerate(sections):
        metrics = evaluation["labels"][label]
        story.extend(
            [
                Paragraph(title, section_style),
                build_metric_table(metrics),
                Spacer(1, 12),
                Paragraph("Matrice de confusion", section_style),
                build_confusion_table(metrics),
                Spacer(1, 10),
            ]
        )
        if index < len(sections) - 1:
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="MoodMetrics - Rapport d'évaluation",
    )
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return destination


if __name__ == "__main__":
    generate_pdf_report(Config.EVALUATION_PATH, Config.REPORT_PATH)
