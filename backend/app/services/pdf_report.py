from io import BytesIO

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.storage.scans import ScanRecord


def build_pdf(record: ScanRecord) -> bytes:
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"DermaScan AI report {record.scan_id}",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#102A43"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Disclaimer",
            parent=styles["BodyText"],
            textColor=colors.HexColor("#8C1D18"),
            borderColor=colors.HexColor("#E3A6A1"),
            borderWidth=1,
            borderPadding=8,
            leading=14,
        )
    )

    story = [
        Paragraph("DermaScan AI Screening Report", styles["ReportTitle"]),
        Paragraph(f"Scan ID: {record.scan_id}", styles["BodyText"]),
        Paragraph(f"Date: {record.created_at.strftime('%Y-%m-%d %H:%M UTC')}", styles["BodyText"]),
        Paragraph(f"Model: {record.model_version}", styles["BodyText"]),
        Spacer(1, 8 * mm),
    ]

    thumbnail_stream = BytesIO()
    with Image.open(BytesIO(record.image_bytes)) as image:
        image.thumbnail((900, 600))
        image.save(thumbnail_stream, format="JPEG", quality=85)
    thumbnail_stream.seek(0)
    story.extend([ReportImage(thumbnail_stream, width=70 * mm, height=46 * mm), Spacer(1, 7 * mm)])

    if record.top_prediction is None:
        result_text = "Uncertain result"
    else:
        result_text = (
            f"{record.top_prediction.class_name.title()} "
            f"({record.top_prediction.confidence * 100:.1f}% confidence)"
        )
    story.extend(
        [
            Paragraph("AI screening result", styles["Heading2"]),
            Paragraph(result_text, styles["BodyText"]),
            Spacer(1, 4 * mm),
        ]
    )

    table_data = [["Possible condition", "Confidence"]]
    table_data.extend(
        [[item.class_name.title(), f"{item.confidence * 100:.1f}%"] for item in record.top_k]
    )
    table = Table(table_data, colWidths=[115 * mm, 35 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 7 * mm),
            Paragraph("AI consultant explanation", styles["Heading2"]),
            Paragraph(record.explanation.summary, styles["BodyText"]),
            Paragraph(record.explanation.next_steps, styles["BodyText"]),
            Paragraph(record.explanation.warning, styles["BodyText"]),
            Spacer(1, 7 * mm),
            Paragraph(record.disclaimer, styles["Disclaimer"]),
        ]
    )
    document.build(story)
    return output.getvalue()
