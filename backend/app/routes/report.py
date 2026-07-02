from fastapi import APIRouter, HTTPException, Response, status

from backend.app.services.pdf_report import build_pdf
from backend.app.storage.scans import scan_store


router = APIRouter(tags=["reports"])


@router.get("/report/{scan_id}", response_class=Response)
def report(scan_id: str) -> Response:
    record = scan_store.get(scan_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found or expired.")
    return Response(
        content=build_pdf(record),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="dermascan-{scan_id}.pdf"'},
    )
