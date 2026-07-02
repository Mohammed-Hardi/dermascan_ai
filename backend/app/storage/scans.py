from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from backend.app.schemas import Explanation, PredictionItem


@dataclass(slots=True)
class ScanRecord:
    scan_id: str
    created_at: datetime
    image_bytes: bytes
    top_prediction: PredictionItem | None
    top_k: list[PredictionItem]
    confidence_level: str
    risk_level: str
    explanation: Explanation
    disclaimer: str
    model_version: str
    metadata: dict[str, str | None]


class ScanStore:
    """Process-local storage that avoids permanently retaining user images."""

    def __init__(self, max_records: int = 100) -> None:
        self._records: dict[str, ScanRecord] = {}
        self._order: deque[str] = deque()
        self._max_records = max_records
        self._lock = RLock()

    def create(self, **values: object) -> ScanRecord:
        with self._lock:
            scan_id = str(uuid4())
            record = ScanRecord(
                scan_id=scan_id,
                created_at=datetime.now(timezone.utc),
                **values,
            )
            self._records[scan_id] = record
            self._order.append(scan_id)
            while len(self._order) > self._max_records:
                expired_id = self._order.popleft()
                self._records.pop(expired_id, None)
            return record

    def get(self, scan_id: str) -> ScanRecord | None:
        with self._lock:
            return self._records.get(scan_id)


scan_store = ScanStore()
