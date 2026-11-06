from __future__ import annotations

from app.core.app import app
from app.services.data import get_store

@app.on_event("startup")
def _warm() -> None:
    get_store()


