from __future__ import annotations

from fastapi import APIRouter

from core.settings import load_settings

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    settings = load_settings()
    return {"status": "ok", "version": settings.app_version}
