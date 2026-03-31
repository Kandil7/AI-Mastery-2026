from __future__ import annotations

from fastapi import APIRouter

from app.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback")


@router.post("")
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    _ = request
    return FeedbackResponse(status="ok")
