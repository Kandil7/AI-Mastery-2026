from __future__ import annotations

from fastapi import APIRouter

from app.schemas import EvalRequest, EvalResponse
from evaluation.harness import run_evaluation

router = APIRouter(prefix="/eval")


@router.post("/run")
def eval_run(request: EvalRequest) -> EvalResponse:
    run_id = run_evaluation(dataset_id=request.dataset_id, mode=request.mode)
    return EvalResponse(run_id=run_id, status="running")
