# run_api.py

## main() -> None
- Adds package path for imports when running as a script.
- Runs `uvicorn` with:
  - app target: `app.main:app`
  - host: `0.0.0.0`
  - port: `8000`
  - reload: `True` (dev mode)

## Script usage
- Run: `python research/week5-backend/week5_backend/run_api.py`
