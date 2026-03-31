from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

sys.path.append(str(Path(__file__).resolve().parents[0]))


def main() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
