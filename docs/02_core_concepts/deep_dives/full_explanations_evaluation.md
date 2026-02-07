# Evaluation package (evaluation/*.py)

## evaluation/datasets.py

### EvalExample (dataclass)
- Fields:
  - `question: str`
  - `answer: str`

### load_dataset(dataset_id: str) -> List[EvalExample]
- Looks for `evaluation/{dataset_id}.jsonl`.
- Each line is parsed as JSON with `question` and `answer`.
- Returns an empty list if dataset file does not exist.

## evaluation/metrics.py

### EvalResult (dataclass)
- Fields:
  - `accuracy: float`
  - `faithfulness: float`
  - `citation_coverage: float`

### compute_metrics(exact_matches: int, total: int) -> EvalResult
- Computes accuracy as `exact_matches / total`.
- Returns zeros if `total == 0`.
- Faithfulness and citation coverage are placeholders (0.0).

## evaluation/harness.py

### run_evaluation(dataset_id: str, mode: str) -> str
- Loads dataset by ID.
- Runs `run_query_pipeline` for each example.
- Counts exact matches against reference answers (case-insensitive).
- Computes metrics (currently not returned).
- Returns a UUID run id.
