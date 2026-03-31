"""
Arabic LLM - I/O Utilities

File I/O utilities for reading and writing various file formats.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Union


def read_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """
    Read JSONL file into list of dictionaries.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def write_jsonl(
    data: List[Dict],
    filepath: Union[str, Path],
    ensure_ascii: bool = False,
) -> None:
    """
    Write list of dictionaries to JSONL file.

    Args:
        data: List of dictionaries
        filepath: Path to output file
        ensure_ascii: Escape non-ASCII characters
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")


def read_json(filepath: Union[str, Path]) -> Dict:
    """
    Read JSON file into dictionary.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: Dict,
    filepath: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Write dictionary to JSON file.

    Args:
        data: Dictionary to write
        filepath: Path to output file
        indent: JSON indentation
        ensure_ascii: Escape non-ASCII characters
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_yaml(filepath: Union[str, Path]) -> Dict:
    """
    Read YAML file into dictionary.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_yaml_config(config_name: str) -> Dict:
    """
    Read YAML configuration from configs/ directory.

    Args:
        config_name: Configuration file name (e.g., 'training_config.yaml')

    Returns:
        Configuration dictionary
    """
    # Try relative to current directory
    config_path = Path("configs") / config_name

    if not config_path.exists():
        # Try relative to arabic_llm package
        import arabic_llm
        package_dir = Path(arabic_llm.__file__).parent
        config_path = package_dir.parent / "configs" / config_name

    return read_yaml(config_path)
