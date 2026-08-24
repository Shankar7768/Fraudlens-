"""Input/Output helper functions for JSON Lines (JSONL) files."""
import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union


def stream_jsonl(file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """Streams records one by one from a JSON Lines file.

    Args:
        file_path: Path to the .jsonl file.

    Yields:
        Dictionary parsed from each line.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                yield json.loads(line_str)
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse line {line_num} in {path}: {err}") from err


def read_jsonl(file_path: Union[str, Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Reads all or up to `limit` records from a JSON Lines file into a list.

    Args:
        file_path: Path to the .jsonl file.
        limit: Optional maximum number of lines to load.

    Returns:
        List of dictionaries.
    """
    records: List[Dict[str, Any]] = []
    for record in stream_jsonl(file_path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def write_jsonl(
    records: Iterable[Dict[str, Any]],
    file_path: Union[str, Path],
    mode: str = "w",
) -> int:
    """Writes an iterable of records strictly as a JSON Lines file.

    Args:
        records: Iterable of serializable dictionaries.
        file_path: Path to output .jsonl file.
        mode: File open mode ('w' for overwrite, 'a' for append).

    Returns:
        Total number of records written.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count
