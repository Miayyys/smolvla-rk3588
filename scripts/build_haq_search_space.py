#!/usr/bin/env python3
"""Create a provisional all-listed-candidate HAQ action-space manifest."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qvla_haq.search_space import build_search_space  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=ROOT / "docs/hardware/tables.json")
    parser.add_argument("--output", type=Path, default=ROOT / "config/haq_action_space_v1.json")
    args = parser.parse_args()
    space = build_search_space(args.tables)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(space, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "status": space["status"],
        "search_ready": space["search_ready"],
        **space["counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
