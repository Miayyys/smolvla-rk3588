#!/usr/bin/env python3
"""Sample unscored provisional HAQ precision configurations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qvla_haq.policy import RecurrentPolicyGradient  # noqa: E402
from qvla_haq.search_space import validate_assignment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=Path, default=ROOT / "config/haq_action_space_v1.json")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    space = json.loads(args.space.read_text())
    if space.get("search_ready") is not True:
        print("WARNING: space is provisional; these configurations are proposals only and will not be evaluated.",
              file=sys.stderr)
    policy = RecurrentPolicyGradient(space, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for index in range(args.count):
            assignment, _ = policy.sample()
            validate_assignment(space, assignment)
            encoded = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
            record = {
                "candidate_id": hashlib.sha256(encoded.encode()).hexdigest()[:16],
                "proposal_index": index,
                "action_space_sha256": space["source"]["tables_sha256"],
                "status": "provisional_unscored",
                "assignment": assignment,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"written": args.count, "output": str(args.output),
                      "status": "provisional_unscored"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
