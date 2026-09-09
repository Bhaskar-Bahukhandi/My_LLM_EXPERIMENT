"""Inspect an authored configuration and emit a reviewable structural audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from unified_edge.config import ConfigError, load_config
from unified_edge.resolve import resolve_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output", type=Path, help="new JSON report; existing files are never overwritten"
    )
    args = parser.parse_args(argv)
    try:
        result = resolve_config(load_config(args.config))
        report = json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(report)
        print(report, end="")
        return {"WITHIN_TARGET": 0, "OUTSIDE_TARGET": 2, "NO_LEGAL_CANDIDATE": 3}[result.status]
    except (ConfigError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
