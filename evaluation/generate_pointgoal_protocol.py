#!/usr/bin/env python3
"""Generate or verify the deterministic Week 4 PointGoal EpisodeSpec manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.pointgoal_protocol import generate_week04_classical_v1


DEFAULT_OUTPUT = (
    Path(__file__).parent
    / "pointgoal_protocols"
    / "week04_classical_v1.json"
)


def rendered_protocol() -> str:
    return (
        json.dumps(generate_week04_classical_v1(), indent=2, ensure_ascii=False)
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the frozen Week 4 PointGoal manifest."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the existing file exactly matches deterministic generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing protocol file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    rendered = rendered_protocol()
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(f"protocol does not exist: {output}")
        actual = json.loads(output.read_text(encoding="utf-8"))
        expected = json.loads(rendered)
        generated_keys = (
            "generation",
            "selections",
            "episode_manifest_sha256",
            "episodes",
        )
        if any(actual.get(key) != expected.get(key) for key in generated_keys):
            raise ValueError(
                f"frozen EpisodeSpec manifest differs from deterministic generation: {output}"
            )
        print(f"frozen EpisodeSpec generation is reproducible: {output}")
        return 0
    if output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite existing protocol without --force: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote frozen EpisodeSpec manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
