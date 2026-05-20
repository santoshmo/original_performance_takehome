#!/usr/bin/env python3
"""Create isolated git worktrees for structural TTC experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(
    args: list[str],
    cwd: Path = REPO_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True)


def ensure_worktree(
    name: str,
    base_ref: str,
    worktree_root: Path,
    recreate: bool,
) -> tuple[Path, str]:
    worktree_path = worktree_root / name
    branch = f"ttc/{name}"
    if worktree_path.exists():
        if recreate:
            run(["git", "worktree", "remove", "--force", str(worktree_path)])
        else:
            return worktree_path, branch

    worktree_root.mkdir(parents=True, exist_ok=True)
    existing_branches = run(["git", "branch", "--format=%(refname:short)"]).stdout.splitlines()
    if branch in existing_branches:
        run(["git", "worktree", "add", str(worktree_path), branch])
    else:
        run(["git", "worktree", "add", "-b", branch, str(worktree_path), base_ref])
    return worktree_path, branch


def append_metadata(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Experiment/worktree name")
    parser.add_argument("--base-ref", default="HEAD", help="Base ref for new worktree branch")
    parser.add_argument(
        "--worktree-root",
        default="../ttc-worktrees",
        help="Directory where worktrees are created",
    )
    parser.add_argument(
        "--command",
        help="Optional command to run inside the worktree after creation",
    )
    parser.add_argument(
        "--metadata-output",
        default="ttc-results/worktrees.jsonl",
        help="JSONL file for experiment metadata in the main repo",
    )
    parser.add_argument("--recreate", action="store_true", help="Recreate the worktree if present")
    args = parser.parse_args()

    worktree_root = (REPO_ROOT / args.worktree_root).resolve()
    worktree_path, branch = ensure_worktree(args.name, args.base_ref, worktree_root, args.recreate)

    command_result = None
    if args.command:
        completed = subprocess.run(
            args.command,
            cwd=worktree_path,
            shell=True,
            text=True,
            capture_output=True,
        )
        command_result = {
            "command": args.command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            print(f"Command failed with exit code {completed.returncode}", file=sys.stderr)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "branch": branch,
        "base_ref": args.base_ref,
        "worktree_path": str(worktree_path),
        "command": command_result,
    }
    append_metadata(REPO_ROOT / args.metadata_output, record)

    print(f"worktree={shlex.quote(str(worktree_path))}")
    print(f"branch={branch}")
    if command_result is not None and command_result["returncode"] != 0:
        raise SystemExit(command_result["returncode"])


if __name__ == "__main__":
    main()
