"""Clone and verify the official Mem0 source used by mem0_local."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_REPO_PATH = HERE / "upstream" / "mem0"
LOCK_PATH = HERE / "UPSTREAM.json"


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def clone_upstream(
    repo_path: Path = DEFAULT_REPO_PATH,
    *,
    lock_path: Path = LOCK_PATH,
) -> dict[str, str]:
    lock = load_lock(lock_path)
    repository = str(lock["repository"])
    expected_commit = str(lock["commit"])

    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", str(repo_path)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "remote", "add", "origin", repository],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "fetch",
                "--depth",
                "1",
                "origin",
                expected_commit,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "checkout",
                "--detach",
                "FETCH_HEAD",
            ],
            check=True,
        )

    actual_commit = git_output(repo_path, "rev-parse", "HEAD")
    origin = git_output(repo_path, "remote", "get-url", "origin")
    status = git_output(repo_path, "status", "--porcelain")
    if actual_commit != expected_commit:
        raise RuntimeError(
            "Official Mem0 checkout does not match UPSTREAM.json: "
            f"expected={expected_commit} actual={actual_commit}"
        )
    if origin.rstrip("/") != repository.rstrip("/"):
        raise RuntimeError(
            "Unexpected Mem0 origin: "
            f"expected={repository} actual={origin}"
        )
    if status:
        raise RuntimeError(
            "Official Mem0 checkout contains local modifications; "
            "mem0_local requires a clean upstream tree."
        )

    package_root = repo_path / "mem0"
    if not (package_root / "memory" / "main.py").is_file():
        raise RuntimeError(f"Invalid Mem0 source checkout: {repo_path}")

    return {
        "repository": repository,
        "commit": actual_commit,
        "repo_path": str(repo_path.resolve()),
        "checkout_clean": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=Path, default=DEFAULT_REPO_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            clone_upstream(args.repo_path),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
