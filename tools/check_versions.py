"""Check that the package version is consistent across the repository.

The version is declared in both ``pyproject.toml`` and ``meson.build``. This
script compares the two and, when ``--tag`` is given, also checks the release
tag. It is wired into the pre-commit hooks and the release workflow.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PROJECT_VERSION = re.compile(r"project\([^)]*version\s*:\s*'([^']+)'")


def pyproject_version() -> str:
    """Return the version declared in ``pyproject.toml``."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def meson_version() -> str:
    """Return the version declared in ``meson.build``."""
    text = (ROOT / "meson.build").read_text(encoding="utf-8")
    match = _PROJECT_VERSION.search(text)
    if match is None:
        raise SystemExit("Could not find a version in meson.build")
    return match.group(1)


def main() -> int:
    """Validate the versions and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag to check, e.g. v0.1.0a1")
    args = parser.parse_args()

    version = pyproject_version()
    problems = []

    meson = meson_version()
    if meson != version:
        problems.append(f"pyproject.toml has {version!r} but meson.build has {meson!r}")

    if args.tag is not None:
        tag = args.tag.removeprefix("v")
        if tag != version:
            problems.append(f"tag v{tag} does not match pyproject.toml version {version!r}")

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    print(f"version {version} is consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
