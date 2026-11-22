"""Checks on the Quarto sources.

The documentation pages carry their title and engine in YAML front matter. If that block gets
mangled, Quarto silently drops the title and renders the page without one, so it is worth asserting
rather than noticing on the rendered site.
"""

from pathlib import Path

import pytest
import yaml

DOCS = Path(__file__).parent.parent / "docs"
PAGES = sorted(
    p
    for p in DOCS.rglob("*.qmd")
    if not p.name.startswith("_") and "reference" not in p.relative_to(DOCS).parts
)


def _front_matter(page: Path) -> dict:
    text = page.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{page} does not start with a front matter block"
    _, block, _ = text.split("---\n", 2)
    parsed = yaml.safe_load(block)
    assert isinstance(parsed, dict), f"{page} has front matter that is not a mapping"
    return parsed


def test_pages_are_found():
    assert len(PAGES) >= 6


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_has_front_matter_with_a_title(page: Path):
    front_matter = _front_matter(page)
    assert front_matter.get("title"), f"{page} has no title"
