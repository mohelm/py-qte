"""Regenerate the README and the assets it embeds.

Run with ``uv run python -m docs.readme`` after changing the estimator or its
presentation. The committed figures under ``assets/`` are what the README
references, and ``README_PYPI.md`` is generated from ``README.md`` with
repo-relative links made absolute, so this is a deliberate step rather than
part of a docs build.

Pass ``--pypi-readme-only`` to skip the slow, Firefox-dependent figure
generation and only refresh ``README_PYPI.md``.
"""

import argparse
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

from qte.constants import DECILES
from qte.cross_sectional import estimate_aipw_qte
from qte.datasets import load_lalonde, load_mpdta
from qte.nonlinear_did import estimate_nonlinear_did_for_panel
from qte.nonlinear_did.custom_types import CounterfactualModel, TrtGroupConfig
from qte.results import _BasicQteResult

_ROOT = Path(__file__).resolve().parent.parent

# Assets are referenced from PyPI as raw GitHub URLs on this branch.
_PYPI_README_BRANCH = "main"

# A link target that is not already absolute, root-relative, an anchor, or a
# scheme (http:, mailto:, ...) is treated as repository-relative.
_RELATIVE_PATH = r"""(?!#|/|[a-z][a-z0-9+.-]*:)[^\s)"']+"""
_HTML_LINK = re.compile(r'(?P<attr>\b(?:src|href))="(?P<path>' + _RELATIVE_PATH + r')"')
_MD_LINK = re.compile(r"\]\((?P<path>" + _RELATIVE_PATH + r")\)")


def _make_links_absolute(readme: str, raw_base: str) -> str:
    """Rewrite relative HTML/Markdown links in *readme* into *raw_base* URLs."""
    readme = _HTML_LINK.sub(lambda m: f'{m["attr"]}="{raw_base}/{m["path"]}"', readme)
    return _MD_LINK.sub(lambda m: f"]({raw_base}/{m['path']})", readme)


def generate_pypi_readme() -> None:
    """Write ``README_PYPI.md``: the README with absolute asset URLs.

    PyPI renders the long description in isolation, so repo-relative paths
    (``src="assets/..."``) show up broken there.
    """
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    repo = pyproject["project"]["urls"]["Repository"]
    raw_repo = repo.replace("github.com", "raw.githubusercontent.com")
    raw_base = f"{raw_repo}/{_PYPI_README_BRANCH}"

    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    (_ROOT / "README_PYPI.md").write_text(_make_links_absolute(readme, raw_base), encoding="utf-8")


def generate_readme_assets(
    res: _BasicQteResult, model_id: str, out_dir: str | Path = "assets"
) -> None:
    """Save the README plot and table for a fitted result.

    Parameters
    ----------
    res : _BasicQteResult
        Fitted result exposing ``plot`` and ``tabulate``.
    model_id : str
        Stem for the artifact names. ``"aipw_qte"`` writes ``aipw_qte_results.svg``
        and ``aipw_qte_table.png``.
    out_dir : str or Path, optional
        Directory to write the artifacts to.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    res.plot().save(out / f"{model_id}_results.svg")

    # Great Tables has no PNG export without selenium, so render its HTML in a
    # headless Firefox and crop the surrounding whitespace with Pillow. A
    # throwaway profile is required so an already running Firefox does not
    # swallow the screenshot request.
    from PIL import Image, ImageChops

    html_path = (out / f"{model_id}_table.html").resolve()
    png_path = (out / f"{model_id}_table.png").resolve()
    html_path.write_text(res.tabulate().as_raw_html())

    with tempfile.TemporaryDirectory(prefix="qte-readme-firefox-") as profile:
        subprocess.run(
            [
                "firefox",
                "--headless",
                "--profile",
                profile,
                "--window-size=1200,800",
                "--screenshot",
                str(png_path),
                f"file://{html_path}",
            ],
            check=True,
            capture_output=True,
        )

    img = Image.open(png_path)
    if img.mode in ("RGBA", "LA"):
        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
        background.paste(img, img.split()[-1])
        img = background
    else:
        img = img.convert("RGB")
    bbox = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255))).getbbox()
    if bbox:
        img = img.crop(
            (
                max(0, bbox[0] - 20),
                bbox[1],
                min(img.width, bbox[2] + 20),
                min(img.height, bbox[3] + 20),
            )
        )
        img.save(png_path)

    html_path.unlink()


def main() -> None:
    """Regenerate ``README_PYPI.md`` and, unless asked otherwise, the figures."""
    parser = argparse.ArgumentParser(description="Regenerate README artifacts.")
    parser.add_argument(
        "--pypi-readme-only",
        action="store_true",
        help="only regenerate README_PYPI.md (skip the figures)",
    )
    args = parser.parse_args()

    generate_pypi_readme()
    if args.pypi_readme_only:
        return

    ds = load_lalonde(controls_source="psid", use_panel_structure=False)
    xf = "age + education + black + hispanic + married"
    res = estimate_aipw_qte(
        ds,
        "re78",
        "treat",
        qs=DECILES,
        or_x_formular=xf,
        ps_x_formular=xf,
        bootstrap_config=50,
    )
    generate_readme_assets(res, "aipw_qte")

    ds = load_mpdta()
    res = estimate_nonlinear_did_for_panel(
        ds,
        "lemp",
        TrtGroupConfig("first.treat", 0),  # never-treated group is 0
        "year",
        "countyreal",
        qs=[0.25, 0.5, 0.75],
        counterfactual_model=CounterfactualModel.CIC,
        bootstrap_config=50,
    )
    generate_readme_assets(res.overall, "nonlinear_did")


if __name__ == "__main__":
    main()
