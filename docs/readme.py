"""Regenerate the figures embedded in the project README.

Run with ``uv run python -m docs.readme`` after changing the estimator or its
presentation. The committed figures under ``assets/`` are what the README
references, so this is a deliberate step rather than part of a docs build.
"""

import subprocess
import tempfile
from pathlib import Path

from qte.constants import DECILES
from qte.cross_sectional import estimate_aipw_qte
from qte.datasets import load_lalonde, load_mpdta
from qte.non_linear_did import estimate_changes_in_changes_for_panel
from qte.non_linear_did.custom_types import CounterfactualModel, TrtGroupConfig
from qte.results import _BasicQteResult


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
    """Estimate the README examples and regenerate their figures."""
    ds = load_lalonde(controls_source="psid", use_panel_structure=False)
    xf = "age + education + black + hispanic + married"
    res = estimate_aipw_qte(
        ds,
        "re78",
        "treat",
        qs=DECILES,
        or_x_formular=xf,
        ps_x_formular=xf,
        n_bootstrap_iter=50,
    )
    generate_readme_assets(res, "aipw_qte")

    ds = load_mpdta()
    res = estimate_changes_in_changes_for_panel(
        ds,
        "lemp",
        TrtGroupConfig("first.treat", 0),  # never-treated group is 0
        "year",
        "countyreal",
        qs=[0.25, 0.5, 0.75],
        counterfactual_model=CounterfactualModel.CIC,
        n_bootstrap_iter=50,
    )
    generate_readme_assets(res.overall, "cic")


if __name__ == "__main__":
    main()
