import os
import subprocess

from qte.constants import DECILES
from qte.cross_sectional import estimate_aipw_qte
from qte.datasets import load_lalonde


def main():
    print("Loading data...")
    ds = load_lalonde(controls_source="psid", use_panel_structure=False)
    xf = "age + education + black + hispanic + married"

    print("Estimating AIPW QTE (this might take a moment due to bootstrap)...")
    res = estimate_aipw_qte(
        ds,
        "re78",
        "treat",
        qs=DECILES,
        or_x_formular=xf,
        ps_x_formular=xf,
        n_bootstrap_iter=50,
    )

    print("Saving AIPW chart...")
    os.makedirs("assets", exist_ok=True)
    chart = res.plot()
    chart.save("assets/aipw_qte_results.svg")
    print("Chart saved to assets/aipw_qte_results.svg!")

    print("Saving AIPW table...")
    try:
        table = res.tabulate()
        html_content = table.as_raw_html()
        html_path = os.path.abspath("assets/aipw_qte_table.html")
        png_path = os.path.abspath("assets/aipw_qte_table.png")

        with open(html_path, "w") as f:
            f.write(html_content)

        print("Exporting table to PNG via Firefox...")
        try:
            os.remove(png_path)
        except FileNotFoundError:
            pass
        subprocess.run(
            [
                "firefox",
                "--headless",
                "--window-size=1200,800",
                "--screenshot",
                png_path,
                f"file://{html_path}",
            ],
            capture_output=True,
            check=True,
        )

        # Crop the white space using Pillow
        from PIL import Image, ImageChops

        img = Image.open(png_path)

        # Convert to RGB to safely evaluate white background
        if img.mode in ("RGBA", "LA"):
            background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
            background.paste(img, img.split()[-1])
            img = background
        else:
            img = img.convert("RGB")

        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()

        if bbox:
            # Add a padding around the table, but force Top padding to 0
            padded_bbox = (
                max(0, bbox[0] - 20),  # Left
                bbox[1],  # Top (No extra padding)
                min(img.width, bbox[2] + 20),  # Right
                min(img.height, bbox[3] + 20),  # Bottom
            )
            img = img.crop(padded_bbox)
            img.save(png_path)

        os.remove(html_path)
        print("Table saved to assets/aipw_qte_table.png!")
    except Exception as e:  # noqa
        print(f"Failed to save table: {e}")

    print("All artifacts generated successfully!")


if __name__ == "__main__":
    main()
