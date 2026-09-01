import os
import subprocess
import polars as pl

from qte.datasets import load_lalonde
from qte.cross_sectional import estimate_aipw_qte
from qte.constants import DECILES

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
        n_bootstrap_iter=50
    )

    print("Saving AIPW chart...")
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
        subprocess.run([
            "firefox", "--headless", "--screenshot", png_path, f"file://{html_path}"
        ], capture_output=True, check=True)
        
        os.remove(html_path)
        print("Table saved to assets/aipw_qte_table.png!")
    except Exception as e:
        print(f"Failed to save table: {e}")

    print("All artifacts generated successfully!")

if __name__ == "__main__":
    main()
