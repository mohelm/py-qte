from qte.constants import DECILES
from qte.cross_sectional import estimate_aipw_qte
from qte.datasets import load_lalonde

print("Loading data...")
ds = load_lalonde(controls_source="psid", use_panel_structure=False)
xf = "age + education + black + hispanic + married"

print("Estimating QTE (this might take a moment due to bootstrap)...")
res = estimate_aipw_qte(
    ds,
    "re78",
    "treat",
    qs=DECILES,
    or_x_formular=xf,
    ps_x_formular=xf,
    n_bootstrap_iter=50,
)

print("Saving chart...")
chart = res.plot()
chart.save("assets/aipw_qte_results.svg")
print("Done!")
