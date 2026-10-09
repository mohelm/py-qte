# qte — Quantile Treatment Effects in Python

> **Alpha software:** the API may change without notice between releases.

This is a Python implementation of a collection of quantile treatment effect estimators inspired by
the R [`qte`](https://github.com/bcallaway11/qte) package and the Stata
[`ivqte`](https://sites.google.com/site/blaisemelly/home/computer-programs/estimation-of-quantile-treatment-effects-in-stata?authuser=0)
package. See the [project website](https://mohelm.github.io/py-qte/) for the API reference and
benchmarks.

The available estimators are

- **Cross-sectional quantile treatment effects** — unadjusted, inverse probability weighted, outcome
  regression and doubly robust;
- **nonlinear difference-in-differences** — changes-in-changes and quantile
  difference-in-differences; and
- **instrumental variables** — local quantile treatment effects for compliers.

Other features:

- **Fast**: `py-qte` uses more efficient algorithms and parallelizes the bootstrap standard errors
  effectively. On larger datasets it is more than 20× faster than `qte` for the AIPW estimator of
  the quantile treatment effect (see the
  [benchmark timings](https://mohelm.github.io/py-qte/benchmark.html#tbl-aipw-timings)).
- **Beautiful**: Graphs and tables for the console, the web, and LaTeX powered by
  [Altair](https://github.com/vega/altair),
  [Great Tables](https://github.com/posit-dev/great-tables) and
  [Rich](https://github.com/textualize/rich).

---

# Installation

Requires Python 3.12 or newer.

```bash
uv add py-qte # or pip install py-qte
```

On Linux and macOS the Fortran solver for quantile regression should work and be used in the
computations. On Windows a fallback (based on the `statsmodels` library) will be used.

---

# Examples

## Cross-Sectional Data

We can estimate quantile treatment (and average) treatment effects using an augmented inverse
probability weighting (AIPW) estimator.

```python
from qte.cross_sectional import estimate_aipw_effects
from qte.datasets import load_lalonde

ds = load_lalonde()

res = estimate_aipw_effects(
    ds=ds,
    outcome="re78",
    treatment="treat",
    outcome_regression_config="age + education",
    propensity_score_formula="age + education",
)

res.plot()  # Vega-Altair plot (see below)
res.tabulate()  # Great Tables output (see below)
```

<table width="100%" border="0" cellpadding="0" cellspacing="0">
  <tr>
    <td width="48%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/mohelm/py-qte/main/assets/aipw_qte_results.svg" width="100%">
    </td>
    <td width="4%"></td>
    <td width="48%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/mohelm/py-qte/main/assets/aipw_qte_table.png" width="100%">
    </td>
  </tr>
</table>

## Nonlinear Difference-in-Differences

We can estimate quantile treatment (and average) treatment effects with the changes-in-changes
estimator.

```python
from qte.nonlinear_did import (
    CounterfactualModel,
    TrtGroupConfig,
    estimate_nonlinear_did_for_panel,
)
from qte.datasets import load_mpdta

ds = load_mpdta()
res = estimate_nonlinear_did_for_panel(
    ds,
    "lemp",
    TrtGroupConfig("first.treat", 0),  # never-treated group is 0
    "year",
    "countyreal",
    qs=[0.25, 0.5, 0.75],
    counterfactual_model=CounterfactualModel.CIC,
)
```

<table width="100%" border="0" cellpadding="0" cellspacing="0">
  <tr>
    <td width="48%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/mohelm/py-qte/main/assets/nonlinear_did_results.svg" width="100%">
    </td>
    <td width="4%"></td>
    <td width="48%" valign="top" align="center">
      <img src="https://raw.githubusercontent.com/mohelm/py-qte/main/assets/nonlinear_did_table.png" width="100%">
    </td>
  </tr>
</table>

## Instrumental Variables

When the treatment is endogenous, a valid instrument identifies effects for compliers. Here we use
college proximity (`nearc4`) as an instrument for college completion (`college`) in the Card data,
following Abadie's kappa weighting.

```python
from qte.datasets import load_card
from qte.iv import estimate_local_effects

ds = load_card()

res = estimate_local_effects(
    ds,
    outcome="lwage",
    treatment="college",
    instrument="nearc4",
    qs=[0.1, 0.25, 0.5, 0.75, 0.9],
)

res.plot()  # Vega-Altair plot
res.tabulate()  # Great Tables output
```

---

# Development

Development is somewhat complicated due to the inclusion of the Fortran code for solving quantile
regression. A working way to set up the project for development is:

```sh
git clone git@github.com:mohelm/py-qte.git
cd py-qte
uv sync --all-groups --no-install-project && uv sync --all-groups
```

## Website

See [`docs/README.md`](https://github.com/mohelm/py-qte/blob/main/docs/README.md) for how to build
and preview the Quarto website and regenerate the API reference.

## Regenerating the README assets

Regenerate the figures above after changing the estimators or their presentation:

```sh
uv run python -m docs.readme
```

`README_PYPI.md` (what `project.readme` points at) is generated from this file with relative links
made absolute, since PyPI can't resolve repo-relative assets. Regenerate it after editing this file:

```sh
uv run python -m docs.readme --pypi-readme-only
```

## Releasing

See [`RELEASING.md`](https://github.com/mohelm/py-qte/blob/main/RELEASING.md) for how to cut a
release.

---

# Credits

The cross-sectional and nonlinear difference-in-differences estimators reimplement the R
[`qte`](https://github.com/bcallaway11/qte) package by Brantly Callaway. The instrumental-variables
estimator roughly follows the Stata
[`ivqte`](https://sites.google.com/site/blaisemelly/home/computer-programs/estimation-of-quantile-treatment-effects-in-stata?authuser=0)
package by Frölich and Melly. The bundled datasets are the LaLonde (1986), Card (1995), Abadie,
Angrist & Imbens (2002) and Engel (1857) datasets, together with the county teen-employment data
used in the `did` and `qte` R packages.
