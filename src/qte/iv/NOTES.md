# Non-bootstrap standard errors for the local QTE

This note records whether the complier-quantile standard errors can be computed
analytically, and what it would take. It is a design note, not public API.

## Setup

For `d in {0, 1}` the complier distribution and its quantile are

```
F_d(y) = mu_d(y) / m_d,   Q_d(tau) = F_d^{-1}(tau),
mu_d(y) = E[w_d * 1{Y <= y}],   m_d = E[w_d],
```

with the Abadie / Froelich-Melly weight `w_d` depending on the instrument
propensity `p(X; gamma) = P(Z = 1 | X)` and `gamma` the logit coefficients. The
estimand is `LQTE(tau) = Q_1(tau) - Q_0(tau)`.

## Known propensity

If `gamma` were known, the influence function of the weighted quantile is
standard. The weighted empirical CDF has

```
IF_{F_d(y)} = w_d * (1{Y <= y} - F_d(y)) / m_d,
```

so, by the implicit-function theorem applied to `F_d(Q_d) = tau`,

```
IF_{Q_d}   = w_d * (tau - 1{Y <= Q_d}) / (m_d * f_d(Q_d)),
IF_{LQTE}  = IF_{Q_1} - IF_{Q_0},
Var(LQTE)  ~ (1 / n) * Var(IF_{LQTE}).
```

The only non-trivial plug-in pieces are `m_d` (a sample mean) and the density
`f_d(Q_d)`, which can be estimated with a signed-weight Gaussian kernel and a
Silverman bandwidth. Everything else is already computed.

## Why `gamma` cannot simply be ignored

`w_d` depends on `p(X; gamma_hat)`, so the influence function above is missing a
term. The correct two-step expansion is

```
IF_{Q_d} = -(1 / f_d) * ( IF_{F_d} + E[dF_d/dgamma] * IF_gamma ),
```

where, for the logit, `IF_gamma(O) = I(gamma)^{-1} (Z - p) X` with
`I(gamma) = E[p(1-p) X X']`, and

```
E[dF_d/dgamma] = (1 / m_d) * E[ (dw_d/dgamma) * (1{Y <= Q_d} - F_d) ].
```

This projection onto the propensity-score is **not zero**: the Abadie weights
are not orthogonal to the instrument model (unlike a doubly-robust moment). A
plug-in standard error that treats `p(X)` as fixed is therefore inconsistent,
and the correction is first-order, not a refinement. It can be computed from
sample means of `dw_d/dgamma`, which is available in closed form because
`p = logistic(X'gamma)`.

## Caveats

1. **Density.** `f_d(Q_d)` is a nuisance. With signed weights the kernel estimate
   can be negative, especially in the tails, and the whole expansion degrades
   exactly where the local QTE is most interesting. A bootstrap avoids this.
2. **Non-smoothness.** The quantile map is Hadamard-differentiable only when the
   density at the quantile is bounded away from zero; the IF variance is finite
   under that condition but is a poor approximation near points of flat density.
3. **Weak instruments.** If the first stage is weak the complier quantiles are
   only set-identified and the estimator has non-standard asymptotics. No normal
   interval, bootstrap or analytic, is valid there; Anderson-Rubin /
   Chernozhukov-Hansen weak-IV-robust inference is needed instead. The
   bootstrap is also not uniformly consistent in that regime.
4. **Sign correction.** The complier CDF is estimated with negative weights and
   need not be monotone. If the estimate is monotonised before inversion, the
   limit changes and the IF above no longer applies to the monotonised version.

## Recommendation

- Keep the bootstrap as the default (as implemented). The cross-section is
  i.i.d., so the nonparametric bootstrap is valid away from weak instruments.
- An analytic option is feasible: implement the score-corrected IF above, and
  validate it against the bootstrap on `simulate_data`. It is cheap
  after the point estimate and needs no resampling.
- Use analytic standard errors only in the interior of the distribution and
  with an explicit overlap / trimming rule.
- For weak instruments, neither route is enough; use AR-type inference
  (cf. the `ivmodels` Python package for the mean case and Chernozhukov-Hansen
  for IVQR).

## References

- Abadie, A. (2003). Semiparametric instrumental variable estimation of
  treatment response models. *Journal of Econometrics*.
- Abadie, A., Angrist, J. & Imbens, G. (2002). Instrumental variables estimates
  of the effect of subsidized training on the quantiles of trainee earnings.
  *Econometrica*.
- Newey, W. K. & McFadden, D. (1994). Large sample estimation and hypothesis
  testing. *Handbook of Econometrics*, Vol. 4 (influence functions for quantiles
  and two-step estimators).
- Chernozhukov, V. & Hansen, C. (2006, 2008). Instrumental quantile regression
  inference / robust inference. *Journal of Econometrics*.
