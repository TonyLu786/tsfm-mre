# Methodology

This document defines the forecasting design, the prediction schema and every statistic
reported by TSFM-MRE. Equations refer to the implementation in `src/tsfm_mre`.

## 1. Forecasting design

Let $y_{i,t}$ be the yield (percent) of series $i$ on trading day $t$, observed on a
calendar that contains only the dates on which every series is available. For an origin
$t$ and a horizon $h \in \{1, 5, 10, 20\}$ trading days, a model issues the predictive
distribution $F_{i,t+h\mid t}$ of $y_{i,t+h}$ using only $\{y_{i,s} : s \le t\}$.

Origins run over every trading day from the evaluation start to the last day whose
target lies inside the sample. Models are re-estimated at every origin (expanding or
rolling windows), so every evaluated forecast is out of sample. The test suite checks
the absence of look-ahead: perturbing data after an origin leaves that origin's forecast
unchanged for every model.

## 2. Prediction schema

A forecast is one row with keys `model`, `series`, `origin`, `horizon` and `target_date`,
the predictive mean, the quantiles $q_\tau$ at

$$
\tau \in \{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, \dots, 0.9, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999\},
$$

and the expected shortfall of both tails,

$$
\mathrm{ES}_\tau = \mathbb{E}\left[\,Y \mid Y \le q_\tau\right] \ (\tau < 0.5), \qquad
\mathrm{ES}_\tau = \mathbb{E}\left[\,Y \mid Y \ge q_\tau\right] \ (\tau > 0.5),
$$

for $\tau \in \{0.01, 0.025, 0.975, 0.99\}$. Between grid points the predictive quantile
function is linear. The column conventions and validation rules are listed in
[`prediction_schema.md`](prediction_schema.md).

## 3. Scoring rules and calibration

**Quantile loss.** For level $\tau$, $\mathrm{QS}_\tau(q, y) = (\mathbf{1}\{y < q\} - \tau)(q - y)$,
which is strictly consistent for the $\tau$-quantile.

**CRPS.** The continuous ranked probability score, a strictly proper scoring rule
(Gneiting and Raftery, 2007), has the quantile representation (Laio and Tamea, 2007;
Gneiting and Ranjan, 2011)

$$
\mathrm{CRPS}(F, y) = 2 \int_0^1 \mathrm{QS}_\tau\!\left(F^{-1}(\tau), y\right) d\tau,
$$

evaluated with the midpoint rule over the quantile grid: level $\tau_k$ receives the
weight of the interval between the midpoints to its neighbours, with end points 0 and 1.
As a positively weighted sum of quantile scores, the result is a consistent scoring
function for the quantiles on the grid. For Gaussian forecasts it reproduces the closed-form CRPS to within one
percent (`tests/test_scoring.py`).

**Tail quantile loss.** $\tfrac12\left[\mathrm{QS}_{\alpha}(q_\alpha, y) + \mathrm{QS}_{1-\alpha}(q_{1-\alpha}, y)\right]$
with $\alpha$ equal to the VaR level (1%). It scores exactly the two quantiles that the
VaR backtests examine.

**Coverage and PIT.** Interval coverage is the share of outcomes inside the central 80%,
90% and 98% predictive intervals. The probability integral transform is $u = F(y)$,
where $F$ interpolates the forecast's quantile grid linearly. H.15 yields are published
in hundredths of a percentage point. For such rounded observations the randomised PIT
draws $u$ uniformly between $F(y - r/2)$ and $F(y + r/2)$ with $r = 0.01$, which is
uniformly distributed for a calibrated forecast (Czado, Gneiting and Held, 2009). PIT
histograms diagnose dispersion: a hump indicates overly wide distributions, a U shape
overly narrow ones (Gneiting, Balabdaoui and Raftery, 2007).

## 4. Risk backtests

Risk is assessed for the position exposed to each tail of the yield distribution. With
origin value $y_t$ and realised value $y_{t+h}$, the loss is

$$
L^{\uparrow}_{t+h} = y_{t+h} - y_t \quad \text{(rising yields, the loss of a long-duration position)},\qquad
L^{\downarrow}_{t+h} = y_t - y_{t+h} \quad \text{(falling yields)}.
$$

At level $\alpha$, the upper-tail VaR and ES in loss units are $q_{1-\alpha} - y_t$ and
$\mathrm{ES}_{1-\alpha} - y_t$; the lower-tail ones are $y_t - q_\alpha$ and
$y_t - \mathrm{ES}_\alpha$. The reference configurations use the levels of the Basel
framework: one-day VaR at 99% ($\alpha = 1\%$), as in the backtesting framework (Basel
Committee on Banking Supervision, 1996), and ES at 97.5% ($\alpha = 2.5\%$), as in the
revised market-risk standard (Basel Committee on Banking Supervision, 2019).

**Unconditional coverage** (Kupiec, 1995). With $x$ exceedances $I_t = \mathbf{1}\{L_t > \mathrm{VaR}_t\}$
in $n$ forecasts and $\hat p = x / n$,

$$
\mathrm{LR}_{uc} = -2\left[(n - x)\log(1 - \alpha) + x \log \alpha - (n - x)\log(1 - \hat p) - x \log \hat p\right] \sim \chi^2_1 .
$$

**Independence and conditional coverage** (Christoffersen, 1998). With transition counts
$n_{ij}$ of the exceedance sequence and $\hat\pi_{01} = n_{01}/(n_{00}+n_{01})$,
$\hat\pi_{11} = n_{11}/(n_{10}+n_{11})$, $\hat\pi = (n_{01}+n_{11})/\sum n_{ij}$,

$$
\mathrm{LR}_{ind} = -2\left[\ell(\hat\pi) - \ell(\hat\pi_{01}, \hat\pi_{11})\right] \sim \chi^2_1, \qquad
\mathrm{LR}_{cc} = \mathrm{LR}_{uc} + \mathrm{LR}_{ind} \sim \chi^2_2 ,
$$

where $\ell$ denotes the Bernoulli (Markov) log-likelihood.

**Traffic-light zones** (Basel Committee on Banking Supervision, 1996). For a window of
$n$ forecasts, the yellow zone starts at the smallest count $k$ with
$P(\mathrm{Bin}(n, \alpha) \le k) \ge 0.95$ and the red zone at the smallest count with
probability at least 0.9999. For $n = 250$ and $\alpha = 1\%$ this gives the familiar
boundaries of 5 and 10 exceedances; other windows and levels follow from the same rule.

**Expected shortfall** (Acerbi and Székely, 2014). The test uses the statistic

$$
Z_2 = 1 - \frac{1}{T\alpha} \sum_{t=1}^{T} \frac{L_t\, I_t}{\mathrm{ES}_t}, \qquad I_t = \mathbf{1}\{L_t > \mathrm{VaR}_t\},
$$

whose expectation is zero when the ES forecasts are correct; negative values indicate
underestimated tail risk. The statistic is defined for $\mathrm{ES}_t > 0$ in loss units,
which holds at the one-day horizon of the reference configurations. The p-value is
obtained by simulation under the null hypothesis that each loss is drawn from its own
predictive distribution. In each of $M$ replications every forecast exceeds its VaR with
probability $\alpha$, and an exceeding loss is drawn from the predictive tail,
reconstructed from the forecast's quantile grid. The p-value is
$\left(1 + \#\{Z_2^{(m)} \le Z_2\}\right) / (M + 1)$. The test suite checks the size of
the test under correct forecasts and its power when tail risk is underestimated.

## 5. Model comparison

Losses are CRPS and tail quantile loss. Comparisons are run for each series and for the
series-pooled loss, defined as the average loss over series at each origin.

**Diebold–Mariano** (Diebold and Mariano, 1995). For the loss differential
$d_t = L^{(a)}_t - L^{(b)}_t$,

$$
\mathrm{DM} = \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}\;\frac{\bar d}{\sqrt{\hat\omega^2 / n}},
$$

where $\hat\omega^2$ is the Newey–West (1987) long-run variance with
$\max\!\left(h - 1, \lfloor 4 (n/100)^{2/9} \rfloor\right)$ lags (Newey and West, 1994),
and the scaling factor is the small-sample adjustment of Harvey, Leybourne and Newbold
(1997). The statistic is compared with a $t_{n-1}$ distribution; positive values favour
the benchmark $b$.

**Reality Check and SPA.** With benchmark loss $L^{(0)}_t$ and competitor losses
$L^{(k)}_t$, let $d_{k,t} = L^{(0)}_t - L^{(k)}_t$ and $\bar d_k$ its mean. The null
hypothesis is $\max_k \mathbb{E}[d_{k,t}] \le 0$: no competitor beats the benchmark.
White's (2000) Reality Check uses $T^{RC} = \max_k \sqrt{n}\,\bar d_k$. Hansen's (2005)
SPA test studentises each competitor,

$$
T^{SPA} = \max\!\left(0, \max_k \frac{\sqrt{n}\,\bar d_k}{\hat\omega_k}\right),
$$

and recentres the bootstrap distribution at
$\hat\mu^c_k = \bar d_k\,\mathbf{1}\{\sqrt{n}\,\bar d_k/\hat\omega_k \le -\sqrt{2\log\log n}\}$,
so that clearly inferior models do not inflate the p-value. The lower and upper variants
($\hat\mu^l_k = \min(\bar d_k, 0)$ and $\hat\mu^u_k = 0$) are reported alongside the
consistent one. Bootstrap samples are drawn with the stationary bootstrap of Politis and
Romano (1994), with geometric block lengths of mean 10 in the reference runs;
$\hat\omega_k^2$ is the bootstrap variance of $\sqrt{n}\,\bar d_k$.

## 6. Baseline models

**Random walk with EWMA volatility** (`rw_ewma`). With $v_t = \lambda v_{t-1} + (1-\lambda)(\Delta y_t)^2$,
$\lambda = 0.94$ (J.P. Morgan/Reuters, 1996) and $v$ initialised at the mean squared
change of the first 60 days, $y_{t+h} \mid \mathcal F_t \sim N(y_t, h\, v_t)$.

**AR(1)** (`ar1`). $y_s = c + \phi\, y_{s-1} + \varepsilon_s$ with Gaussian errors,
estimated by least squares on all data up to the origin (or on a rolling window), with
$\sigma^2$ the residual variance on $n - 2$ degrees of freedom. The $h$-step distribution
is exact under the fitted model:

$$
y_{t+h} \mid \mathcal F_t \sim N\!\left(c \sum_{j=0}^{h-1} \phi^j + \phi^h y_t,\;
\sigma^2 \sum_{j=0}^{h-1} \phi^{2j}\right).
$$

**Historical simulation** (`hs`). The predictive distribution of $y_{t+h}$ is $y_t$ plus
the empirical distribution of the last 500 observed $h$-day changes
$y_s - y_{s-h}$, $s = t - 499, \dots, t$. Quantiles interpolate linearly between order
statistics; the lower-tail ES at level $\tau$ is the mean of the $\lceil \tau n \rceil$
smallest outcomes and the upper-tail ES the mean of the $\lceil (1 - \tau) n \rceil$
largest.

## References

- Acerbi, C., and Székely, B. (2014). Back-testing expected shortfall. *Risk*, 27, 76–81.
- Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of
  "backtesting" in conjunction with the internal models approach to market risk capital
  requirements*. Bank for International Settlements.
- Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market
  risk*. Bank for International Settlements.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic
  Review*, 39(4), 841–862. https://doi.org/10.2307/2527341
- Czado, C., Gneiting, T., and Held, L. (2009). Predictive model assessment for count data.
  *Biometrics*, 65(4), 1254–1261. https://doi.org/10.1111/j.1541-0420.2009.01191.x
- Diebold, F. X., and Li, C. (2006). Forecasting the term structure of government bond
  yields. *Journal of Econometrics*, 130(2), 337–364. https://doi.org/10.1016/j.jeconom.2005.03.005
- Diebold, F. X., and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of
  Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599
- Gneiting, T., Balabdaoui, F., and Raftery, A. E. (2007). Probabilistic forecasts,
  calibration and sharpness. *Journal of the Royal Statistical Society Series B*, 69(2),
  243–268. https://doi.org/10.1111/j.1467-9868.2007.00587.x
- Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
  https://doi.org/10.1198/016214506000001437
- Gneiting, T., and Ranjan, R. (2011). Comparing density forecasts using threshold- and
  quantile-weighted scoring rules. *Journal of Business & Economic Statistics*, 29(3),
  411–422. https://doi.org/10.1198/jbes.2010.08110
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business &
  Economic Statistics*, 23(4), 365–380. https://doi.org/10.1198/073500105000000063
- Harvey, D., Leybourne, S., and Newbold, P. (1997). Testing the equality of prediction mean
  squared errors. *International Journal of Forecasting*, 13(2), 281–291.
  https://doi.org/10.1016/S0169-2070(96)00719-4
- J.P. Morgan/Reuters (1996). *RiskMetrics—Technical Document* (4th ed.). Morgan Guaranty
  Trust Company, New York.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models.
  *The Journal of Derivatives*, 3(2), 73–84. https://doi.org/10.3905/jod.1995.407942
- Laio, F., and Tamea, S. (2007). Verification tools for probabilistic forecasts of
  continuous hydrological variables. *Hydrology and Earth System Sciences*, 11(4),
  1267–1277. https://doi.org/10.5194/hess-11-1267-2007
- Lu, J. (2025). Time-series foundation models in finance: Pretraining corpora,
  architectures, financial benchmarks, and risk-aware evaluation. In *Proceedings of the
  2025 2nd International Conference on Digital Economy and Computer Science (DECS 2025)*
  (pp. 144–151). ACM. https://doi.org/10.1145/3785706.3785728
- Nelson, C. R., and Siegel, A. F. (1987). Parsimonious modeling of yield curves. *The
  Journal of Business*, 60(4), 473–489. https://doi.org/10.1086/296409
- Newey, W. K., and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
  https://doi.org/10.2307/1913610
- Newey, W. K., and West, K. D. (1994). Automatic lag selection in covariance matrix
  estimation. *The Review of Economic Studies*, 61(4), 631–653. https://doi.org/10.2307/2297912
- Politis, D. N., and Romano, J. P. (1994). The stationary bootstrap. *Journal of the
  American Statistical Association*, 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
  https://doi.org/10.1111/1468-0262.00152
