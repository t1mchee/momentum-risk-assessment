## Recent-period validation: 2023–July 2026

**Bottom line: keep the simple reference models, but downgrade confidence in path-probability calibration.** This extension does not establish a new predictive signal. The earlier through-2022 worked assessment remains historical; it is not the latest forecast.

1. **What was frozen:** the −2.16 pp weekly loss threshold, EWMA .94, 260-shock history, 252-session exposure lookback, 20-session/−5 pp path target, 1,260-path history and existing logistic penalties. Annual chronological refits and trailing inputs update as previously specified. No specification was retuned after these results.
2. **What was extended:** cached public factor and academic leg data through 31 July 2026. Weekly risk evaluation ends 24 July (the existing procedure conservatively drops the last weekly bin); conditional response evaluation ends 31 July; path cutoffs end 3 July so all 20 future sessions are observed. Company-level and options coverage is not extended.
3. **What was checked:** all 2,035 historical weekly forecasts, 2,035 conditional forecasts and 2,030 path forecasts reproduce the saved pre-2023 results within numerical tolerance. Unfinished outcomes cannot enter training. Conditional tests supply the realised market move to evaluate a scenario relationship, not an unconditional forecast.
4. **What this is not:** an untouched holdout. Earlier broader research already examined this history. This pass records a frozen protocol before its new calculation, not a fictional preregistration of the whole project. Current-vintage factor data are not a point-in-time publication archive.

### Weekly downside: reasonable pooled frequency, uneven through time

| Model | Weeks / losses | Mean probability | Observed frequency | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained empirical/EWMA | 186 / 23 | 13.17% | 12.37% | .104425 | .356248 |
| Same-EWMA logistic | 186 / 23 | 14.15% | 12.37% | .103648 | .353955 |
| Expanding historical frequency | 186 / 23 | 8.45% | 12.37% | .110003 | .383599 |

The logistic comparator is slightly better on both scores. The retained model's log-loss difference is +.00229; its 26-week block interval [−.00433, +.01124] spans zero. There is no demonstrated superiority over that comparator. The q05 forecast is breached in 4/186 weeks (2.15%, versus nominal 5%); pinball loss is .19767 pp. This suggests conservative recent lower-tail estimates, not proven calibration.

| Year | Weekly losses / weeks | Weekly estimated / observed | Path breaches / cutoffs | Path estimated / observed |
| --- | ---: | ---: | ---: | ---: |
| 2023 | 10 / 52 | 12.55% / 19.23% | 9 / 52 | 20.23% / 17.31% |
| 2024 | 1 / 52 | 10.54% / 1.92% | 0 / 52 | 14.22% / 0.00% |
| 2025 | 8 / 52 | 13.26% / 15.38% | 5 / 52 | 20.29% / 9.62% |
| 2026, partial | 4 / 30 | 18.68% / 13.33% | 3 / 27 | 27.33% / 11.11% |

Pooled agreement conceals underprediction in 2023 and overprediction in quiet 2024. Only 33 weekly bear-state observations, containing six losses, are available; this is not a strong new bear-regime validation. Zero 2024 path events does not prove the probabilities should have been zero.

### Conditional response: simplicity survives

| Conditional model | Net MAE, pp | Net MSE, pp² |
| --- | ---: | ---: |
| Retained rolling beta | 1.5964 | 4.0649 |
| Nonlinear exposure challenger | 1.6258 | 4.1358 |
| Matched-size volatility response | 1.6381 | 4.1316 |

Across 187 weeks the rolling reference has .0294 pp lower MAE than the nonlinear challenger; the 26-week interval is [−.0606, −.0010]. The MSE interval spans zero. The challenger's small MAE advantage over the volatility-response comparator also has an interval spanning zero. Keep the simple reference; do not promote a richer model or claim precise crash scenario magnitudes.

### Path risk: useful distinction, weak recent probability calibration

Across 183 cutoffs, the empirical path model estimates 19.59% breach probability against 17/183 observed breaches (9.29%). The volatility logistic also overpredicts (19.48%). Their log losses are .32231 and .32039, respectively; the paired 26-week interval for the difference is [−.00218, +.00604]. Switching between them does not solve the calibration weakness.

Only eight paths finish beyond −5 pp: nine of the 17 breached paths recover above the barrier before the endpoint. This supports showing interim losses separately from endpoint losses, **not** treating the displayed path probabilities as calibrated limits or dependable alerts. These are overlapping windows, not 17 independent crashes.

Intervals use 1,000 fixed-seed moving-calendar-block draws at both 13 and 26 weeks. They are pointwise, short-sample and unadjusted for the wider research search. A few recent years cannot establish stability or a structural post-2010 breakdown. No model is promoted from this evaluation. Prediction rows, annual scores, paired intervals, source hashes and replay checks accompany the reproduction package.
