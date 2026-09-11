# GARCH path experiment

Question: does allowing daily volatility to evolve within a 20-trading-day forecast
improve the existing -5 pp daily-close breach estimates?

Recorded before running this experiment. Earlier results and 2023+ history have been
examined; this is an exploratory comparison, not a new holdout. Do not modify the
submission notebooks, original forecasts, thresholds or selected model.

- Use the same public long/short net daily contributions and weekly cutoffs as the
  saved path evaluation. Verify outcomes against those saved rows.
- Primary model: zero-mean daily GARCH(1,1), standardised Student-t innovations,
  expanding annual maximum-likelihood fits. Fit through the first cutoff of each
  year; filter subsequent daily returns only when observed by a cutoff. Do not use
  future returns to filter forecasts or select specifications.
- Sensitivity: GJR-GARCH(1,1,1) with the same Student-t innovation assumption.
- Mechanism control: the primary model's same initial conditional variance and same
  simulated innovations, but constant variance throughout each simulated path.
  This separates within-path variance evolution from changing the shock distribution
  and starting scale relative to the existing historical simulation.
- Simulate 8,192 paths per forecast, in antithetic pairs, with fixed date-specific
  seeds. No drift. Score any daily-close cumulative loss <= -5 pp in exactly 20
  future trading days. Also record endpoint probability using the same paths.
- Compare all models on identical dates against saved FHS and volatility-logit
  probabilities. Report original through-2022 saved dates, 2023+ and individual years.
  Exclude extra late-2022 forecasts absent from the original saved evaluation.
- Report Brier/log loss and predicted/observed frequency; 13/26/52-week paired
  circular calendar-block intervals, 1,000 draws, fixed seed. Intervals are pointwise,
  exploratory and do not adjust for model search or capture Monte Carlo error.
- Verify library one-step variance, finite probabilities, endpoint<=path probability,
  exact return-label reproduction and convergence for every fit. Record warnings
  and persistence rather than discard inconvenient fits/results.
- Recompute recent-year primary and constant-variance simulations with 32,768 paths
  and a second fixed seed to assess Monte Carlo sensitivity, without choosing models
  based on that repeat. Record source and code hashes.
