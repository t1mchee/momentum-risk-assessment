# Economic extension — fixed before new results

Objective: interpretable economic vulnerability beyond volatility, not trading P&L.
Existing notebooks and frozen predictions are preserved. All history through 2022
has already been examined: these are exploratory chronological tests, NOT a fresh
holdout. Do not access the other project's sealed post-2022 company observations.

## Audit and hypotheses

Earlier work already tested market volatility/bear interactions, 252-day net beta,
run-up/drawdown, scale-vs-shape, a public CFTC proxy, and trailing leg attribution.
Unresolved questions here are (1) changing directional leg exposure in conditional
responses, (2) multi-resolution trailing shape as a forward predictor, and (3) a
small market-priced macro block. No company-price pipeline rebuild is justified
given the existing corporate-action quality gate.

1. Exposure: lagged rolling linear and positive-market betas of the net factor,
   plus bear × positive-market beta. Estimate legs jointly on identical daily
   designs so addition holds. Primary lookback 252 sessions, 126/504 sensitivity.
2. Distribution: trailing 252 ending windows of 1/5/10/20-session summed P&L.
   Standardize within each distribution to remove location/scale. Use average
   standardized q15, average central-tail asymmetry, and 20-vs-1-session contrasts.
   These are trailing inputs; subsequent weekly outcomes are evaluation targets.
3. Macro: VIX relative to realised market volatility, 21-session change in 10-year
   Treasury yield, 10y–2y curve, BAA-minus-Treasury spread and its 21-session change.
   Download official FRED data; record hashes. Use a conservative seven-calendar-day
   availability lag and repeat at 14 days. These are current-vintage histories,
   not release-vintage proof. Missing histories are never backfilled.

## Tests

Annual expanding fits, completed outcomes only, primary 1984–2022 exposure/shape
sample; macro evaluations start 2000 with at least 260 training weeks on common
dates. Fixed regularisation: logistic C=0.1; normalized-outcome quantile regression
alpha=0.01 at q05. All new block models compare to the same fitted volatility-only
model and to frozen empirical forecasts on exactly the same weeks. No penalty grid.
Report all individual blocks and exposure+distribution (plus all on macro sample).

Conditional response is a distinct task, not an unconditional forecast: feed actual
weekly market move ONLY when evaluating a response conditional on that move. Compare
existing six-term state/asymmetric OLS with an augmented design including lagged
net linear beta × market and lagged directional-beta contrast × positive market.
Also compare to a simple lagged rolling-beta response with training mean intercept.
Fit identical design to long/short, then sum; this is not extra predictive information
versus fitting net directly. Primary 252-day exposures; 126/504 sensitivity.

Evaluate Brier/log loss and q05 pinball/coverage; conditional MAE/RMSE and directional
response. Chronological block resampling 13/26/52 weeks, 1000 draws, fixed seed
20260914; paired differences, not independent forecast samples. Report 1984–1999,
2000–2016, 2017–2022; nonbear/bear and bear rebound >2%; excluding 2008–09 and 2020;
year win fractions. No single crisis/date pair establishes incremental usefulness.
Bootstrap intervals are pointwise, conditional on fitted forecasts, not search-adjusted.

Retain a predictive extension only if average loss improves, 26/52-block intervals
support improvement, recent performance and non-crisis evidence do not contradict it.
Descriptive exposure can remain separately labelled if forecasting gates fail, but
must not be relabelled a discovery. Do not tune until a desired result appears.

## Benchmark clarification after initial conditional results

The dynamic conditional model improves average loss against the old static model,
but that alone does not isolate information beyond volatility. Add the matched-size
comparator static design + log(scale) × market + log(scale) × positive market;
also add the exposure terms to that comparator. This tests the actual claim rather
than promoting a favourable comparison against an insufficient benchmark. This
clarification is recorded after seeing the first static/dynamic scores; it is not
represented as an original preregistration. No change to primary exposure window.

## Follow-on economic question (after the forward macro failure)

The macro block has not improved pooled unconditional downside forecasts. Test its
different, PM-relevant use: conditional response to supplied market/10y-yield/BAA-spread
changes. Outcomes and supplied driver moves are contemporaneous; this is explicitly
NOT forecasting the moves, and coefficients are associations, not structural shocks.
Primary linear regression uses last 260 completed weeks; 156/520-week sensitivities.
Compare market-only, market+rate, market+credit, and market+both on identical dates,
with identical training windows, to identify which information matters. Use seven-day
lag before each annual fit for macro data availability; repeat fourteen days. Frozen
annual coefficients. Score net and both legs; same designs ensure addition. Compare
to existing static and exposure conditional responses on common weeks as well.
No new model/penalty search; do not select lookback by best historical score.

Validation: prefix reconstruction, future-outcome perturbation, common-date alignment,
bounded probabilities, quantile score calculation, additive legs, source hashes.
Deliver a reproducible research module and readable results; integrate only supported
additions, with uncertainty/support and explicit scope boundaries.
