# Frozen recent-period validation — no retuning

Registered before calculating recent scores in this pass. This is evaluation on
previously examined history, not an untouched holdout. Use the existing versioned
public-data ZIP snapshots, report their actual last dates, and acquire no replacement
vintage silently. Do not extend constituent or options claims by extending the factor.

Evaluate 2023 onward separately. No changes to features, thresholds, lookbacks,
penalties, portfolio definitions or model choice. Allowed updates are the existing
annual expanding chronological refits and daily trailing inputs. Do not change the
main 2022 worked example or its frozen evaluation files; append a separately labelled
recent-evidence section irrespective of outcome.

1. Baseline: empirical last 260 completed weekly shocks, EWMA .94, five-session
   rescaling, -2.16 pp event and q05. Same-EWMA logit C=1 and historical frequency
   comparators. Use the existing expanded weekly-calendar construction, including
   its conservative final-week exclusion. Replay pre-2023 predictions against saved
   inputs before trusting the extension. Preserve the known holiday-week mismatch.
2. Conditional: annual rolling-beta reference with training residual intercept;
   nonlinear exposure challenger and matched-size volatility-aware comparator.
   252-session exposure inputs. Realised market move is supplied only for conditional
   evaluation, never to generate an unconditional probability. Replay old predictions.
3. Path: next 20 daily-close sessions, fixed -5 pp barrier and endpoint; FHS last
   1,260 completed weekly-start paths scaled by starting daily sigma. Comparator:
   annual volatility logit C=.1. Exclude unfinished horizons from every training set.
   Replay pre-2023 forecasts against saved path test before accepting the extension.

Report annual and pooled recent Brier/log loss/calibration, q05 breach and pinball,
conditional MAE/MSE, path breach versus endpoint counts; event support alongside
scores. Paired 13/26-week moving-calendar-block intervals, 1,000 draws, fixed seed:
short recent sample, pointwise, overlapping horizons and no search adjustment.
Report counts when a subgroup is too small; do not interpret no events as proof
of calibration. Dates/units/clocks stay explicit. Save prediction rows, provenance,
replay checks and results; no model promotion based on this pass.
