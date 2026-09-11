# Book-level explanations inside the existing PM system

Scope: descriptive diagnostics on the existing covered IWV proxy, not new predictors
for the French-factor forecast. Keep the public factor and reconstructed book separate.
Use only 2013–2022 input returns and existing pre-2023 monthly memberships. Preserve
upstream data, prior PCA outputs and all frozen forecast/scenario predictions.

1. For the accepted 252-session PCA snapshots, compute constituent/sector conditional
   mean contributions on the same net lower-15% windows at 1/5/10/20 sessions.
   Concentration denominator: sum of negative NAME conditional-average contributions,
   not net loss or sum of sector net contributions. Report top-five names, simultaneous
   long/short losses, connected episode concentration and fixed-tail largest-episode
   omission. Omit the largest adverse name as a sensitivity with no renormalisation;
   show both original-tail accounting and reselected-tail mean. Neither is a trade.
2. Summarise first-three-PC risk and long/short covariance within that subspace,
   alongside PC1. Offset = 1 - Var(L+S)/(Var(L)+Var(S)) within the selected subspace.
   Negative means reinforcement; always show absolute net component SD too. Repeat
   after market removal. Invariant to sign flips and rotations inside the fixed
   selected subspace; track boundary eigenvalue gap because the subspace can change.
3. Consecutive calendar month-end books only: form a common UNION of names with
   adequate return history at both endpoints, including entrants/exits when possible.
   Require >=95% of EACH previously accepted leg's weight at BOTH endpoints and
   >=90% calendar coverage in each window. No extrapolation across failed snapshots.
   Common-support selection is made at the later date: this is a retrospective bridge,
   NOT a revision to what was known/displayed at the earlier date.
4. Evaluate old/new signed weights against old/new covariance matrices. Symmetric
   attribution in variance (pp²): weight effect .5[(B-A)+(D-C)], covariance effect
   .5[(C-A)+(D-B)], where A=old/old, B=new/old, C=old/new, D=new/new. Record exact
   reconciliation and name-level weight effects. This describes membership/weight
   updates and changed co-movement, not PM trade intent or a causal intervention.

Integration: one dated constituent-investigation slot in Notebook 09, with explicit
unavailable states and coverage, plus short method recipes. No extra dashboard or
composite signal. Raw PC1 concentration is not promoted; forecast probabilities and
scenarios remain unchanged. Descriptive outputs remain in-fit and selected-subset
statistics with unresolved source corporate-action/dividend/delisting issues.
