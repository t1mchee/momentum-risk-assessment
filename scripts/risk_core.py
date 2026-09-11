"""Proposed interpretable risk core, selected during the expanded exploratory search.

Percent-point fixed-notional P&L throughout. This is a research model, not a
production claim. Scenario decomposition remains a separate conditional model.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class MomentumRiskCore:
    residual_weeks: int = 260
    horizon_sessions: int = 5
    variance_decay: float = .94
    residuals: np.ndarray = field(default=None, repr=False)
    fitted_asof: pd.Timestamp = None
    training_end: pd.Timestamp = None

    def __post_init__(self):
        if not 0 < self.variance_decay < 1:
            raise ValueError("variance_decay must be strictly between zero and one")
        if self.residual_weeks < 26 or self.horizon_sessions < 1:
            raise ValueError("Require at least 26 residual weeks and a positive horizon")

    def volatility_series(self, daily_returns, asof):
        """Actual configurable EWMA, in daily percentage-point units, through asof."""
        history = daily_returns.loc[:pd.Timestamp(asof)]
        if not history.index.is_monotonic_increasing or not history.index.is_unique:
            raise ValueError("Daily return dates must be ordered and unique")
        if len(history) < 126 or not np.isfinite(history).all():
            raise ValueError("At least 126 finite daily returns required")
        return np.sqrt(history.pow(2).ewm(alpha=1-self.variance_decay, adjust=False).mean())

    def fit(self, weekly_history, asof, daily_returns=None):
        """Completed outcomes only. Supply daily returns when changing the decay.

        Legacy input `ewma` is explicitly the upstream 0.94 series. Nondefault
        decay cannot silently reuse it; daily returns recompute historical scales.
        """
        asof = pd.Timestamp(asof)
        train = weekly_history.loc[weekly_history.end <= asof].sort_values('end')
        if len(train) < self.residual_weeks:
            raise ValueError(f"At least {self.residual_weeks} completed weeks required")
        assert (train.cutoff < train.start).all()
        recent = train.iloc[-self.residual_weeks:].copy()
        if daily_returns is not None:
            scale = self.volatility_series(daily_returns, asof)
            recent['ewma'] = scale.reindex(pd.DatetimeIndex(recent.cutoff)).to_numpy()
        elif self.variance_decay != .94:
            raise ValueError("Supply daily_returns to fit when changing variance_decay")
        if not np.isfinite(recent[["y", "ewma"]]).all().all() or (recent.ewma <= 0).any():
            raise ValueError("Finite returns and positive preceding EWMA scales required")
        self.residuals = (recent.y/(np.sqrt(self.horizon_sessions)*recent.ewma)).to_numpy()
        self.fitted_asof, self.training_end = asof, train.end.max()
        return self

    def forecast_from_returns(self, daily_returns, asof, loss_threshold_pp=-2.16, uncertainty=False):
        """Use the configured decay for the current scale as well as historical scales."""
        if self.fitted_asof is None or pd.Timestamp(asof) < self.fitted_asof:
            raise ValueError("Forecast cutoff must be on or after the fit cutoff")
        scale = self.volatility_series(daily_returns, asof)
        return self.forecast(float(scale.iloc[-1]), loss_threshold_pp, uncertainty)

    def forecast(self, sigma_daily_pp, loss_threshold_pp=-2.16, uncertainty=False):
        """Scale a frozen empirical distribution by the risk known at a cutoff.

        Optional intervals cover empirical-distribution sampling uncertainty only,
        not future-outcome uncertainty, volatility-model error or revisions.
        """
        if self.residuals is None: raise ValueError("Fit the model first")
        sigma = float(sigma_daily_pp)
        if not np.isfinite(sigma) or sigma <= 0: raise ValueError("Risk scale must be positive")
        if not np.isfinite(loss_threshold_pp): raise ValueError("Loss threshold must be finite")
        scale = np.sqrt(self.horizon_sessions)*sigma
        outcomes = self.residuals*scale
        result = {"probability": float(((outcomes < loss_threshold_pp).sum()+.5)/(len(outcomes)+1)),
                  "q05_pp": float(np.quantile(outcomes, .05)), "q95_pp": float(np.quantile(outcomes, .95)),
                  "loss_threshold_pp": float(loss_threshold_pp), "sigma_daily_pp": sigma,
                  "residual_weeks": len(outcomes), "fit_asof": str(self.fitted_asof.date()),
                  "status": "exploratory research; calendar-week horizon approximated by five sessions"}
        if uncertainty:
            rng = np.random.default_rng(20260911); n = len(outcomes); block = 13
            starts = rng.integers(0, n-block+1, (1000, int(np.ceil(n/block))))
            indices = (starts[..., None]+np.arange(block)).reshape(1000, -1)[:, :n]
            boot = outcomes[indices]
            probabilities = ((boot < loss_threshold_pp).sum(axis=1)+.5)/(n+1)
            quantiles = np.quantile(boot, .05, axis=1)
            result["probability_sampling_interval95"] = np.quantile(probabilities, [.025, .975]).tolist()
            result["q05_sampling_interval95_pp"] = np.quantile(quantiles, [.025, .975]).tolist()
        return result

def verify_replay(panel, predictions):
    """Replay every annual fit, plus future-outcome and mathematical invariance tests."""
    rows = []
    for year in sorted(predictions.start.dt.year.unique()):
        test = predictions.loc[predictions.start.dt.year == year]
        model = MomentumRiskCore().fit(panel, test.cutoff.min())
        for _, row in test.iterrows(): rows.append(model.forecast(row.ewma))
    replay = pd.DataFrame(rows, index=predictions.index)
    np.testing.assert_allclose(replay.probability, predictions.p_fhs5y_ewma, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay.q05_pp, predictions.q_fhs5y_ewma, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay.q95_pp, predictions.q95_fhs5y_ewma, rtol=1e-12, atol=1e-12)
    cutoff = predictions.cutoff.iloc[0]
    original = MomentumRiskCore().fit(panel, cutoff)
    altered = panel.copy()
    altered.loc[altered.end > cutoff, ["y", "ewma"]] = 999.
    perturbed = MomentumRiskCore().fit(altered, cutoff)
    np.testing.assert_array_equal(original.residuals, perturbed.residuals)
    assert original.forecast(.5)["probability"] <= original.forecast(1.)["probability"]
    assert original.forecast(1.)["q05_pp"] <= original.forecast(1.)["q95_pp"]
    for bad in [0., -1., np.nan]:
        try: original.forecast(bad)
        except ValueError: pass
        else: raise AssertionError("Invalid risk scale accepted")
    return f"PASS: {len(replay)} forecast replay, future-outcome invariance, monotone negative-threshold risk, quantile order, invalid-input rejection"
