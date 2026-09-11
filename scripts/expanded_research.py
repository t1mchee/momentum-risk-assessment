"""Sequential literature challengers. All outputs are exploratory research.

No historical holdout is claimed: it has already been consumed. Units are
percentage points internally. Seeds and model families are fixed here.
"""
from pathlib import Path
import hashlib, json, warnings, time
from zipfile import ZipFile
from io import StringIO
import re
import inspect
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize, special
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research"
OUT.mkdir(exist_ok=True)
THRESHOLD = -2.16
SEED = 20260911

def read_zip(name):
    with ZipFile(ROOT / "data/raw" / name) as z:
        lines = z.read(z.namelist()[0]).decode("utf-8-sig").splitlines()
    start = next(i for i, s in enumerate(lines) if re.match(r"^\s*\d{8},", s))
    end = start
    while end < len(lines) and re.match(r"^\s*\d{8},", lines[end]): end += 1
    f = pd.read_csv(StringIO("\n".join(lines[start-1:end])), index_col=0)
    f.columns = f.columns.str.strip()
    f.index = pd.to_datetime(f.index.astype(str), format="%Y%m%d")
    assert f.index.is_unique and f.index.is_monotonic_increasing
    f = f.replace([-99.99, -999], np.nan)
    assert np.isfinite(f).all().all()
    return f

def data():
    manifest = json.loads((ROOT / "data/raw/manifest.json").read_text())
    for name, record in manifest.items():
        assert hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest() == record["sha256"]
    daily = read_zip("momentum.zip").join(read_zip("factors.zip")[["Mkt-RF", "RF"]]).loc["1964":]
    daily["market"] = daily["Mkt-RF"]+daily.RF
    daily["rv126"] = daily.Mom.rolling(126).std()
    daily["ewma"] = np.sqrt(daily.Mom.pow(2).ewm(alpha=.06, adjust=False).mean())
    daily["market_vol"] = daily.market.rolling(126).std()
    daily["market_2y"] = np.expm1(np.log1p(daily.market/100).rolling(504).sum())
    g = daily.resample("W-FRI")
    w = g.Mom.sum(min_count=1).to_frame("y")
    w["market"] = g.market.apply(lambda x: ((1+x/100).prod()-1)*100)
    dates = pd.Series(daily.index, index=daily.index)
    w["start"] = dates.resample("W-FRI").min()
    w["end"] = dates.resample("W-FRI").max()
    w["sessions"] = g.size()
    w["daily_rv"] = g.Mom.apply(lambda x: np.mean(x**2))
    for col in ["rv126", "ewma", "market_vol", "market_2y"]:
        w[col] = daily[col].reindex(pd.DatetimeIndex(w.end)).to_numpy()
    w = w.iloc[1:-1].dropna()
    p = w[["y", "market", "start", "end", "sessions", "daily_rv"]].copy()
    p["cutoff"] = w.end.shift(1)
    for col in ["rv126", "ewma", "market_vol", "market_2y"]:
        p[col] = w[col].shift(1)
    p = p.dropna().loc[lambda x: x.start >= pd.Timestamp("1974-01-01")]
    p["bear"] = p.market_2y < 0
    p["event"] = p.y < THRESHOLD
    assert (p.cutoff < p.start).all()
    return daily, p

def logistic(train, test, columns):
    scaler = StandardScaler()
    a = scaler.fit_transform(train[columns])
    model = LogisticRegression(C=1., max_iter=1500, tol=1e-9)
    model.fit(a, train.event.astype(int))
    assert model.n_iter_[0] < 1500
    return model.predict_proba(scaler.transform(test[columns]))[:, 1]

def garch_forecasts(daily, panel):
    """Annual daily fits, observed-return filtering, fixed five-session simulation.

    Five sessions is a deliberately fixed approximation to a calendar trading
    week; no future holiday/closure count is used in forming the forecast.
    """
    result = panel.copy()
    audits = []
    for year in sorted(panel.start.dt.year.unique()):
        test = panel.loc[panel.start.dt.year == year]
        cutoff = test.cutoff.min()
        past = daily.loc[:cutoff, "Mom"]
        assert past.index.max() <= cutoff
        for name, o, dist in [("garch_normal", 0, "normal"), ("garch_t", 0, "t"), ("gjr_t", 1, "t")]:
            model = arch_model(past, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist, rescale=False)
            with warnings.catch_warnings(record=True) as caught:
                fitted = model.fit(disp="off", options={"maxiter": 1500})
            if fitted.convergence_flag != 0:
                raise RuntimeError(f"GARCH fit failed: {year}, {name}")
            params = fitted.params
            omega, alpha, beta = [float(params[k]) for k in ["omega", "alpha[1]", "beta[1]"]]
            gamma = float(params.get("gamma[1]", 0))
            nu = float(params.get("nu", np.inf))
            persistence = alpha+beta+gamma/2
            assert omega > 0 and alpha >= 0 and beta >= 0 and alpha+gamma >= -1e-8
            r_last = float(past.iloc[-1]); h_last = float(fitted.conditional_volatility.iloc[-1]**2)
            h_next = max(omega+(alpha+gamma*(r_last < 0))*r_last**2+beta*h_last, 1e-10)
            reference = fitted.forecast(horizon=5, reindex=False).variance.iloc[-1].to_numpy()
            expected = [h_next]
            for _ in range(4): expected.append(omega+persistence*expected[-1])
            np.testing.assert_allclose(expected, reference, rtol=1e-7, atol=1e-7)
            previous = cutoff
            for date, row in test.iterrows():
                observations = daily.loc[(daily.index > previous)&(daily.index <= row.cutoff), "Mom"]
                for r in observations:
                    h_next = max(omega+(alpha+gamma*(r < 0))*r*r+beta*h_next, 1e-10)
                previous = row.cutoff
                rng = np.random.default_rng(SEED+int(date.strftime("%Y%m%d")))
                h = np.full(6000, h_next); paths = np.zeros(6000)
                h_expected = h_next; var_sum = 0.
                for _ in range(5):
                    z = rng.normal(size=6000) if np.isinf(nu) else rng.standard_t(nu, size=6000)*np.sqrt((nu-2)/nu)
                    r = np.sqrt(h)*z
                    paths += r
                    h = np.maximum(omega+(alpha+gamma*(r < 0))*r*r+beta*h, 1e-10)
                    var_sum += h_expected
                    h_expected = omega+persistence*h_expected
                result.loc[date, "p_"+name] = ((paths < THRESHOLD).sum()+.5)/6001
                result.loc[date, "q_"+name] = np.quantile(paths, .05)
                result.loc[date, "q95_"+name] = np.quantile(paths, .95)
                result.loc[date, "sigma_"+name] = np.sqrt(var_sum/5)
            audits.append({"year": int(year), "model": name, "persistence": persistence, "nu": nu,
                           "converged": True, "warnings": len(caught), "fit_end": str(cutoff.date())})
        if year % 5 == 0: print("Volatility fits through", year, flush=True)
    return result, pd.DataFrame(audits)

def emission_logpdf(x, means, covs, dfs):
    return np.column_stack([stats.multivariate_normal.logpdf(x, mean=means[s], cov=covs[s])
                            if np.isinf(dfs[s]) else stats.multivariate_t.logpdf(x, loc=means[s], shape=covs[s], df=dfs[s])
                            for s in range(2)])

def forward_backward(log_emission, trans, start, backward=True):
    """Scaled two-state forward recursion. Filtered probabilities never smooth."""
    offsets = log_emission.max(axis=1)
    emission = np.maximum(np.exp(log_emission-offsets[:, None]), 1e-250)
    n = len(emission)
    filt = np.empty((n, 2)); scale = np.empty(n)
    prior = start
    for i in range(n):
        raw = prior*emission[i]; scale[i] = raw.sum(); filt[i] = raw/scale[i]
        prior = filt[i] @ trans
    ll = float(np.sum(np.log(scale)+offsets))
    if not backward: return filt, ll
    b = np.ones((n, 2))
    for i in range(n-2, -1, -1):
        b[i] = trans @ (emission[i+1]*b[i+1])/scale[i+1]
    gamma = filt*b; gamma /= gamma.sum(axis=1, keepdims=True)
    xi = filt[:-1, :, None]*trans[None, :, :]*(emission[1:]*b[1:])[:, None, :]
    xi /= xi.sum(axis=(1, 2), keepdims=True)
    return filt, ll, gamma, xi.sum(axis=0)

def fit_hmm(x, heavy=False):
    """Three starts, training likelihood only. Heavy model: Gaussian + t(df=5).

    This is a tractable analogue of the paper, NOT its full leverage model.
    """
    best = None
    for seed in [17, 41, 83]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, tol=.01,
                             min_covar=1e-4, random_state=seed, implementation="scaling").fit(x)
        order = np.argsort(gm.covars_[:, 1, 1])
        means = gm.means_[order].copy(); covs = gm.covars_[order].copy()+np.eye(2)*1e-5
        trans = gm.transmat_[order][:, order].copy(); start = gm.startprob_[order].copy()
        dfs = np.array([np.inf, 5. if heavy else np.inf])
        if heavy: covs[1] *= 3/5
        old = -np.inf; converged = not heavy; iterations = 0
        if heavy:
            for iterations in range(100):
                logs = emission_logpdf(x, means, covs, dfs)
                _, ll, responsibilities, counts = forward_backward(logs, trans, start)
                if np.isfinite(old) and abs(ll-old) < .01:
                    converged = True; break
                old = ll
                start = responsibilities[0]+1e-4; start /= start.sum()
                trans = counts+1e-3; trans /= trans.sum(axis=1, keepdims=True)
                for s in range(2):
                    delta = x-means[s]
                    mahal = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(covs[s]), delta)
                    latent = np.ones(len(x)) if np.isinf(dfs[s]) else (dfs[s]+2)/(dfs[s]+mahal)
                    weights = responsibilities[:, s]*latent
                    means[s] = np.sum(weights[:, None]*x, axis=0)/weights.sum()
                    centered = x-means[s]
                    covs[s] = (centered.T*weights) @ centered/responsibilities[:, s].sum()+np.eye(2)*1e-5
        filt, ll = forward_backward(emission_logpdf(x, means, covs, dfs), trans, start, False)
        history = list(gm.monitor_.history)
        gaussian_converged = len(history) >= 2 and abs(history[-1]-history[-2]) < .01
        record = {"means": means, "covs": covs, "trans": trans, "start": start, "dfs": dfs,
                  "filtered": filt, "ll": ll, "iterations": iterations+1,
                  "converged": converged if heavy else gaussian_converged, "seed": seed}
        if best is None or ll > best["ll"]: best = record
    return best

def hmm_forecasts(panel):
    out = panel.copy(); audits = []
    for year in range(1984, int(panel.start.dt.year.max())+1):
        test = panel.loc[panel.start.dt.year == year]
        train = panel.loc[panel.end <= test.cutoff.min()]
        center = train[["market", "y"]].mean().to_numpy()
        std = train[["market", "y"]].std().to_numpy()
        x = (train[["market", "y"]].to_numpy()-center)/std
        for name, heavy in [("hmm_normal", False), ("hmm_heavy", True)]:
            model = fit_hmm(x, heavy)
            mu = model["means"][:, 1]*std[1]+center[1]
            scales = np.sqrt(model["covs"][:, 1, 1])*std[1]
            dfs = model["dfs"]
            variances = scales**2*np.where(np.isinf(dfs), 1., np.where(np.isinf(dfs), 1., dfs)/(np.where(np.isinf(dfs), 3., dfs)-2))
            turbulent = int(np.argmax(variances))
            state = model["filtered"][-1]
            for date, row in test.iterrows():
                prior = state @ model["trans"]
                def cdf(value):
                    z = (value-mu)/scales
                    probs = np.array([stats.norm.cdf(z[s]) if np.isinf(dfs[s]) else stats.t.cdf(z[s], dfs[s]) for s in range(2)])
                    return float(prior @ probs)
                out.loc[date, "p_"+name] = cdf(THRESHOLD)
                low, high = min(mu-100*scales), max(mu+100*scales)
                out.loc[date, "q_"+name] = optimize.brentq(lambda q: cdf(q)-.05, low, high)
                out.loc[date, "q95_"+name] = optimize.brentq(lambda q: cdf(q)-.95, low, high)
                out.loc[date, "risk_state_"+name] = prior[turbulent]
                # Only AFTER predicting do we observe the outcome and update the filter.
                observed = (np.array([[row.market, row.y]])-center)/std
                likelihood = emission_logpdf(observed, model["means"], model["covs"], dfs)[0]
                posterior = prior*np.exp(likelihood-likelihood.max())
                state = posterior/posterior.sum()
            audits.append({"year": year, "model": name, "training_weeks": len(train), "loglik": model["ll"],
                           "converged": model["converged"], "iterations": model["iterations"],
                           "seed": model["seed"], "turbulent_state": turbulent,
                           "turbulent_duration_weeks": 1/(1-model["trans"][turbulent, turbulent])})
        if year % 5 == 0: print("Regime fits through", year, flush=True)
    return out, pd.DataFrame(audits)

def caviar_path(y, params, initial, asymmetric=False):
    lag = np.r_[0., y[:-1]]
    forcing = params[0]+params[2]*np.abs(lag)
    if asymmetric: forcing = params[0]+params[2]*np.maximum(lag, 0)+params[3]*np.maximum(-lag, 0)
    return signal.lfilter([1.], [1., -params[1]], forcing, zi=[params[1]*initial])[0]

def fit_caviar(y, asymmetric=False):
    initial = float(np.quantile(y, .05))
    starts = [.3, .7, .95]
    best = None
    for persistence in starts:
        x0 = [initial*(1-persistence)*.5, persistence, -.1]+([-.1] if asymmetric else [])
        def objective(p):
            q = caviar_path(y, p, initial, asymmetric)
            u = y-q
            return np.mean(np.maximum(.05*u, -.95*u))
        bounds = [(-10., 0.), (0., .995), (-5., 0.)]+([(-5., 0.)] if asymmetric else [])
        fitted = optimize.minimize(objective, x0, method="Powell", bounds=bounds,
                                   options={"maxiter": 1000, "xtol": 1e-6, "ftol": 1e-8})
        if fitted.success and (best is None or fitted.fun < best.fun): best = fitted
    if best is None: raise RuntimeError("CAViaR did not converge")
    return best.x, caviar_path(y, best.x, initial, asymmetric)[-1]

def tail_and_logit_forecasts(panel):
    out = panel.copy(); audits = []
    for scale in ["rv126", "ewma", "garch_t", "gjr_t"]:
        source = scale if scale in ["rv126", "ewma"] else "sigma_"+scale
        out["log_"+scale] = np.log(out[source])
    out["log_market"] = np.log(out.market_vol)
    out["interaction"] = out.bear.astype(int)*out.log_market
    for year in range(1984, int(out.start.dt.year.max())+1):
        test = out.loc[out.start.dt.year == year]
        train = out.loc[out.end <= test.cutoff.min()]
        assert train.end.max() <= test.cutoff.min() < test.start.min()
        out.loc[test.index, "p_frequency"] = train.event.mean()
        out.loc[test.index, "p_rolling_frequency"] = train.iloc[-260:].event.mean()
        out.loc[test.index, "q_historical"] = train.y.quantile(.05)
        for scale in ["rv126", "ewma", "garch_t", "gjr_t"]:
            name = "logit_"+scale
            out.loc[test.index, "p_"+name] = logistic(train, test, ["log_"+scale])
            source = scale if scale in ["rv126", "ewma"] else "sigma_"+scale
            residuals = (train.y/(np.sqrt(5)*train[source])).to_numpy()
            test_scale = np.sqrt(5)*test[source]
            out.loc[test.index, "p_fhs_"+scale] = [((residuals < THRESHOLD/s).sum()+.5)/(len(residuals)+1) for s in test_scale]
            out.loc[test.index, "q_fhs_"+scale] = test_scale*np.quantile(residuals, .05)
            out.loc[test.index, "q95_fhs_"+scale] = test_scale*np.quantile(residuals, .95)
            if scale in ["ewma", "garch_t"]:
                recent_residuals = residuals[-260:]
                out.loc[test.index, "p_fhs5y_"+scale] = [((recent_residuals < THRESHOLD/s).sum()+.5)/(len(recent_residuals)+1) for s in test_scale]
                out.loc[test.index, "q_fhs5y_"+scale] = test_scale*np.quantile(recent_residuals, .05)
                out.loc[test.index, "q95_fhs5y_"+scale] = test_scale*np.quantile(recent_residuals, .95)
        for name, columns in [("qr_vol", ["rv126"]), ("qr_state", ["rv126", "market_vol", "bear", "interaction"])]:
            scaler = StandardScaler()
            x = scaler.fit_transform(train[columns])
            qr = QuantileRegressor(quantile=.05, alpha=.01, solver="highs").fit(x, train.y)
            out.loc[test.index, "q_"+name] = qr.predict(scaler.transform(test[columns]))
        for name, asym in [("caviar_sav", False), ("caviar_as", True)]:
            params, last_q = fit_caviar(train.y.to_numpy(), asym)
            last_y = float(train.y.iloc[-1])
            for date, row in test.iterrows():
                q = params[0]+params[1]*last_q+params[2]*(max(last_y, 0) if asym else abs(last_y))
                if asym: q += params[3]*max(-last_y, 0)
                out.loc[date, "q_"+name] = q
                last_q, last_y = q, row.y
            audits.append({"year": year, "model": name, "persistence": params[1], "converged": True})
        if year % 5 == 0: print("Probability/tail fits through", year, flush=True)
    return out, pd.DataFrame(audits)

def scores(pred):
    probability, quantile = [], []
    y = pred.event.astype(int).to_numpy()
    for c in [c for c in pred if c.startswith("p_")]:
        p = np.clip(pred[c].to_numpy(), 1e-8, 1-1e-8)
        probability.append({"model": c[2:], "weeks": len(pred), "events": y.sum(), "brier": np.mean((p-y)**2),
                            "logloss": np.mean(-(y*np.log(p)+(1-y)*np.log1p(-p))),
                            "predicted_rate": p.mean(), "observed_rate": y.mean()})
    for c in [c for c in pred if c.startswith("q_")]:
        q = pred[c].to_numpy(); u = pred.y.to_numpy()-q
        breach = u < 0
        runs = int(np.sum(breach & ~np.r_[False, breach[:-1]]))
        quantile.append({"model": c[2:], "pinball": np.mean(np.maximum(.05*u, -.95*u)),
                         "breach_rate": breach.mean(), "breach_weeks": int(breach.sum()), "breach_runs": runs,
                         "mean_q05_pp": q.mean()})
    probability_table = pd.DataFrame(probability).set_index("model") if probability else pd.DataFrame(index=pd.Index([], name="model"))
    quantile_table = pd.DataFrame(quantile).set_index("model") if quantile else pd.DataFrame(index=pd.Index([], name="model"))
    return probability_table, quantile_table

def paired(pred, candidate, base, kind, block=13):
    if kind in ["brier", "logloss"]:
        y = pred.event.astype(int).to_numpy()
        def loss(name):
            p = np.clip(pred["p_"+name].to_numpy(), 1e-8, 1-1e-8)
            return (p-y)**2 if kind == "brier" else -(y*np.log(p)+(1-y)*np.log1p(-p))
    else:
        def loss(name):
            u = pred.y.to_numpy()-pred["q_"+name].to_numpy()
            return np.maximum(.05*u, -.95*u)
    diff = loss(candidate)-loss(base); n = len(diff)
    rng = np.random.default_rng(SEED); boot = []
    for _ in range(2000):
        starts = rng.integers(0, n-block+1, int(np.ceil(n/block)))
        idx = (starts[:, None]+np.arange(block)).ravel()[:n]
        boot.append(diff[idx].mean())
    lo, hi = np.quantile(boot, [.025, .975])
    return {"candidate": candidate, "base": base, "score": kind, "block": block,
            "difference": diff.mean(), "lower95": lo, "upper95": hi}

def scaling_diagnostic(pred):
    """Diagnostic only: fixed-notional P&L and de-risking, NOT compounding a funded portfolio."""
    weekly_target = .12/np.sqrt(52)*100
    rows = []
    weights = {"unscaled": np.ones(len(pred)), "constant_50pct": np.full(len(pred), .5)}
    for scale in ["rv126", "ewma", "garch_t", "gjr_t"]:
        source = scale if scale in ["rv126", "ewma"] else "sigma_"+scale
        weights["vol_target_"+scale] = np.minimum(1., weekly_target/(np.sqrt(5)*pred[source].to_numpy()))
    weights["hmm_reduce"] = 1-pred.risk_state_hmm_heavy.to_numpy()
    # Mean/variance dynamic scaling uses a small predictive mean regression.
    dynamic = np.full(len(pred), np.nan)
    for year in sorted(pred.start.dt.year.unique()):
        mask = pred.start.dt.year == year
        # Pred carries only 1984+ observations: require five past years before diagnostic begins.
        train = pred.loc[pred.end <= pred.loc[mask, "cutoff"].min()]
        if len(train) < 260: continue
        def design(f): return np.column_stack([np.ones(len(f)), f.bear.astype(int), f.market_vol**2, f.bear.astype(int)*f.market_vol**2])
        coef = np.linalg.lstsq(design(train), train.y, rcond=None)[0]
        mean = design(pred.loc[mask]) @ coef
        variance = 5*pred.loc[mask, "sigma_gjr_t"].to_numpy()**2
        # Normalize relative to historical mean / variance; cap at [0,1], no short reversal or leverage.
        reference = max(train.y.mean(), .01)/train.y.var()
        dynamic[mask.to_numpy()] = np.clip((mean/variance)/reference, 0, 1)
    weights["mean_variance_reduce"] = dynamic
    common = np.isfinite(dynamic)
    for name, weight in weights.items():
        w = np.asarray(weight)[common]; y = pred.y.to_numpy()[common]*w
        cumulative = np.r_[0., np.cumsum(y)]
        # Overlay-change turnover only, excludes underlying momentum portfolio turnover.
        turns = np.r_[0., abs(np.diff(w))]
        net = y-.1*turns  # 10 bp per unit change in factor weight, illustrative.
        rows.append({"rule": name, "weeks": len(y), "mean_weight": w.mean(), "weekly_mean_pp": y.mean(),
                     "annualized_mean_sd": np.sqrt(52)*y.mean()/y.std(ddof=1), "worst_week_pp": y.min(),
                     "q05_pp": np.quantile(y, .05), "max_fixed_notional_drawdown_pp": (cumulative-np.maximum.accumulate(cumulative)).min(),
                     "illustrative_cost_mean_pp": net.mean(), "average_overlay_change": turns.mean()})
    return pd.DataFrame(rows).set_index("rule")

def self_tests():
    # Verify likelihood against brute-force enumeration, and filter prefix invariance.
    import itertools
    e = np.array([[.3, .6], [.7, .2], [.4, .8]])
    a = np.array([[.9, .1], [.2, .8]]); start = np.array([.6, .4])
    total = 0.
    for states in itertools.product(range(2), repeat=3):
        p = start[states[0]]*e[0, states[0]]
        for i in range(1, 3): p *= a[states[i-1], states[i]]*e[i, states[i]]
        total += p
    f, ll = forward_backward(np.log(e), a, start, False)
    np.testing.assert_allclose(np.exp(ll), total)
    short, _ = forward_backward(np.log(e[:2]), a, start, False)
    np.testing.assert_allclose(f[:2], short)
    p = np.array([-.1, .8, -.2]); y = np.array([1., -2., .5]); q = caviar_path(y, p, -2.)
    manual = []; last = -2.
    for v in [0., 1., -2.]:
        last = p[0]+p[1]*last+p[2]*abs(v); manual.append(last)
    np.testing.assert_allclose(q, manual)
    return "PASS: HMM likelihood enumeration, filtered-prefix invariance, CAViaR recursion"

def run(force=False):
    path = OUT / "expanded_models.pkl"
    # Validate source snapshots even when returning cached model results.
    daily, panel = data()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()+(ROOT / "data/raw/manifest.json").read_bytes()).hexdigest()
    if path.exists() and not force:
        existing = pd.read_pickle(path)
        if existing["fingerprint"] == fingerprint: return existing
    print(self_tests(), flush=True)
    stage = OUT / "volatility_cache.pkl"
    raw_key = (ROOT / "data/raw/manifest.json").read_text()
    vol_key = hashlib.sha256((inspect.getsource(garch_forecasts)+raw_key).encode()).hexdigest()
    cached = pd.read_pickle(stage) if stage.exists() else None
    if isinstance(cached, dict) and cached.get("key") == vol_key:
        vol, vol_audit = cached["value"]
        assert vol.index.equals(panel.index)
    else:
        vol, vol_audit = garch_forecasts(daily, panel)
        pd.to_pickle({"key": vol_key, "value": (vol, vol_audit)}, stage)
    stage_hmm = OUT / "hmm_cache.pkl"
    hmm_key = hashlib.sha256((inspect.getsource(hmm_forecasts)+inspect.getsource(fit_hmm)+inspect.getsource(forward_backward)+raw_key).encode()).hexdigest()
    cached = pd.read_pickle(stage_hmm) if stage_hmm.exists() else None
    if isinstance(cached, dict) and cached.get("key") == hmm_key: hmm, hmm_audit = cached["value"]
    else:
        hmm, hmm_audit = hmm_forecasts(panel)
        pd.to_pickle({"key": hmm_key, "value": (hmm, hmm_audit)}, stage_hmm)
    final, tail_audit = tail_and_logit_forecasts(vol)
    for col in hmm:
        if col.startswith(("p_", "q_", "q95_", "risk_state_")): final[col] = hmm[col]
    final = final.loc[final.start.dt.year >= 1984].copy()
    assert final.filter(regex=r"^(p_|q_)").notna().all().all()
    results = {"fingerprint": fingerprint, "predictions": final, "volatility_audit": vol_audit,
               "hmm_audit": hmm_audit, "tail_audit": tail_audit, "self_tests": self_tests(),
               "scaling": scaling_diagnostic(final)}
    pd.to_pickle(results, path)
    p, q = scores(final)
    print("PROBABILITY\n", p.sort_values("logloss").to_string(), flush=True)
    print("TAIL\n", q.sort_values("pinball").to_string(), flush=True)
    return results

if __name__ == "__main__": run()
