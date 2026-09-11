"""Beyond-volatility tests. See research/ECONOMIC_EXTENSION_PROTOCOL.md.

Read frozen public-factor data only through 2022; never modify Notebook 08 inputs.
Run: python scripts/economic_extension.py
"""
from pathlib import Path
from io import StringIO
from datetime import datetime, timezone
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/economic_extension'
BASE = ROOT / 'research/notebook_v8'
SEED = 20260914
EXPOSURE = ['beta_net', 'up_beta_net', 'bear_up_beta']
SHAPE = ['shape_left', 'shape_asym', 'shape_left_term', 'shape_asym_term']
MACRO = ['implied_relative', 'rate_change', 'curve', 'credit', 'credit_change']


def get_macro():
    folder = OUT / 'raw'
    folder.mkdir(parents=True, exist_ok=True)
    def fetch(series):
        path = folder / (series + '.csv')
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1964-01-01&coed=2022-12-31'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=35) as response:
                raw = response.read()
            f = pd.read_csv(StringIO(raw.decode()))
            assert series in f and len(f) > 100
            path.write_bytes(raw)
        f = pd.read_csv(path, index_col=0, parse_dates=True)
        f[series] = pd.to_numeric(f[series], errors='coerce')
        f = f.loc[:'2022'].dropna()
        assert f.index.is_unique
        return f, dict(series=series, url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       retrieved_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                       first=str(f.index.min().date()), last=str(f.index.max().date()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, ['VIXCLS', 'DGS10', 'DGS2', 'BAA10Y']))
    (folder / 'manifest.json').write_text(json.dumps([r[1] for r in results], indent=2))
    return pd.concat([r[0] for r in results], axis=1).sort_index()


def daily_features(d, window=252):
    """All regression inputs end at the current daily observation, never beyond."""
    f = pd.DataFrame(index=d.index)
    m = d.market
    up = m.clip(lower=0)
    # Rolling covariance solution for y = a + b*m + c*max(m,0).
    vm = m.rolling(window).var(); vu = up.rolling(window).var()
    cross = m.rolling(window).cov(up)
    determinant = vm * vu - cross**2
    assert (determinant.dropna() > 0).all()
    for leg in ['long', 'short']:
        cm = d[leg].rolling(window).cov(m)
        cu = d[leg].rolling(window).cov(up)
        f['beta_' + leg] = cm / vm
        b = (cm * vu - cu * cross) / determinant
        c = (cu * vm - cm * cross) / determinant
        f['up_beta_' + leg] = b + c
        f['down_beta_' + leg] = b
    for name in ['beta', 'up_beta', 'down_beta']:
        f[name + '_net'] = f[name + '_long'] + f[name + '_short']
    f['directional_gap'] = f.up_beta_net - f.down_beta_net
    left, asym = [], []
    for h in [1, 5, 10, 20]:
        x = d.net.rolling(h).sum(); rolling = x.rolling(252)
        a, b, c = [rolling.quantile(q) for q in [.15, .5, .85]]
        f[f'left_{h}'] = (a - rolling.mean()) / rolling.std()
        f[f'asym_{h}'] = (b - a) / (c - a)
        left.append(f[f'left_{h}']); asym.append(f[f'asym_{h}'])
    f['shape_left'] = pd.concat(left, axis=1).mean(axis=1)
    f['shape_asym'] = pd.concat(asym, axis=1).mean(axis=1)
    f['shape_left_term'] = f.left_20 - f.left_1
    f['shape_asym_term'] = f.asym_20 - f.asym_1
    f['market_vol'] = m.rolling(126).std()
    return f


def panel(d, window=252, macro=None, lag=7):
    dates = pd.Series(d.index, index=d.index)
    p = d[['long', 'short', 'net']].resample('W-FRI').sum(min_count=1)
    p['start'] = dates.resample('W-FRI').min()
    p['end'] = dates.resample('W-FRI').max()
    p['cutoff'] = p.end.shift()
    p['market'] = d.market.resample('W-FRI').apply(lambda x: np.expm1(np.log1p(x/100).sum())*100)
    p['bear'] = d.market_2y.reindex(pd.DatetimeIndex(p.cutoff)).to_numpy() < 0
    p['scale'] = np.sqrt(5) * d.sigma.reindex(pd.DatetimeIndex(p.cutoff)).to_numpy()
    p['log_scale'] = np.log(p.scale)
    # Preserve original rounded Mom outcome and event for fair probability comparison.
    p['y'] = d.Mom.resample('W-FRI').sum(min_count=1)
    p['event'] = p.y < -2.16
    f = daily_features(d, window)
    for col in f:
        p[col] = f[col].reindex(pd.DatetimeIndex(p.cutoff)).to_numpy()
    p['bear_up_beta'] = p.bear.astype(float) * p.up_beta_net
    if macro is not None:
        # Match daily macro levels with <=7 calendar-day stale observation; no backfill.
        calendar = d.index
        joined = pd.DataFrame(index=calendar)
        for col in macro:
            s = macro[col].dropna()
            joined[col] = s.reindex(calendar, method='ffill', tolerance=pd.Timedelta(days=7))
        features = pd.DataFrame(index=calendar)
        features['vix'] = joined.VIXCLS
        features['rate_change'] = joined.DGS10.diff(21)
        features['curve'] = joined.DGS10 - joined.DGS2
        features['credit'] = joined.BAA10Y
        features['credit_change'] = joined.BAA10Y.diff(21)
        # Pair implied/realised readings at the same dated observation, then lag both.
        features['implied_relative'] = np.log(features.vix / (f.market_vol*np.sqrt(252)))
        available = pd.DatetimeIndex(p.cutoff) - pd.Timedelta(days=lag)
        for col in MACRO:
            p[col] = features[col].reindex(available, method='ffill', tolerance=pd.Timedelta(days=7)).to_numpy()
    p = p.loc['1974':'2022'].dropna()
    assert (p.cutoff < p.start).all()
    assert np.isfinite(p.select_dtypes(include='number')).all().all()
    return p


def forecast(p, macro=False):
    blocks = {'vol': [], 'exposure': EXPOSURE, 'shape': SHAPE, 'exposure_shape': EXPOSURE+SHAPE}
    if macro:
        blocks.update(macro=MACRO, all=EXPOSURE+SHAPE+MACRO)
    rows = []; coefs = []
    for year in range(2000 if macro else 1984, 2023):
        te = p.loc[p.start.dt.year == year].copy()
        if te.empty: continue
        tr = p.loc[p.end <= te.cutoff.min()]
        assert len(tr) >= 260 and tr.end.max() < te.start.min()
        for name, cols in blocks.items():
            columns = ['log_scale'] + cols
            scaler = StandardScaler().fit(tr[columns])
            x = scaler.transform(tr[columns]); z = scaler.transform(te[columns])
            logit = LogisticRegression(C=.1, max_iter=2000, tol=1e-9).fit(x, tr.event.astype(int))
            assert logit.n_iter_[0] < 2000
            quant = QuantileRegressor(quantile=.05, alpha=.01, solver='highs').fit(x, tr.y/tr.scale)
            te['p_' + name] = logit.predict_proba(z)[:, 1]
            te['q_' + name] = quant.predict(z) * te.scale
            for kind, coef in [('logit',logit.coef_[0]),('quantile',quant.coef_)]:
                coefs += [dict(year=year, model=name, kind=kind, feature=c, coefficient=float(v)) for c,v in zip(columns, coef)]
        rows.append(te)
    result = pd.concat(rows)
    frozen = pd.read_parquet(BASE/'evaluation.parquet').reindex(result.index)
    np.testing.assert_allclose(frozen.y, result.y)
    np.testing.assert_array_equal(frozen.event, result.event)
    result['p_empirical'] = frozen.p_fhs5y_ewma
    result['q_empirical'] = frozen.q_fhs5y_ewma
    return result, pd.DataFrame(coefs)


def design(p, dynamic=False, volatility=False):
    m = p.market.to_numpy(); up = np.maximum(m, 0); b = p.bear.to_numpy(float)
    x = [np.ones(len(p)), m, up, b, b*m, b*up]
    if dynamic:
        x += [p.beta_net.to_numpy()*m, p.directional_gap.to_numpy()*up]
    if volatility:
        x += [p.log_scale.to_numpy()*m, p.log_scale.to_numpy()*up]
    return np.column_stack(x)


def conditional(p):
    rows = []
    for year in range(1984, 2023):
        te = p.loc[p.start.dt.year == year].copy()
        if te.empty: continue
        tr = p.loc[p.end <= te.cutoff.min()]
        y = tr[['long','short']].to_numpy()
        for name, dynamic, volatility in [('static',False,False),('dynamic',True,False),
                                          ('vol_response',False,True),('full_response',True,True)]:
            x = design(tr, dynamic, volatility); z = design(te, dynamic, volatility)
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            pred = z @ coef
            for j, leg in enumerate(['long', 'short']):
                te[name+'_'+leg] = pred[:, j]
            te[name] = pred.sum(axis=1)
            np.testing.assert_allclose(te[name], z @ np.linalg.lstsq(x, tr.net, rcond=None)[0], atol=1e-10)
        for leg in ['long','short']:
            intercept = (tr[leg] - tr['beta_'+leg]*tr.market).mean()
            te['rolling_'+leg] = intercept + te['beta_'+leg]*te.market
        te['rolling'] = te.rolling_long + te.rolling_short
        rows.append(te)
    return pd.concat(rows)


def masks(p):
    return {'all':np.ones(len(p),bool), 'bear':p.bear.to_numpy(bool),
            'bear_rebound':(p.bear & p.market.gt(2)).to_numpy(),
            '1984_1999':p.end.dt.year.between(1984,1999).to_numpy(),
            '2000_2016':p.end.dt.year.between(2000,2016).to_numpy(),
            '2017_2022':p.end.dt.year.between(2017,2022).to_numpy(),
            'exclude_crises':(~(p.end.dt.year.between(2008,2009) | p.end.dt.year.eq(2020))).to_numpy()}


def interval(values, mask, block):
    values = np.asarray(values); mask = np.asarray(mask, bool)
    rng = np.random.default_rng(SEED); n = len(values)
    starts = rng.integers(0, n, size=(1000,int(np.ceil(n/block))))
    idx = (starts[:,:,None]+np.arange(block)).reshape(1000,-1)[:,:n] % n
    den = mask[idx].sum(axis=1); valid = den > 0
    draws = np.where(mask[idx], values[idx], 0).sum(axis=1)[valid] / den[valid]
    lo, hi = np.quantile(draws,[.025,.975])
    return dict(delta=float(values[mask].mean()), lo=float(lo), hi=float(hi))


def evaluate(p, condition=False):
    losses = {}; rows=[]; pairs=[]
    if condition:
        names = ['static','rolling','dynamic','vol_response','full_response']
        for name in names:
            error = p[name] - p.net
            losses[name] = {'MAE':abs(error).to_numpy(), 'MSE':(error**2).to_numpy()}
        comparisons = [('dynamic','static'),('dynamic','rolling'),('rolling','static'),
                       ('dynamic','vol_response'),('full_response','vol_response')]
    else:
        names = [c[2:] for c in p if c.startswith('p_')]
        for name in names:
            prob = p['p_'+name].clip(1e-12,1-1e-12).to_numpy(); y = p.event.to_numpy(float)
            residual = (p.y-p['q_'+name]).to_numpy()
            losses[name] = {'Brier':(prob-y)**2, 'logloss':-(y*np.log(prob)+(1-y)*np.log1p(-prob)),
                            'pinball':np.maximum(.05*residual,-.95*residual)}
        comparisons = [(name,baseline) for name in names if name not in ['vol','empirical'] for baseline in ['vol','empirical']]
    for subset, mask in masks(p).items():
        if mask.sum() < 20: continue
        for name in names:
            row = dict(subset=subset, model=name, n=int(mask.sum()), **{k:float(v[mask].mean()) for k,v in losses[name].items()})
            if not condition:
                row.update(event_rate=float(p.event.iloc[mask].mean()), mean_p=float(p['p_'+name].iloc[mask].mean()),
                           q05_breach=float((p.y.iloc[mask]<p['q_'+name].iloc[mask]).mean()))
            rows.append(row)
        for name, baseline in comparisons:
            for metric in losses[name]:
                diff = losses[name][metric]-losses[baseline][metric]
                yearly = pd.Series(diff[mask], index=p.end.iloc[mask].dt.year).groupby(level=0).mean()
                for block in [13,26,52]:
                    pairs.append(dict(subset=subset, model=name, baseline=baseline, metric=metric, block=block,
                                      n=int(mask.sum()), years_better=int((yearly<0).sum()), years=len(yearly),
                                      **interval(diff,mask,block)))
    return pd.DataFrame(rows),pd.DataFrame(pairs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(BASE/'daily.parquet')
    p = panel(d)
    # Independently rebuild as-of state with future daily data removed.
    for cutoff in ['1990-12-28','2009-03-27','2020-03-27','2022-12-30']:
        date = d.loc[:cutoff].index[-1]
        a = daily_features(d).loc[date]; b = daily_features(d.loc[:date]).iloc[-1]
        np.testing.assert_allclose(a, b, equal_nan=True)
    p.to_parquet(OUT/'panel.parquet')
    for window in [252,126,504]:
        cp = p if window==252 else panel(d,window)
        pred = conditional(cp); score, pair = evaluate(pred,True)
        pred.to_parquet(OUT/f'conditional_{window}.parquet')
        score.to_json(OUT/f'conditional_scores_{window}.json',orient='records',indent=2)
        pair.to_json(OUT/f'conditional_pairs_{window}.json',orient='records',indent=2)
        print('CONDITIONAL', window, '\n', score.to_string(index=False), flush=True)
    pred, coef = forecast(p)
    score, pair = evaluate(pred)
    pred.to_parquet(OUT/'forecasts.parquet'); coef.to_json(OUT/'coefficients.json',orient='records',indent=2)
    score.to_json(OUT/'scores.json',orient='records',indent=2); pair.to_json(OUT/'pairs.json',orient='records',indent=2)
    print('FORWARD\n',score.to_string(index=False),flush=True)
    try:
        macro = get_macro()
        for lag in [7,14]:
            mp = panel(d,macro=macro,lag=lag)
            pred, coef = forecast(mp,True); score,pair = evaluate(pred)
            pred.to_parquet(OUT/f'macro_forecasts_{lag}.parquet')
            score.to_json(OUT/f'macro_scores_{lag}.json',orient='records',indent=2)
            pair.to_json(OUT/f'macro_pairs_{lag}.json',orient='records',indent=2)
            coef.to_json(OUT/f'macro_coefficients_{lag}.json',orient='records',indent=2)
            print('MACRO', lag, '\n', score.to_string(index=False),flush=True)
    except (OSError, ValueError, AssertionError) as e:
        (OUT/'macro_failure.txt').write_text(repr(e))
        print('MACRO NOT COMPLETE:',repr(e),flush=True)
    manifest = dict(protocol='ECONOMIC_EXTENSION_PROTOCOL.md; exploratory, not fresh holdout',
                    feature_prefix_checks='PASS at four historical cutoffs',
                    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),BASE/'daily.parquet',BASE/'evaluation.parquet',ROOT/'research/ECONOMIC_EXTENSION_PROTOCOL.md']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
