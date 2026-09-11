"""Conditional macro response, distinct from predicting macro moves or losses."""
import json
import numpy as np
import pandas as pd
from economic_extension import OUT, BASE, get_macro, panel, interval, masks


def prepare():
    d = pd.read_parquet(BASE/'daily.parquet')
    p = panel(d)
    macro = get_macro()
    for col, name in [('DGS10','rates'),('BAA10Y','credit')]:
        s = macro[col].dropna()
        end = s.reindex(pd.DatetimeIndex(p.end), method='ffill', tolerance=pd.Timedelta(days=7)).to_numpy()
        start = s.reindex(pd.DatetimeIndex(p.cutoff), method='ffill', tolerance=pd.Timedelta(days=7)).to_numpy()
        p[name] = end-start
    return p.dropna()


def forecasts(p, window=260, lag=7):
    rows=[]; coefficients=[]
    blocks={'market':['market'], 'rates':['market','rates'], 'credit':['market','credit'], 'both':['market','rates','credit']}
    for year in range(2000,2023):
        te=p.loc[p.start.dt.year==year].copy()
        tr=p.loc[p.end<=te.cutoff.min()-pd.Timedelta(days=lag)].iloc[-window:]
        assert len(tr)==window
        for name,columns in blocks.items():
            x=np.column_stack([np.ones(len(tr)),tr[columns]])
            z=np.column_stack([np.ones(len(te)),te[columns]])
            coef=np.linalg.lstsq(x,tr[['long','short']],rcond=None)[0]
            fit=z@coef
            for j,leg in enumerate(['long','short']):
                te[name+'_'+leg]=fit[:,j]
                for c,v in zip(['intercept']+columns,coef[:,j]):
                    coefficients.append(dict(year=year,model=name,leg=leg,driver=c,coefficient=float(v)))
            te[name+'_net']=fit.sum(axis=1)
            np.testing.assert_allclose(fit.sum(axis=1),z@np.linalg.lstsq(x,tr.net,rcond=None)[0],atol=1e-10)
        rows.append(te)
    return pd.concat(rows),pd.DataFrame(coefficients)


def evaluate(p):
    rows=[]; pairs=[]
    for subset,mask in masks(p).items():
        if mask.sum()<20:continue
        for leg in ['long','short','net']:
            for name in ['market','rates','credit','both']:
                error=(p[name+'_'+leg]-p[leg]).to_numpy()
                rows.append(dict(subset=subset,leg=leg,model=name,n=int(mask.sum()),
                                 MAE=float(abs(error[mask]).mean()),RMSE=float(np.sqrt((error[mask]**2).mean()))))
                if name=='market':continue
                for metric,difference in [('MAE',abs(error)-abs(p['market_'+leg]-p[leg]).to_numpy()),
                                          ('MSE',error**2-(p['market_'+leg]-p[leg]).to_numpy()**2)]:
                    for block in [26,52]:
                        pairs.append(dict(subset=subset,leg=leg,model=name,metric=metric,block=block,
                                          **interval(difference,mask,block)))
    return pd.DataFrame(rows),pd.DataFrame(pairs)


def main():
    p=prepare()
    for window,lag in [(260,7),(156,7),(520,7),(260,14)]:
        pred,coefs=forecasts(p,window,lag)
        score,pairs=evaluate(pred)
        suffix=f'{window}_{lag}'
        pred.to_parquet(OUT/f'macro_scenario_{suffix}.parquet')
        coefs.to_json(OUT/f'macro_scenario_coefficients_{suffix}.json',orient='records',indent=2)
        score.to_json(OUT/f'macro_scenario_scores_{suffix}.json',orient='records',indent=2)
        pairs.to_json(OUT/f'macro_scenario_pairs_{suffix}.json',orient='records',indent=2)
        print(suffix,'\n',score.query('leg=="net"').to_string(index=False),flush=True)


if __name__=='__main__':main()
