"""Numerical, time-boundary and source-preservation checks for extension."""
import hashlib
import json
import numpy as np
import pandas as pd
from economic_extension import ROOT, BASE, OUT, daily_features, panel, forecast, conditional, design, get_macro
from economic_companion import EconomicCompanion


def main():
    checks=[]
    originals={BASE/'evaluation.parquet':'39aea934551de5dea06824db25539a08a51d057f0e7af78aec087c36bdf89220',
               BASE/'scenarios.json':'e90bd18d504080cf25eaf35e11c3831cbe917a81274607548b9fc45ec6da8246',
               BASE/'validation/scenario_predictions.parquet':'2f42a4d86d6f4e2a170b9bdf27dae06596e1af757278c50cf58cdf82b8bb9644'}
    for path,h in originals.items(): assert hashlib.sha256(path.read_bytes()).hexdigest()==h
    checks.append('Original Notebook 08 evaluation, scenarios and validation predictions byte-identical')
    d=pd.read_parquet(BASE/'daily.parquet');p=pd.read_parquet(OUT/'panel.parquet')
    f=daily_features(d)
    for date in ['1984-01-03','2009-03-27','2020-03-27','2022-12-30']:
        prefix=d.loc[:date]; x=prefix.tail(252)
        coef=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x.market,x.market.clip(lower=0)]),x[['long','short']],rcond=None)[0]
        np.testing.assert_allclose(f.loc[date,['up_beta_long','up_beta_short']].to_numpy(float),coef[1]+coef[2],atol=1e-10)
        np.testing.assert_allclose(f.loc[date],daily_features(prefix).iloc[-1],atol=1e-10,equal_nan=True)
    checks.append('Rolling directional betas independently reproduced by direct daily OLS; four prefix checks')
    altered=d.copy();altered.loc[altered.index>'2009-03-27',['long','short','net','Mom','market']] *= -10.
    np.testing.assert_allclose(daily_features(altered).loc[:'2009-03-27'],f.loc[:'2009-03-27'],equal_nan=True)
    cp=p.copy();cp.loc[cp.end>'2009-03-27',['long','short']]=456.
    cp['net']=cp.long+cp.short
    rebuilt=conditional(cp)
    stored=pd.read_parquet(OUT/'conditional_252.parquet')
    columns=['static','rolling','dynamic','vol_response','full_response']
    pd.testing.assert_frame_equal(rebuilt.loc[rebuilt.end<='2009-03-27',columns],stored.loc[stored.end<='2009-03-27',columns])
    checks.append('Future daily prices/returns and future weekly outcomes cannot change earlier states/responses')
    short,_=forecast(p.loc[p.end<'1987'])
    full=pd.read_parquet(OUT/'forecasts.parquet')
    columns=[c for c in short if c.startswith(('p_','q_'))]
    pd.testing.assert_frame_equal(short[columns],full.loc[short.index,columns])
    checks.append('Forward first-three-year prefix replay matches full saved logistic and quantile predictions')
    macro=get_macro();mp=panel(d,macro=macro)
    changed=macro.copy();changed.loc[changed.index>'2019-12-31']=999.
    mp2=panel(d,macro=changed)
    before=mp.index[mp.cutoff<='2019-12-31']
    pd.testing.assert_frame_equal(mp.loc[before],mp2.loc[before])
    checks.append('Macro future perturbation leaves earlier lagged inputs unchanged; dated source hashes recorded')
    e=EconomicCompanion(ROOT)
    for date in ['1990-12-31','2009-03-31','2020-03-31','2022-12-31']:
        sc=e.scenarios(date);np.testing.assert_allclose(sc.net,sc.long+sc.short)
        assert (sc.mean_lo<=sc.mean_hi).all()
        assert (sc.nearby_joint_recent<=sc.nearby_joint).all()
    # Replay arbitrary historical conditional forecasts using annual coefficients.
    for i in [0,400,1000,1700,2034]:
        row=stored.iloc[[i]];coef,_=e.fitted(int(row.start.iloc[0].year))
        np.testing.assert_allclose((design(row,True)@coef).sum(axis=1),row.dynamic,atol=1e-10)
    checks.append('Dated scenario engine replays saved conditional predictions; additive legs and ordered mean intervals')
    score=pd.read_json(OUT/'scores.json')
    for name in ['vol','exposure','shape','exposure_shape','empirical']:
        assert full['p_'+name].between(0,1).all()
        residual=full.y-full['q_'+name]
        computed=np.where(residual>=0,.05*residual,-.95*residual).mean()
        expected=score.loc[(score.subset=='all')&(score.model==name),'pinball'].iloc[0]
        assert abs(computed-expected)<1e-9
    checks.append('Probabilities bounded; independent q05 pinball calculation agrees')
    artifacts=list(OUT.glob('*.parquet'))+list(OUT.glob('*scores*.json'))+list(OUT.glob('*pairs*.json'))
    sources=[ROOT/'scripts'/n for n in ['economic_extension.py','economic_macro_scenarios.py','economic_companion.py','test_economic_extension.py']]
    result=dict(status='PASS',checks=checks,limitations=['No untouched holdout','No archive-vintage authentication','No human PM trial','Intervals conditional on fixed feature estimates; not search-adjusted'],
                hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts+sources})
    (OUT/'QA.json').write_text(json.dumps(result,indent=2))
    print('\n'.join(['PASS']+checks))


if __name__=='__main__':main()
