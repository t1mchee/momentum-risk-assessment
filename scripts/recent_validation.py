"""Frozen recent-period evaluation; see research/RECENT_VALIDATION_PROTOCOL.md."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from expanded_research import data, logistic
from distribution_anatomy import french
from economic_extension import daily_features, design

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/recent_validation'

def replay(new, file, cols):
    old=pd.read_parquet(ROOT/file)
    np.testing.assert_allclose(new.loc[old.index,cols],old[cols],rtol=1e-8,atol=1e-9)
    return dict(file=file,rows=len(old),columns=cols,max_error=float(np.abs(new.loc[old.index,cols]-old[cols]).max().max()))

def loss(y,p):
    p=np.clip(p,1e-12,1-1e-12)
    return (y-p)**2, -(y*np.log(p)+(1-y)*np.log1p(-p))

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    d,p=data(); legs=french('portfolios.zip')
    d['sigma']=d.ewma
    for size,label in [('SMALL','small'),('BIG','big')]:
        d[label+'_long']=legs[size+' HiPRIOR'];d[label+'_short']=-legs[size+' LoPRIOR']
    d['long']=(d.small_long+d.big_long)/2;d['short']=(d.small_short+d.big_short)/2
    d['net']=d.long+d.short
    old=pd.read_parquet(ROOT/'research/notebook_v8/daily.parquet')
    np.testing.assert_allclose(d.loc[old.index,old.columns],old,atol=1e-10)
    rows=[]
    p['log_ewma']=np.log(p.ewma)
    for year in range(1984,int(p.start.dt.year.max())+1):
        te=p.loc[p.start.dt.year==year].copy();tr=p.loc[p.end<=te.cutoff.min()]
        te['p_frequency']=tr.event.mean();te['p_logit_ewma']=logistic(tr,te,['log_ewma'])
        shocks=(tr.y/(np.sqrt(5)*tr.ewma)).tail(260).to_numpy();scale=np.sqrt(5)*te.ewma
        te['p_fhs5y_ewma']=[((shocks < -2.16/s).sum()+.5)/(len(shocks)+1) for s in scale]
        te['q_fhs5y_ewma']=scale*np.quantile(shocks,.05);rows.append(te)
    baseline=pd.concat(rows)
    qa=[replay(baseline,'research/notebook_v8/evaluation.parquet',['p_frequency','p_logit_ewma','p_fhs5y_ewma','q_fhs5y_ewma'])]
    dates=pd.Series(d.index,index=d.index)
    p=d[['long','short','net']].resample('W-FRI').sum(min_count=1)
    p['start']=dates.resample('W-FRI').min();p['end']=dates.resample('W-FRI').max();p['cutoff']=p.end.shift()
    p['market']=d.market.resample('W-FRI').apply(lambda x:np.expm1(np.log1p(x/100).sum())*100)
    p['bear']=d.market_2y.reindex(pd.DatetimeIndex(p.cutoff)).to_numpy()<0
    p['scale']=np.sqrt(5)*d.sigma.reindex(pd.DatetimeIndex(p.cutoff)).to_numpy();p['log_scale']=np.log(p.scale)
    p['y']=d.Mom.resample('W-FRI').sum(min_count=1);p['event']=p.y < -2.16
    f=daily_features(d)
    for col in f:p[col]=f[col].reindex(pd.DatetimeIndex(p.cutoff)).to_numpy()
    p['bear_up_beta']=p.bear.astype(float)*p.up_beta_net
    p=p.loc['1974':d.index.max()].dropna();rows=[]
    for year in range(1984,int(p.start.dt.year.max())+1):
        te=p.loc[p.start.dt.year==year].copy();tr=p.loc[p.end<=te.cutoff.min()]
        for name,dynamic,vol in [('dynamic',True,False),('vol_response',False,True)]:
            pred=design(te,dynamic,vol)@np.linalg.lstsq(design(tr,dynamic,vol),tr[['long','short']],rcond=None)[0]
            for j,leg in enumerate(['long','short']):te[name+'_'+leg]=pred[:,j]
            te[name]=pred.sum(axis=1)
        for leg in ['long','short']:te['rolling_'+leg]=(tr[leg]-tr['beta_'+leg]*tr.market).mean()+te['beta_'+leg]*te.market
        te['rolling']=te.rolling_long+te.rolling_short;rows.append(te)
    conditional=pd.concat(rows)
    qa.append(replay(conditional,'research/economic_extension/conditional_252.parquet',['rolling','dynamic','vol_response']))
    w=d[['net','sigma']].resample('W-FRI').agg({'net':'sum','sigma':'last'}).loc['1974':d.index.max()]
    w['log_scale']=np.log(w.sigma)
    for col in ['beta_net','directional_gap','shape_left','shape_asym']:w[col]=f[col].resample('W-FRI').last().reindex(w.index)
    w['score']=w.net.rolling(4).sum()/(w.sigma*np.sqrt(20));records=[]
    for t,r in w.iterrows():
        j=d.index.searchsorted(t,side='right');future=d.net.iloc[j:j+20]
        if len(future)==20:records.append(dict(date=t,path_min=min(0.,future.cumsum().min()),terminal=future.sum(),end=future.index[-1]))
    q=w.join(pd.DataFrame(records).set_index('date')).dropna();q['y']=(q.path_min<=-5).astype(int);rows=[]
    for year in range(1984,int(q.index.year.max())+1):
        te=q.loc[q.index.year==year].copy();tr=q.loc[q.end<=te.index.min()]
        sc=StandardScaler().fit(tr[['log_scale']]);m=LogisticRegression(C=.1,max_iter=2000,tol=1e-9).fit(sc.transform(tr[['log_scale']]),tr.y)
        assert m.n_iter_[0]<2000
        te['p_vol']=m.predict_proba(sc.transform(te[['log_scale']]))[:,1];rows.append(te)
    path=pd.concat(rows);values=[]
    for t,r in path.iterrows():
        tr=q.loc[q.end<=t].tail(1260);sim=tr.path_min/tr.sigma*r.sigma
        values.append(((sim<=-5).sum()+.5)/(len(sim)+1))
    path['p_fhs']=values
    qa.append(replay(path,'research/local_stress/barrier_fixed.parquet',['p_vol','p_fhs']))
    scores=[];pairs=[]
    for name,z,models,target in [('weekly',baseline,['p_fhs5y_ewma','p_logit_ewma','p_frequency'],'event'),('path',path,['p_fhs','p_vol'],'y'),('conditional',conditional,['rolling','dynamic','vol_response'],'net')]:
        z.to_parquet(OUT/(name+'.parquet'));recent=z.loc[z.index.year>=2023]
        for subset,a in [('2023+',recent)]+[(str(year),recent.loc[recent.index.year==year]) for year in sorted(set(recent.index.year))]:
            for model in models:
                rec=dict(component=name,period=subset,model=model,n=len(a),first=str(a.index.min().date()),last=str(a.index.max().date()))
                if name=='conditional':rec.update(mae=float((a.net-a[model]).abs().mean()),mse=float(((a.net-a[model])**2).mean()))
                else:
                    b,l=loss(a[target],a[model]);rec.update(events=int(a[target].sum()),predicted=float(a[model].mean()),observed=float(a[target].mean()),brier=float(b.mean()),logloss=float(l.mean()))
                if name=='weekly' and model=='p_fhs5y_ewma':
                    err=a.y-a.q_fhs5y_ewma;rec.update(q05_breaches=int((err<0).sum()),q05_breach_rate=float((err<0).mean()),pinball=float(np.maximum(.05*err,-.95*err).mean()),bear_weeks=int(a.bear.sum()),bear_events=int(a.loc[a.bear,'event'].sum()))
                if name=='path':rec.update(endpoint_events=int((a.terminal<=-5).sum()),recovered=int(((a.y==1)&(a.terminal>-5)).sum()))
                scores.append(rec)
        comparisons=[(models[0],models[1])]+([('dynamic','vol_response')] if name=='conditional' else [])
        for model,base in comparisons:
            for metric in (['mae','mse'] if name=='conditional' else ['brier','logloss']):
                if name=='conditional':v=(recent.net-recent[model]).abs()**(1 if metric=='mae' else 2)-(recent.net-recent[base]).abs()**(1 if metric=='mae' else 2)
                else:v=loss(recent[target],recent[model])[metric=='logloss']-loss(recent[target],recent[base])[metric=='logloss']
                v=v.reindex(pd.date_range(recent.index.min(),recent.index.max(),freq='W-FRI')).to_numpy()
                for block in [13,26]:
                    rng=np.random.default_rng(7281);boot=[]
                    for _ in range(1000):
                        starts=rng.integers(len(v),size=int(np.ceil(len(v)/block)));ix=((starts[:,None]+np.arange(block))%len(v)).ravel()[:len(v)]
                        boot.append(np.nanmean(v[ix]))
                    pairs.append(dict(component=name,model=model,baseline=base,metric=metric,block=block,difference=float(np.nanmean(v)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975))))
    pd.DataFrame(scores).to_json(OUT/'scores.json',orient='records',indent=2)
    pd.DataFrame(pairs).to_json(OUT/'pairs.json',orient='records',indent=2)
    provenance={name:hashlib.sha256((ROOT/'data/raw'/name).read_bytes()).hexdigest() for name in ['momentum.zip','factors.zip','portfolios.zip']}
    (OUT/'QA.json').write_text(json.dumps(dict(status='PASS',last_source_date=str(d.index.max().date()),replay=qa,source_sha256=provenance,protocol_sha256=hashlib.sha256((ROOT/'research/RECENT_VALIDATION_PROTOCOL.md').read_bytes()).hexdigest()),indent=2))
    print(pd.DataFrame(scores).query("period=='2023+'").to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))

if __name__=='__main__':main()
