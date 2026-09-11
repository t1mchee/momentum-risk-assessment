"""Read-only upstream constituent PCA. See research/PCA_PROTOCOL.md."""
from pathlib import Path
import pickle, json, hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distribution_anatomy import load_companies, VERIFIED_ACTIONS
from distribution_panel import tail_weights

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/constituent_pca'
MEMBERS=Path('SOURCE_DATA_ROOT/data/processed/leg_members.pkl')


def pca(x, weights, market, residual=False, correlation=True, sectors=None):
    a=np.asarray(x,float);m=np.asarray(market,float)
    design=np.column_stack([np.ones(len(m)),m])
    fit=design@np.linalg.lstsq(design,a,rcond=None)[0]
    if residual:a=a-fit
    # Follow-on diagnostic: remove same-sector peer co-movement, not a causal factor.
    sector_adjusted=0
    if sectors is not None:
        before=a.copy();labels=np.asarray(sectors)
        for sector in np.unique(labels):
            ix=np.flatnonzero(labels==sector)
            if len(ix)<5:continue
            peer=(before[:,ix].sum(axis=1)[:,None]-before[:,ix])/(len(ix)-1)
            peer-=peer.mean(axis=0)
            centered=before[:,ix]-before[:,ix].mean(axis=0)
            denom=(peer**2).sum(axis=0)
            coef=np.divide((peer*centered).sum(axis=0),denom,out=np.zeros(len(ix)),where=denom>1e-12)
            a[:,ix]=centered-peer*coef
            sector_adjusted+=len(ix)
    a=a-a.mean(axis=0)
    sd=a.std(axis=0,ddof=1) if correlation else np.ones(a.shape[1])
    z=a/sd
    cov=z.T@z/(len(z)-1)
    vals,vec=np.linalg.eigh(cov);vals=vals[::-1];vec=vec[:,::-1]
    v=vec[:,0].copy();score=z@v
    anchor=np.corrcoef(score,m)[0,1] if not residual else v[np.argmax(abs(v))]
    if anchor<0:v=-v;score=-score
    load=sd*v
    w=np.asarray(weights);wl=np.maximum(w,0);ws=np.minimum(w,0)
    net=a@w;long=a@wl;short=a@ws
    exposure=float(w@load);l=float(wl@load);s=float(ws@load)
    variance=float(net.var(ddof=1));component=score*exposure
    pcvar=float(vals[0]*exposure**2)
    np.testing.assert_allclose(l+s,exposure,atol=1e-10)
    np.testing.assert_allclose(component.var(ddof=1),pcvar,atol=1e-9)
    np.testing.assert_allclose(variance, np.sum(vals*(vec.T@(sd*w))**2),atol=1e-9)
    np.testing.assert_allclose(component,(-score)*(-exposure))
    total=float(vals.sum()); corr=np.corrcoef(a,rowvar=False)
    avg=float((corr.sum()-len(w))/(len(w)*(len(w)-1)))
    return (dict(pc1_share=float(vals[0]/total),top3_share=float(vals[:3].sum()/total),
                eig_gap=float((vals[0]-vals[1])/vals[0]),average_corr=avg,
                long_exposure=l,short_exposure=s,net_exposure=exposure,
                long_one_sd=l*float(np.sqrt(vals[0])),short_one_sd=s*float(np.sqrt(vals[0])),net_one_sd=exposure*float(np.sqrt(vals[0])),
                sector_adjusted_names=sector_adjusted,
                pc1_book_variance_share=float(pcvar/variance),book_variance=variance,
                beta_net=float(np.cov(net,m)[0,1]/np.var(m,ddof=1)),
                gross_cancellation=float(1-abs(exposure)/(abs(l)+abs(s))) if abs(l)+abs(s)>0 else None),
           dict(vector=v,sd=sd,score=score,component=component,net=net,long=long,short=short,load=load,
                raw_mean=np.asarray(x).mean(axis=0),market_fit=fit))


def select(date,window,members,P,Q,R,factor,S):
    calendar=R.loc[:date].index[-window:]
    observed=P.loc[:date].dropna(how='all').index
    audit=dict(date=str(date.date()),window=window,passed=False)
    if not len(observed):return None,audit
    asof=observed[-1];audit['price_asof']=str(asof.date())
    if (date-asof).days>7:audit['reason']='stale formation snapshot';return None,audit
    names=list(dict.fromkeys(members['winners']+members['losers']))
    names=[n for n in names if n in P]
    side=pd.Series({n:1. if n in members['winners'] else -1. for n in names})
    mv=P.loc[asof,names]*Q.loc[asof,names]
    original_counts={leg:len(members[leg]) for leg in ['winners','losers']}
    audit.update(original_long_names=original_counts['winners'],original_short_names=original_counts['losers'],
                 formation_missing_or_nonpositive=int((~mv.gt(0)).sum())+sum(original_counts.values())-len(names))
    names=mv.index[mv.gt(0)].tolist();mv=mv.loc[names];side=side.loc[names]
    audit['formation_long_name_coverage']=float(side.eq(1).sum()/original_counts['winners'])
    audit['formation_short_name_coverage']=float(side.eq(-1).sum()/original_counts['losers'])
    if min(audit['formation_long_name_coverage'],audit['formation_short_name_coverage'])<.9:
        audit['reason']='positive formation value for fewer than 90% of original leg names';return None,audit
    weight=mv.copy()
    for sign in [1,-1]:
        ix=side.eq(sign);weight.loc[ix]=mv.loc[ix]/mv.loc[ix].sum()*sign
    x=R.loc[calendar,names]
    # Calendar-wide missing return dates, not selected-book missingness.
    dates=calendar[R.loc[calendar].notna().any(axis=1)]
    audit['common_calendar_sessions']=len(dates)
    if len(dates)<.9*window:audit['reason']='calendar return coverage below 90%';return None,audit
    x=x.loc[dates]
    complete=x.notna().all() & x.std().gt(1e-8)
    extreme=x.abs().gt(45).any()
    inferred=factor.loc[calendar,names].ne(1).copy()
    for dt,ticker,_,_ in VERIFIED_ACTIONS:
        if pd.Timestamp(dt) in inferred.index and ticker in inferred:inferred.loc[dt,ticker]=False
    uncertain_action=inferred.any()
    keep=complete & ~extreme & ~uncertain_action
    audit.update(incomplete_names=int((~complete).sum()),extreme_names=int(extreme.sum()),
                 inferred_action_names=int(uncertain_action.sum()))
    coverage={sign:float(weight.loc[keep & side.eq(sign)].abs().sum()) for sign in [1,-1]}
    audit.update(long_weight_coverage=coverage[1],short_weight_coverage=coverage[-1],
                 retained_long=int((keep&side.eq(1)).sum()),retained_short=int((keep&side.eq(-1)).sum()))
    if min(coverage.values())<.8 or min(audit['retained_long'],audit['retained_short'])<30:
        audit['reason']='less than 80% known weight or 30 names in a leg';return None,audit
    weight=weight.loc[keep]
    for sign in [1,-1]:weight.loc[side.loc[weight.index].eq(sign)]/=coverage[sign]
    sectors=S.loc[asof,weight.index].fillna('Unknown')
    audit.update(passed=True,reason='covered-subset proxy only',n_names=len(weight))
    return (x.loc[:,keep],weight,sectors,asof),audit


def tail_attribution(x,component,weight,calendar,date):
    # Do not bridge omitted return dates; centred PCs plus remainder exactly reconstruct raw net.
    net=pd.Series(x.to_numpy()@weight,index=x.index).reindex(calendar)
    pc=pd.Series(component,index=x.index).reindex(calendar)
    rows=[]
    for h in [1,5,10,20]:
        f=pd.DataFrame({'net':net.rolling(h).sum(),'pc1':pc.rolling(h).sum()}).dropna()
        if len(f)<30:continue
        tw=tail_weights(f.net.to_numpy(),.15)
        # Existing helper returns exactly 15% * n units of mass, then normalise.
        tw=tw/tw.sum()
        assert np.isclose(np.sum(tw),1)
        a=float(tw@f.net);b=float(tw@f.pc1)
        rows.append(dict(date=str(date.date()),h=h,n_windows=len(f),tail_mean=a,pc1_tail=b,remainder_tail=a-b))
    return rows


def future(x,weight,details,R,date,factor):
    stop=date+pd.offsets.MonthEnd(1)
    ix=R.index[(R.index>date)&(R.index<=stop)]
    f=R.loc[ix,x.columns]
    complete=f.notna().all(axis=1) & ~f.abs().gt(45).any(axis=1)
    inferred=factor.loc[ix,x.columns].ne(1)
    for dt,ticker,_,_ in VERIFIED_ACTIONS:
        if pd.Timestamp(dt) in inferred.index and ticker in inferred:inferred.loc[dt,ticker]=False
    complete &= ~inferred.any(axis=1)
    valid=f.loc[complete]
    result=dict(date=str(date.date()),future_sessions=len(ix),future_usable=len(valid),future_pass=False)
    if len(ix)<15 or len(valid)<.8*len(ix):return result
    if valid.std().le(1e-8).any():return result
    z=(valid.to_numpy()-details['raw_mean'])/details['sd']
    score=z@details['vector'];pc=score*float(weight@details['load']);net=valid.to_numpy()@weight
    corr=valid.corr().to_numpy();n=len(weight)
    result.update(future_pass=True,future_average_corr=float((corr.sum()-n)/(n*(n-1))),
                  future_book_sd=float(np.std(net,ddof=1)),future_pc1_sd=float(np.std(pc,ddof=1)),
                  future_pc1_net_corr=float(np.corrcoef(pc,net)[0,1]) if np.std(pc)>1e-10 else None,
                  future_lower15=float((tail_weights(net,.15)/tail_weights(net,.15).sum())@net))
    # Frozen PCs need not be orthogonal out of sample: no additive future variance-share claim.
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(ROOT/'research/notebook_v8/daily.parquet').loc['2013':'2022']
    print('Loading audited 2013–2022 holdings panels',flush=True)
    P,W,S,R,raw,factor,audit=load_companies(d.index)
    # Recover unrounded position quantity from source with same duplicate quarantine.
    src=Path('SOURCE_DATA_ROOT/data/raw/ishares/IWV/panel')
    parts=[pd.read_parquet(src/f'IWV_{year}.parquet',columns=['as_of','ticker','quantity']) for year in range(2013,2023)]
    q=pd.concat(parts,ignore_index=True);q=q.loc[~q.duplicated(['as_of','ticker'],keep=False)&q.ticker.notna()&q.ticker.ne('-')]
    Q=q.pivot(index='as_of',columns='ticker',values='quantity').reindex(index=P.index,columns=P.columns).where(lambda x:x>0)
    del parts,q,W,raw
    books=pickle.loads(MEMBERS.read_bytes());books={pd.Timestamp(k):v for k,v in books.items() if pd.Timestamp(k)<pd.Timestamp('2023-01-01')}
    rows=[];gates=[];tails=[];future_rows=[];loadings=[]
    with threadpool_limits(limits=1):
        for date,members in sorted(books.items()):
            for window in [252,126]:
                chosen,gate=select(date,window,members,P,Q,R,factor,S);gates.append(gate)
                if chosen is None:continue
                x,weight,sectors,asof=chosen;m=d.market.reindex(x.index)
                sector_long=weight.clip(lower=0).groupby(sectors).sum();sector_short=(-weight.clip(upper=0)).groupby(sectors).sum()
                for kind,residual,correlation in [('raw_correlation',False,True),('raw_covariance',False,False),('residual_correlation',True,True),('market_sector_residual',True,True)]:
                    values,details=pca(x,weight,m,residual,correlation,sectors if kind=='market_sector_residual' else None)
                    rows.append(dict(date=str(date.date()),window=window,kind=kind,**values,
                                     long_sector_hhi=float((sector_long**2).sum()),short_sector_hhi=float((sector_short**2).sum()),
                                     long_weight_hhi=float((weight.clip(lower=0)**2).sum()),short_weight_hhi=float((weight.clip(upper=0)**2).sum()),
                                     n_names=len(weight),n_sessions=len(x)))
                    if kind=='raw_correlation' and window==252:
                        tails.extend(tail_attribution(x,details['component'],weight.to_numpy(),R.loc[:date].index[-window:],date))
                        future_rows.append(future(x,weight.to_numpy(),details,R,date,factor))
                    if window==252 and kind!='raw_covariance':
                        for name,v,load,w in zip(x.columns,details['vector'],details['load'],weight):
                            loadings.append(dict(date=str(date.date()),kind=kind,ticker=name,sector=str(sectors[name]),weight=float(w),
                                                 eigenvector=float(v),return_loading=float(load),exposure_contribution=float(w*load)))
                    if window==252 and date==max(books):
                        # Latest accepted panel supports independent numerical QA and bootstrap;
                        # no new market data or forward observations are read.
                        x.to_parquet(OUT/'latest_return_panel.parquet')
                        pd.DataFrame({'weight':weight,'sector':sectors}).to_parquet(OUT/'latest_weights.parquet')
                # Equal-weight sensitivity on the exact same accepted names and dates.
                ew=np.where(weight>0,1/(weight>0).sum(),-1/(weight<0).sum())
                values,_=pca(x,ew,m)
                rows.append(dict(date=str(date.date()),window=window,kind='equal_weight_same_names',**values,n_names=len(weight),n_sessions=len(x)))
            if date.month==12:print(date.year,'passed so far',sum(g['passed'] for g in gates),'/',len(gates),flush=True)
    pd.DataFrame(gates).to_json(OUT/'gates.json',orient='records',indent=2)
    pd.DataFrame(rows).to_parquet(OUT/'metrics.parquet')
    pd.DataFrame(tails).to_json(OUT/'tails.json',orient='records',indent=2)
    pd.DataFrame(future_rows).to_json(OUT/'future.json',orient='records',indent=2)
    pd.DataFrame(loadings).to_parquet(OUT/'loadings.parquet')
    (OUT/'source_audit.json').write_text(json.dumps(audit,indent=2))
    (OUT/'manifest.json').write_text(json.dumps(dict(status='exploratory covered-subset PCA, no clean-book certification',
         hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MEMBERS,Path(__file__),ROOT/'research/PCA_PROTOCOL.md']},
         checks='Eigen-risk identity, signed leg additivity, eigenvector-sign invariance passed for every fitted matrix'),indent=2))
    print(pd.DataFrame(gates).groupby(['window','passed']).size().to_string(),flush=True)


if __name__=='__main__':main()
