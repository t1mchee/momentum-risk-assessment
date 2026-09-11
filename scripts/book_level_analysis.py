"""Book-level explanatory features; no forward model or upstream writes."""
from pathlib import Path
import json, pickle, hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from constituent_pca import select, MEMBERS
from distribution_anatomy import load_companies, VERIFIED_ACTIONS
from distribution_panel import tail_weights

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research/book_level'


def tail(x,w,sectors,calendar,h):
    contributions=x.mul(w).reindex(calendar).rolling(h,min_periods=h).sum()
    a=contributions.dropna();y=a.sum(axis=1);mass=tail_weights(y.to_numpy());z=mass/mass.sum()
    means=pd.Series(z@a.to_numpy(),index=a.columns);adverse=(-means).clip(lower=0)
    gross=float(adverse.sum());top=adverse.sort_values(ascending=False).head(5)
    long=a.loc[:,w>0].sum(axis=1);short=a.loc[:,w<0].sum(axis=1)
    positions=pd.Series(np.arange(len(calendar)),index=calendar).reindex(a.index).to_numpy()
    selected=np.flatnonzero(mass>0);groups=[]
    for i in selected:
        if not groups or positions[i]-h+1>positions[groups[-1][-1]]:groups.append([int(i)])
        else:groups[-1].append(int(i))
    loss=mass*np.maximum(-y.to_numpy(),0);amounts=np.array([loss[g].sum() for g in groups]);big=int(amounts.argmax())
    remaining=mass.copy();remaining[groups[big]]=0
    omitted=None
    if remaining.sum()>0:
        rm=pd.Series((remaining/remaining.sum())@a.to_numpy(),index=a.columns)
        rad=(-rm).clip(lower=0);omitted=dict(tail=float(rm.sum()),top_name=str(rad.idxmax()),
            top5_share=float(rad.nlargest(5).sum()/rad.sum()) if rad.sum()>0 else None)
    worst=str(adverse.idxmax());without=y-a[worst];newmass=tail_weights(without.to_numpy());newmass/=newmass.sum()
    sectors_net=means.groupby(sectors).sum().sort_values()
    np.testing.assert_allclose(means.sum(),z@y,atol=1e-9)
    np.testing.assert_allclose(float(z@without),float(z@y-means[worst]),atol=1e-9)
    return dict(h=h,n_windows=len(y),tail_mean=float(z@y),gross_adverse=gross,
        long_tail=float(z@long),short_tail=float(z@short),top5_share=float(top.sum()/gross) if gross>0 else None,
        top_names={str(k):float(v) for k,v in top.items()},sector_net={str(k):float(v) for k,v in sectors_net.items()},
        both_legs_lose_share=float(z@((long<0)&(short<0)).to_numpy()),episodes=len(groups),
        largest_episode_share=float(amounts[big]/amounts.sum()) if amounts.sum()>0 else None,
        omit_episode=omitted,largest_name=worst,omit_name_fixed_tail=float(z@without),
        omit_name_reselected_tail=float(newmass@without))


def subspace(x,w,market,residual=False,k=3):
    a=x.to_numpy().copy();m=np.asarray(market)
    if residual:
        design=np.column_stack([np.ones(len(m)),m]);a-=design@np.linalg.lstsq(design,a,rcond=None)[0]
    a-=a.mean(axis=0);sd=a.std(axis=0,ddof=1);z=a/sd
    c=z.T@z/(len(z)-1);ev,v=np.linalg.eigh(c);ev=ev[::-1];v=v[:,::-1]
    score=z@v[:,:k];wl=np.maximum(w,0);ws=np.minimum(w,0)
    l=score@(v[:,:k].T@(sd*wl));s=score@(v[:,:k].T@(sd*ws))
    vl=np.var(l,ddof=1);vs=np.var(s,ddof=1);vn=np.var(l+s,ddof=1)
    offset=1-vn/(vl+vs)
    # General orthogonal rotation, not merely sign reversal, leaves projected P&L invariant.
    rng=np.random.default_rng(19);q,_=np.linalg.qr(rng.normal(size=(k,k)))
    rotated=(score@q)@((v[:,:k]@q).T@(sd*w))
    np.testing.assert_allclose(rotated,l+s,atol=1e-9)
    return dict(k=k,market_removed=residual,net_component_sd=float(np.sqrt(vn)),
        gross_leg_sd=float(np.sqrt(vl+vs)),offset=float(offset),
        net_variance_share=float(vn/np.var(a@w,ddof=1)),
        boundary_gap=float((ev[k-1]-ev[k])/ev[k-1]))


def bridge(old,new,R,factor):
    d0,x0,w0=old;d1,x1,w1=new
    result=dict(previous=str(d0.date()),date=str(d1.date()),passed=False)
    if d0+pd.offsets.MonthEnd(1)!=d1:result['reason']='not consecutive accepted calendar formations';return result
    names=w0.index.union(w1.index)
    frames=[]
    for d in [d0,d1]:
        cal=R.loc[:d].index[-252:];dates=cal[R.loc[cal].notna().any(axis=1)]
        if len(dates)<.9*252:result['reason']='calendar coverage';return result
        frames.append(R.loc[dates,names])
    keep=np.ones(len(names),bool)
    for f in frames:keep &= f.notna().all().to_numpy() & f.std().gt(1e-8).to_numpy() & ~f.abs().gt(45).any().to_numpy()
    actions=factor.loc[frames[0].index.union(frames[1].index),names].ne(1)
    for dt,ticker,_,_ in VERIFIED_ACTIONS:
        if pd.Timestamp(dt) in actions.index and ticker in actions:actions.loc[dt,ticker]=False
    keep &= ~actions.any().to_numpy()
    names=names[keep];coverage=[];weights=[]
    for w in [w0,w1]:
        ww=w.reindex(names,fill_value=0.).copy()
        for sign in [1,-1]:
            mask=ww*sign>0;covered=float(ww[mask].abs().sum());coverage.append(covered)
            if covered>0:ww.loc[mask]/=covered
        weights.append(ww.to_numpy())
    result['coverage_old_long_short_new_long_short']=coverage
    if min(coverage)<.95:result['reason']='common union fails 95% endpoint leg-weight gate';return result
    cov=[f.loc[:,names].cov().to_numpy() for f in frames];a,b=weights
    A=float(a@cov[0]@a);B=float(b@cov[0]@b);C=float(a@cov[1]@a);D=float(b@cov[1]@b)
    weight=.5*((B-A)+(D-C));covariance=.5*((C-A)+(D-B))
    per_name=.5*(b-a)*((cov[0]+cov[1])@(a+b))
    np.testing.assert_allclose(weight+covariance,D-A,atol=1e-9)
    np.testing.assert_allclose(per_name.sum(),weight,atol=1e-9)
    effects=pd.Series(per_name,index=names)
    result.update(passed=True)
    return dict(**result,reason='retrospective common-union bridge',A=A,B=B,C=C,D=D,
                weight_effect=weight,covariance_effect=covariance,total_change=D-A,
                top_increases=effects.nlargest(5).to_dict(),top_decreases=effects.nsmallest(5).to_dict())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(ROOT/'research/notebook_v8/daily.parquet').loc['2013':'2022']
    P,W,S,R,raw,factor,audit=load_companies(d.index)
    src=Path('SOURCE_DATA_ROOT/data/raw/ishares/IWV/panel')
    q=pd.concat([pd.read_parquet(src/f'IWV_{y}.parquet',columns=['as_of','ticker','quantity']) for y in range(2013,2023)],ignore_index=True)
    q=q.loc[~q.duplicated(['as_of','ticker'],keep=False)&q.ticker.notna()&q.ticker.ne('-')]
    Q=q.pivot(index='as_of',columns='ticker',values='quantity').reindex(index=P.index,columns=P.columns).where(lambda x:x>0)
    del W,raw,q
    books=pickle.loads(MEMBERS.read_bytes());books={pd.Timestamp(k):v for k,v in books.items() if pd.Timestamp(k)<pd.Timestamp('2023-01-01')}
    rows=[];bridges=[];last=None
    with threadpool_limits(limits=1):
        for date,members in sorted(books.items()):
            chosen,gate=select(date,252,members,P,Q,R,factor,S)
            result=dict(date=str(date.date()),gate=gate)
            if chosen is None:rows.append(result);last=None;continue
            x,w,sectors,_=chosen;cal=R.loc[:date].index[-252:]
            result['tails']=[tail(x,w,sectors,cal,h) for h in [1,5,10,20]]
            result['offsets']=[subspace(x,w.to_numpy(),d.market.reindex(x.index),resid,k) for resid in [False,True] for k in [1,3]]
            current=(date,x,w)
            if last is not None:
                b=bridge(last,current,R,factor);bridges.append(b);result['bridge']=b
            rows.append(result);last=current
            print(date.date(),'tail top5',round(result['tails'][1]['top5_share'],3),'bridge',result.get('bridge',{}).get('passed'),flush=True)
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    (OUT/'snapshots.json').write_text(json.dumps(rows,indent=2,default=convert))
    (OUT/'bridges.json').write_text(json.dumps(bridges,indent=2,default=convert))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MEMBERS,Path(__file__),ROOT/'research/BOOK_LEVEL_PROTOCOL.md']}
    (OUT/'manifest.json').write_text(json.dumps(dict(hashes=hashes,source_files=audit['fingerprints'],checks='Tail additivity, omission identity, PC subspace rotation invariance, symmetric variance bridge and per-name weight effects passed'),indent=2))
    print('PASS',sum(r['gate']['passed'] for r in rows),'snapshots;',sum(b['passed'] for b in bridges),'bridges')


if __name__=='__main__':main()
