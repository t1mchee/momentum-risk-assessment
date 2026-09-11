"""Panel follow-through: trailing mechanisms, episode influence and ordering.

No forecasts or trading rules. Fixed historical grid; resampling uses fixed seeds.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from distribution_anatomy import french

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/distribution_panel'
SEED=20260912
CATS=['winner_loss','loser_rally','both_lose','neither_loses']

def sums(x,h):
    c=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)])
    return c[h:]-c[:-h]

def tail_weights(y,p=.15):
    """Exactly p*n observations of mass; equally fractionally weight cutoff ties."""
    mass=len(y)*p;k=int(np.ceil(mass))-1;cut=np.partition(y,k)[k]
    w=(y<cut).astype(float);tie=y==cut
    w[tie]=(mass-w.sum())/tie.sum()
    np.testing.assert_allclose(w.sum(),mass,atol=1e-9)
    return w

def categories(a):
    return np.where((a[:,0]<0)&(a[:,1]<0),2,np.where(a[:,0]<0,0,np.where(a[:,1]<0,1,3)))

def metrics(a):
    y=a.sum(axis=1);w=tail_weights(y);z=w/w.sum()
    q=np.quantile(y,[.15,.5,.85]);sd=y.std(ddof=1)
    return np.array([z@y,(z@y-y.mean())/sd,(q[1]-q[0])/(q[2]-q[0]),q[0]])

def profile(a,h,episodes=True):
    y=a.sum(axis=1);w=tail_weights(y);z=w/w.sum();cat=categories(a)
    leg=z@a;v=metrics(a)
    row=dict(tail_mean=v[0],standardized_tail=v[1],downside_share=v[2],q15=v[3],
             long=leg[0],short=leg[1],n=len(y),mean=y.mean(),sd=y.std(ddof=1),
             marginal_long_q15=np.quantile(a[:,0],.15),marginal_short_q15=np.quantile(a[:,1],.15),
             covariance= np.cov(a.T)[0,1])
    for k,name in enumerate(CATS):
        mask=cat==k;mass=w[mask].sum()
        row[name+'_share']=z[mask].sum()
        row[name+'_contribution']=z[mask]@y[mask]
        row[name+'_severity']=float(w[mask]@y[mask]/mass) if mass else None
    np.testing.assert_allclose(sum(row[c+'_contribution'] for c in CATS),row['tail_mean'],atol=1e-9)
    if not episodes:return row
    indices=np.flatnonzero(w>0);groups=[]
    for i in indices:
        if not groups or i-h+1>groups[-1][-1]:groups.append([int(i)])
        else:groups[-1].append(int(i))
    loss=w*np.maximum(-y,0);amounts=np.array([loss[g].sum() for g in groups])
    big=int(amounts.argmax());remaining=w.copy();remaining[groups[big]]=0
    row.update(episodes=len(groups),largest_episode_gross_loss_share=amounts[big]/amounts.sum() if amounts.sum()>0 else None,
               largest_episode_windows=len(groups[big]),largest_episode_first_window=groups[big][0],
               largest_episode_last_window=groups[big][-1])
    if remaining.sum()>0:
        rl=remaining@a/remaining.sum()
        row.update(loo_long=rl[0],loo_short=rl[1],loo_tail=rl.sum(),
                   dominant_leg_survives=bool(np.argmin(rl)==np.argmin(leg)))
    else:row.update(loo_long=None,loo_short=None,loo_tail=None,dominant_leg_survives=None)
    return row

def block_resample(x,block,rng):
    n=len(x);starts=rng.integers(0,n,size=int(np.ceil(n/block)))
    ix=((starts[:,None]+np.arange(block))%n).ravel()[:n]
    return x[ix]

def reorder(x,block,rng,random_phase=False):
    if random_phase:
        x=np.roll(x,int(rng.integers(0,len(x))),axis=0)
    chunks=[x[i:i+block] for i in range(0,len(x),block)]
    return np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=french('portfolios.zip');published=french('momentum.zip').Mom
    books={}
    for size,label in [('SMALL','small'),('BIG','big')]:
        books[label]=pd.DataFrame({'long':p[size+' HiPRIOR'],'short':-p[size+' LoPRIOR']})
    books['broad']=(books['small']+books['big'])/2
    diff=(books['broad'].sum(axis=1)-published).abs().max()
    assert diff<.02
    # Synthetic invariants: exact fractional/tied tail mass and additive mechanisms.
    np.testing.assert_allclose(tail_weights(np.ones(7)),np.full(7,.15))
    test=np.array([[-1.,2.],[2.,-3.],[-3.,-2.],[2.,3.],[-1.,1.]])
    assert categories(test).tolist()==[0,1,2,3,0]
    rng=np.random.default_rng(SEED)
    dates=pd.date_range('1975-01-31','2022-12-31',freq='ME')
    grid=[]
    for name,frame in books.items():
        for h in [1,5,10,20]:
            rolled=frame.rolling(h,min_periods=h).sum()
            for window in [126,252,504]:
                for date in dates:
                    a=rolled.loc[:date].tail(window).to_numpy()
                    if len(a)!=window or not np.isfinite(a).all():continue
                    r=profile(a,h);r.update(book=name,h=h,window=window,date=str(date.date()))
                    grid.append(r)
            print('grid',name,h,flush=True)
    g=pd.DataFrame(grid)
    summary=[]
    for (name,h),f in g[g.window==252].groupby(['book','h']):
        summary.append(dict(book=name,h=int(h),snapshots=len(f),median_episodes=f.episodes.median(),
            median_largest_episode_share=f.largest_episode_gross_loss_share.median(),
            share_largest_episode_over_half=(f.largest_episode_gross_loss_share>.5).mean(),
            dominant_leg_survival_among_nonempty=f.dominant_leg_survives.dropna().mean(),
            empty_after_removal=int(f.dominant_leg_survives.isna().sum()),
            median_winner_loss_share=f.winner_loss_share.median(),median_loser_rally_share=f.loser_rally_share.median(),
            median_both_lose_share=f.both_lose_share.median()))
    # Entire sample is exploratory, including unselected grid; no fresh holdout.
    checks={}
    broad=g[g.book=='broad']
    for h in [5,20]:
        f=broad[broad.h==h].copy();f['dominant']=np.where(f.long<f.short,'long','short')
        pv=f.pivot(index='date',columns='window',values='dominant')
        checks[str(h)]={'dominant_leg_agrees_all_lookbacks':float((pv.nunique(axis=1)==1).mean())}
        a=g[(g.h==h)&(g.window==252)].copy();a['dominant']=a.long<a.short
        z=a.pivot(index='date',columns='book',values='dominant')
        checks[str(h)]['small_big_dominant_leg_agreement']=float((z.small==z.big).mean())
    fixed=[('2015-04-30','2018-11-30',5),('2019-08-31','2022-08-31',20)]
    uncertainty=[]
    for da,db,h in fixed:
        aa=books['broad'].loc[:da].tail(252+h-1).to_numpy();bb=books['broad'].loc[:db].tail(252+h-1).to_numpy()
        point=metrics(sums(bb,h))-metrics(sums(aa,h))
        for block in [40,60,120]:
            draws=[]
            for _ in range(600):
                draws.append(metrics(sums(block_resample(bb,block,rng),h))-metrics(sums(block_resample(aa,block,rng),h)))
            draws=np.asarray(draws)
            # Bonferroni percentile bootstrap intervals over this four-metric family.
            # Approximate, conditional on selected pairs, not selection-adjusted inference.
            lo,hi=np.quantile(draws,[.05/8,1-.05/8],axis=0)
            uncertainty.append(dict(a=da,b=db,h=h,block=block,draws=600,
                metrics=['tail_mean','standardized_tail','downside_share','q15'],difference=point.tolist(),
                simultaneous_approx95_lo=lo.tolist(),simultaneous_approx95_hi=hi.tolist()))
        print('uncertainty',h,flush=True)
    temporal=[]
    for date in ['2009-04-30','2015-04-30','2018-11-30','2019-08-31','2020-11-30','2022-08-31']:
        for h in [5,20]:
            x=books['broad'].loc[:date].tail(252+h-1).to_numpy();observed=metrics(sums(x,h))
            for block in [1,5,20,60]:
                for phase in ['fixed','random_circular_cut']:
                    draws=np.array([metrics(sums(reorder(x,block,rng,phase!='fixed'),h)) for _ in range(399)])
                    temporal.append(dict(date=date,h=h,block=block,phase=phase,draws=399,observed=observed.tolist(),
                        reordered_median=np.median(draws,axis=0).tolist(),
                        reordered_05=np.quantile(draws,.05,axis=0).tolist(),reordered_95=np.quantile(draws,.95,axis=0).tolist(),
                        observed_percentile=np.mean(draws<=observed,axis=0).tolist()))
        print('temporal',date,flush=True)
    selected=g[(g.window==252)&g.date.isin([a for a,b,h in fixed]+[b for a,b,h in fixed]+['2009-04-30','2020-11-30'])]
    result=dict(protocol=dict(sample='monthly 1975–2022; no held-out inference',seed=SEED,
        tail='exact 15% empirical mass; tied cutoff equally fractionally weighted',
        return_units='daily-reset fixed-notional percentage points; net constructed additively from French legs',
        episodes='connected overlapping selected-window supports; calendar session indices',
        loo='remove largest episode selected windows from fixed selected tail, renormalize remaining mass; do not recompute cutoff',
        temporal='paired days/block permutations without replacement; descriptive, not causal or predictive',
        uncertainty='circular paired-day block bootstrap; approximate simultaneous four-metric intervals; postselection unadjusted',
        caveats='monthly snapshots overlap; proportions are descriptive calendar prevalence, not independent trials'),
        rounding_max_pp=float(diff),summary=summary,stability=checks,uncertainty=uncertainty,temporal=temporal,
        selected=selected.replace({np.nan:None}).to_dict('records'),
        tests=['exact 15% tail mass including ties','sign-category mapping','mechanism contribution additivity for every profile','published-factor rounding check'])
    g.to_parquet(OUT/'monthly_profiles.parquet',index=False)
    def conv(x):
        if isinstance(x,(np.integer,np.floating)):return x.item()
        if isinstance(x,np.bool_):return bool(x)
        raise TypeError(type(x))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=conv))
    print(json.dumps({'summary':summary,'stability':checks},default=conv),flush=True)

if __name__=='__main__':main()
