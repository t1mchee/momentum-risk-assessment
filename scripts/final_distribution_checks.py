"""Frozen pre-notebook checks: window accounting, interim paths, chronological replay."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from distribution_anatomy import french
from distribution_panel import tail_weights,profile,CATS
from risk_core import MomentumRiskCore

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/final_distribution_checks'
METRICS=['tail_mean','long','short']

def amounts(a):
    z=tail_weights(a.sum(axis=1));z=z/z.sum();leg=z@a
    return np.array([leg.sum(),leg[0],leg[1]])

def change(old,new):
    common=old.index.intersection(new.index);enter=new.index.difference(old.index)
    joined=pd.concat([old,new.loc[enter]]).sort_index()
    a,b,c,u=[amounts(x.to_numpy()) for x in [old,new,old.loc[common],joined]]
    incoming=.5*((u-a)+(b-c));outgoing=.5*((c-a)+(b-u))
    np.testing.assert_allclose(incoming+outgoing,b-a,atol=1e-10)
    return dict(old=a.tolist(),new=b.tolist(),delta=(b-a).tolist(),entering=incoming.tolist(),
                leaving=outgoing.tolist(),n_entering=len(enter),n_leaving=len(old)-len(common))

def path_windows(daily,h):
    arr=np.lib.stride_tricks.sliding_window_view(daily.to_numpy(),h)
    paths=arr.cumsum(axis=1);low=np.minimum(0,paths.min(axis=1));end=paths[:,-1]
    assert np.all(low<=end+1e-10)
    return pd.DataFrame(dict(endpoint=end,low=low,recovery=end-low),index=daily.index[h-1:])

def path_summary(f):
    w=tail_weights(f.endpoint.to_numpy());z=w/w.sum()
    v=tail_weights(f.low.to_numpy());v=v/v.sum()
    crossed=f.low<-2.16;recovered=crossed&f.endpoint.ge(-2.16)
    return dict(n=len(f),crossed_fixed_loss=int(crossed.sum()),recovered_above_threshold=int(recovered.sum()),
                recovered_fraction_of_crossings=float(recovered.sum()/crossed.sum()) if crossed.any() else None,
                endpoint_tail_mean=float(z@f.endpoint),low_on_endpoint_tail=float(z@f.low),
                recovery_on_endpoint_tail=float(z@f.recovery),
                path_tail_low=float(v@f.low),endpoint_on_path_tail=float(v@f.endpoint),
                recovery_on_path_tail=float(v@f.recovery))

def weekly(mom):
    scale=np.sqrt(mom.pow(2).ewm(alpha=.06,adjust=False).mean())
    groups=mom.resample('W-FRI');dates=pd.Series(mom.index,index=mom.index)
    f=pd.DataFrame(dict(y=groups.sum(min_count=1),start=dates.resample('W-FRI').min(),
                        end=dates.resample('W-FRI').max()))
    f['cutoff']=f.end.shift();f['ewma']=scale.reindex(pd.DatetimeIndex(f.cutoff)).to_numpy()
    return f.dropna()

def observation(legs,mom,date):
    actual=legs.loc[:date].index[-1];profiles={};warnings={}
    for h in [5,20]:
        hist=legs.loc[:actual].rolling(h).sum()
        p=profile(hist.tail(252).to_numpy(),h)
        other=[profile(hist.tail(w).to_numpy(),h,False) for w in [126,504]]
        dominant=int(p['long']>p['short'])
        sensitive=any(int(x['long']>x['short'])!=dominant for x in other)
        flags=[]
        if sensitive:flags.append('Dominant leg changes with lookback')
        if p['largest_episode_gross_loss_share'] is not None and p['largest_episode_gross_loss_share']>.5:
            flags.append('One connected episode accounts for more than half the adverse losses')
        if p['dominant_leg_survives'] is not True:flags.append('Dominant leg is not robust to largest-episode removal')
        profiles[str(h)]=p;warnings[str(h)]=flags
    # Forward engine remains separate. Fit annually, previous Dec 31; only COMPLETE
    # Friday-ending calendar weeks at that cutoff, never a partially observed week.
    fitcut=pd.Timestamp(actual.year-1,12,31)
    wk=weekly(mom.loc[:fitcut]);wk=wk.loc[wk.index<=fitcut]
    engine=MomentumRiskCore().fit(wk,fitcut,daily_returns=mom)
    forward=engine.forecast_from_returns(mom,actual)
    assert engine.training_end<=fitcut<actual
    return dict(requested_date=str(pd.Timestamp(date).date()),last_observation=str(actual.date()),
                historical_trailing_description=profiles,description_warnings=warnings,
                description_status='Retrospective empirical descriptions, not forward probabilities or stable regime estimates. No diagnostic flags does not imply safety or statistical confidence.',
                separate_forward_research_estimate=forward,
                forward_warning='Existing five-session scale/empirical-shock model; exploratory, not recalibrated here; bear-state fixed-loss probabilities underpredict in prior evaluation. Trailing statistics are not forecasts.',
                source_warning='Latest cached historical vintage, not contemporaneously archived releases; verifies cutoff arithmetic, not original real-time data availability.')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=french('portfolios.zip');mom=french('momentum.zip').Mom.loc['1964':]
    legs=pd.DataFrame({'long':(p['SMALL HiPRIOR']+p['BIG HiPRIOR'])/2,
                       'short':-(p['SMALL LoPRIOR']+p['BIG LoPRIOR'])/2}).loc['1964':]
    dates=pd.date_range('1975-01-31','2022-12-31',freq='ME')
    changes=[];paths=[]
    for h in [5,20]:
        rolling=legs.rolling(h).sum()
        for prior,now in zip(dates[:-1],dates[1:]):
            old=rolling.loc[:prior].tail(252);new=rolling.loc[:now].tail(252)
            r=change(old,new);r.update(date=str(now.date()),prior=str(prior.date()),h=h)
            changes.append(r)
        path=path_windows(legs.sum(axis=1),h)
        full=path_summary(path.loc['1975':'2022']);full.update(h=h,sample='1975–2022 all ending windows')
        paths.append(full)
    snapshots=[]
    # Mechanical quarterly schedule fixed before viewing outputs; no hand-picked cases.
    for date in pd.date_range('1990-03-31','2022-12-31',freq='QE'):
        r=observation(legs,mom,date)
        truncated=observation(legs.loc[:date],mom.loc[:date],date)
        assert json.dumps(r,sort_keys=True,default=float)==json.dumps(truncated,sort_keys=True,default=float)
        for h in [5,20]:
            r.setdefault('historical_path_description',{})[str(h)]=path_summary(path_windows(legs.loc[:date].sum(axis=1),h).tail(252))
        snapshots.append(r)
    summaries=[]
    for h in [5,20]:
        c=[x for x in changes if x['h']==h]
        # Avoid unstable entry/exit shares when changes cancel. 'Dominates' uses
        # absolute contribution magnitudes; examples remain historical descriptions.
        counts=sum(abs(x['leaving'][0])>abs(x['entering'][0]) for x in c)
        summary=dict(h=h,n_changes=len(c),leaving_dominates_absolute_change_fraction=counts/len(c))
        worsening=[x for x in c if x['delta'][0]<-1e-10]
        summary['worsenings']=len(worsening)
        summary['worsenings_with_incoming_improving']=sum(x['entering'][0]>1e-8 for x in worsening)
        summary['materiality_sensitivity']=[]
        for threshold in [.05,.1]:
            meaningful=[x for x in c if x['delta'][0]<-threshold]
            summary['materiality_sensitivity'].append(dict(minimum_worsening_pp=threshold,
                worsenings=len(meaningful),incoming_improves=sum(x['entering'][0]>1e-8 for x in meaningful)))
        summary['examples_largest_changes']=sorted(c,key=lambda x:-abs(x['delta'][0]))[:4]
        summary['exit_driven_worsening_examples']=sorted([x for x in worsening if x['entering'][0]>0],key=lambda x:x['delta'][0])[:3]
        summaries.append(summary)
    # Synthetic path test: endpoint positive despite an adverse interim loss.
    t=path_windows(pd.Series([-3.,4.],index=pd.date_range('2000-01-01',periods=2)),2).iloc[0]
    assert t.endpoint==1 and t.low==-3 and t.recovery==4
    result=dict(protocol=dict(metrics=METRICS,dates='month-end 1975–2022; quarterly rehearsal 1990–2022',
        accounting='Two-order Shapley accounting: entering h-day windows and leaving h-day windows. Intermediate sample sizes vary; not causal attribution.',
        paths='Minimum cumulative fixed-notional P&L from window start including initial zero, NOT peak-to-trough drawdown. Completed trailing windows only.',
        threshold='-2.16 pp fixed illustrative loss level for both horizons; not a newly estimated target.',
        availability='Cutoff-safe calculations on latest cached vintage; actual historical publication/vintage availability unverified.',
        uncertainty='All samples reused/exploratory. Overlapping windows and snapshots are not independent trials.'),
        change_summary=summaries,changes=changes,path_summary=paths,rehearsal=snapshots,
        tests=dict(exact_entry_exit_reconciliation=len(changes),full_vs_truncated_replay=len(snapshots),
                   synthetic_path=True,forward_fits_precede_cutoffs=True),
        hashes={name:hashlib.sha256((ROOT/'data/raw'/name).read_bytes()).hexdigest() for name in ['momentum.zip','portfolios.zip']})
    def conv(x):
        if isinstance(x,(np.floating,np.integer)):return x.item()
        if isinstance(x,np.bool_):return bool(x)
        raise TypeError(type(x))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=conv))
    print(json.dumps({k:result[k] for k in ['change_summary','path_summary','tests']},default=conv),flush=True)

if __name__=='__main__':main()
