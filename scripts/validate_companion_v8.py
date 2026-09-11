"""Claim-specific audit of frozen models; no parameter selection or new holdout."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from prepare_companion_v8 import design

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/notebook_v8'
OUT = BASE / 'validation'

def paired(delta, mask, block, seed=20260913):
    """Resample chronological blocks BEFORE selecting a conditional subset."""
    delta, mask = np.asarray(delta), np.asarray(mask, bool)
    n = len(delta); rng = np.random.default_rng(seed); draws = []
    for _ in range(1000):
        ix = (rng.integers(0, n, int(np.ceil(n/block)))[:, None] + np.arange(block)).ravel()[:n] % n
        eligible = ix[mask[ix]]
        if len(eligible): draws.append(float(delta[eligible].mean()))
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(difference=float(delta[mask].mean()), lo=float(lo), hi=float(hi), block=block)

def losses(p, event):
    p = np.clip(np.asarray(p), 1e-8, 1-1e-8)
    return {'Brier': (p-event)**2, 'Log loss': -(event*np.log(p)+(1-event)*np.log1p(-p))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = pd.read_parquet(BASE/'evaluation.parquet').sort_values('end')
    assert len(e)==2035 and e.end.max().year==2022
    masks = {'all': np.ones(len(e), bool), 'bear': e.bear.to_numpy().astype(bool),
             'nonbear': ~e.bear.to_numpy().astype(bool)}
    for name, a, b in [('1984–1999',1984,1999), ('2000–2016',2000,2016), ('2017–2022',2017,2022)]:
        masks[name] = e.end.dt.year.between(a,b).to_numpy()
    event = e.event.to_numpy(); retained = losses(e.p_fhs5y_ewma, event)
    rows = []
    for comparator, col in [('Frequency','p_frequency'), ('Volatility logit','p_logit_rv126'), ('Matched EWMA logit','p_logit_ewma')]:
        for metric, base in losses(e[col],event).items():
            for subset, mask in masks.items():
                for block in [13,26,52]:
                    rows.append(dict(comparator=comparator,metric=metric,subset=subset,n=int(mask.sum()),
                        events=int(event[mask].sum()),**paired(retained[metric]-base,mask,block)))
    def pinball(q):
        residual=e.y.to_numpy()-np.asarray(q)
        return np.maximum(.05*residual,-.95*residual)
    for subset, mask in masks.items():
        for block in [13,26,52]:
            rows.append(dict(comparator='Historical quantile', metric='q05 pinball',subset=subset,
                n=int(mask.sum()),events=int(event[mask].sum()),
                **paired(pinball(e.q_fhs5y_ewma)-pinball(e.q_historical),mask,block)))

    # Same annual previous-Dec31 specification as the displayed scenario curves.
    d=pd.read_parquet(BASE/'daily.parquet')
    w=d[['long','short']].resample('W-FRI').sum(min_count=1)
    w['market']=d.market.resample('W-FRI').apply(lambda s:np.expm1(np.log1p(s/100).sum())*100)
    w['end']=pd.Series(d.index,index=d.index).resample('W-FRI').max()
    w['cutoff']=w.end.shift()
    w['bear']=d.market_2y.reindex(pd.DatetimeIndex(w.cutoff)).to_numpy()<0
    w=w.loc['1974':].dropna(); forecasts=[]
    for year in range(1984,2023):
        cutoff=pd.Timestamp(year-1,12,31)
        tr=w.loc[(w.index<=cutoff)&(w.end<=cutoff)]
        te=w.loc[w.index.year==year]
        assert tr.end.max()<te.end.min()
        for name, cols in [('linear',[0,1]),('asymmetric',[0,1,2]),('state asymmetric',list(range(6)))]:
            x=design(tr.market,tr.bear.astype(float))[:,cols]
            z=design(te.market,te.bear.astype(float))[:,cols]
            coef=np.linalg.lstsq(x,tr[['long','short']],rcond=None)[0]
            prediction=(z@coef).sum(axis=1)
            residual=tr[['long','short']].sum(axis=1).to_numpy()-(x@coef).sum(axis=1)
            for i,(_,r) in enumerate(te.iterrows()):
                # Explicit diagnostic outcome band, distinct from mean-CI in plot.
                pool=residual[tr.bear.to_numpy()==r.bear] if name=='state asymmetric' else residual
                lo,hi=np.quantile(pool,[.05,.95]); actual=r.long+r.short
                forecasts.append(dict(end=str(r.end.date()),model=name,bear=bool(r.bear),market=r.market,
                    actual=actual,prediction=prediction[i],error=prediction[i]-actual,
                    covered=bool(prediction[i]+lo<=actual<=prediction[i]+hi)))
    f=pd.DataFrame(forecasts); scenario=[]; differences=[]
    assert len(f)==3*len(e)
    for subset in ['all','bear rebound >2%','2017–2022']:
        chosen=f if subset=='all' else f.loc[(f.bear & f.market.gt(2)) if subset.startswith('bear') else f.end.ge('2017')]
        for name,g in chosen.groupby('model'):
            scenario.append(dict(subset=subset,model=name,n=len(g),MAE=float(g.error.abs().mean()),
                RMSE=float(np.sqrt(g.error.pow(2).mean())),coverage90=float(g.covered.mean())))
        wide=f.pivot(index='end',columns='model',values='error').sort_index()
        metadata=f[f.model=='linear'].set_index('end').reindex(wide.index)
        mask=np.ones(len(wide),bool) if subset=='all' else ((metadata.bear & metadata.market.gt(2)).to_numpy() if subset.startswith('bear') else wide.index>='2017')
        for comparator in ['linear','asymmetric']:
            for block in [13,26,52]:
                differences.append(dict(subset=subset,comparator=comparator,metric='MAE',
                    **paired(wide['state asymmetric'].abs()-wide[comparator].abs(),mask,block)))
    f.to_parquet(OUT/'scenario_predictions.parquet')

    checks=json.loads((ROOT/'research/final_distribution_checks/results.json').read_text())
    ablation=[]
    for h in ['5','20']:
        r=checks['rehearsal']; profiles=[x['historical_trailing_description'][h] for x in r]
        ablation.append(dict(horizon=int(h),scheduled_snapshots=len(r),
            episode_over_half=sum(p['largest_episode_gross_loss_share'] is not None and p['largest_episode_gross_loss_share']>.5 for p in profiles),
            dominant_leg_not_surviving_omission=sum(p['dominant_leg_survives'] is not True for p in profiles),
            any_description_warning=sum(bool(x['description_warnings'][h]) for x in r)))
    sourcefiles=[BASE/'daily.parquet',BASE/'evaluation.parquet',ROOT/'research/final_distribution_checks/results.json',Path(__file__)]
    result=dict(protocol='Frozen, already-examined history; seed 20260913; 1000 circular chronological block draws; pointwise 95% intervals, not search-adjusted. Negative paired loss difference favours retained model. Periods are diagnostic, not untouched holdouts.',
        forward=rows,scenario=scenario,scenario_paired=differences,explanatory_audit=ablation,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles})
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    # Prepared human evaluation materials, not synthetic human evaluation results.
    cards={1:[],2:[]}; key=[]
    selected=[r for r in checks['rehearsal'] if int(r['requested_date'][:4]) in range(1990,2021,5) and r['requested_date'][5:]=='12-31']
    for i,r in enumerate(selected):
        p=r['historical_trailing_description']['20']; label=f'Case {i+1}'
        base=(f'## {label}\n\nPublic momentum factor; completed trailing 20-session outcomes, '
              f'252 ending windows. Mean {p["mean"]:.2f} pp; SD {p["sd"]:.2f} pp; '
              f'lower-15% average {p["tail_mean"]:.2f} pp. Historical description, not a forward forecast.\n\n')
        full=(f'Joint-tail winner-long contribution {p["long"]:+.2f} pp; short-loser contribution {p["short"]:+.2f} pp. '
              f'Connected episodes: {p["episodes"]}. Qualifications: '+('; '.join(r['description_warnings']['20']) or 'No diagnostic flags; this does not establish safety or confidence.')+'\n\n')
        questions=('1. Is the lower-tail average a forecast or a historical description?\n'
                   '2. Which leg contributed more adversely? State evidence, or “not identifiable from this card”.\n'
                   '3. Is a robust recurring mechanism established? Explain what is missing or qualifies the conclusion.\n'
                   '4. Does this establish the risk of your own portfolio? Why?\n'
                   '5. In one sentence, what would you investigate next?\n\n'
                   'Record time taken, confidence (1–5), and usefulness (1–5).\n\n')
        for group in [1,2]:
            enhanced=(i+group)%2==0
            cards[group].append(base+(full if enhanced else '')+questions)
        key.append(dict(case=label,date=r['requested_date'],group1_enhanced=(i+1)%2==0,
            adverse_leg='winner long' if p['long']<p['short'] else 'short loser',
            warnings=r['description_warnings']['20']))
    for group,contents in cards.items():
        (OUT/f'READER_GROUP_{group}.md').write_text('# Reader exercise — unadministered\n\n'+''.join(contents))
    (OUT/'READER_ANSWER_KEY.json').write_text(json.dumps(key,indent=2))
    print(pd.DataFrame(rows).query("block==26 and subset in ['all','bear']").to_string(index=False))
    print(pd.DataFrame(scenario).to_string(index=False))
    print(pd.DataFrame(differences).query('block==26').to_string(index=False))
    print(ablation)

if __name__=='__main__':main()
