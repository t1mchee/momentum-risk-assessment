"""Exploratory trailing-distribution anatomy; no forecasts, text or trading rule.

Run with the unstructured_momentum Python runtime (pandas/pyarrow/matplotlib).
Reads existing caches only; does not import or execute the other project's pipeline.
"""
from pathlib import Path
from zipfile import ZipFile
from io import StringIO
import json, re, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research' / 'distribution_anatomy'
SRC = Path('SOURCE_DATA_ROOT/data/raw/ishares/IWV/panel')
VERIFIED_ACTIONS = [
    ('2020-04-15','CHK',200.,'https://www.sec.gov/Archives/edgar/data/895126/000089512621000078/chk-20201231.htm'),
    ('2020-04-23','NBR',50.,'https://investor.nabors.com/2020-04-20-Nabors-Announces-Results-of-Special-Meeting-of-Shareholders-and-1-for-50-Reverse-Stock-Split-of-its-Common-Shares'),
    ('2022-05-09','TXMD',50.,'https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-89'),
]

def french(name):
    with ZipFile(ROOT/'data/raw'/name) as z:
        lines=z.read(z.namelist()[0]).decode('utf-8-sig').splitlines()
    a=next(i for i,s in enumerate(lines) if re.match(r'^\s*\d{8},',s))
    b=a
    while b<len(lines) and re.match(r'^\s*\d{8},',lines[b]): b+=1
    f=pd.read_csv(StringIO('\n'.join(lines[a-1:b])),index_col=0)
    f.columns=f.columns.str.strip();f.index=pd.to_datetime(f.index.astype(str),format='%Y%m%d')
    assert f.index.is_unique and f.index.is_monotonic_increasing
    return f.replace([-99.99,-999],np.nan)

def stats(s):
    s=s.dropna(); q=s.quantile([.05,.15,.5,.85,.95]);sd=s.std()
    return dict(n=len(s),mean=s.mean(),sd=sd,q05=q.loc[.05],q15=q.loc[.15],
                median=q.loc[.5],q85=q.loc[.85],q95=q.loc[.95],
                lower15_mean=s[s<=q.loc[.15]].mean(),
                downside_share=(q.loc[.5]-q.loc[.15])/(q.loc[.85]-q.loc[.15]),
                standardized_q15=(q.loc[.15]-s.mean())/sd,
                standardized_lower15_mean=(s[s<=q.loc[.15]].mean()-s.mean())/sd)

def distribution(s,date,h,w=252):
    return s.loc[:date].rolling(h,min_periods=h).sum().iloc[-w:]

def pair_scan(s):
    dates=s.loc['2014':'2022'].resample('ME').last().index
    rows=[]
    for h in [1,5,10,20]:
        ss=[stats(distribution(s,d,h)) for d in dates]
        pairs=[]
        for i,a in enumerate(ss):
            for j in range(i+24,len(ss)):
                b=ss[j];pool=np.sqrt((a['sd']**2+b['sd']**2)/2)
                if abs(a['mean']-b['mean'])>.05*pool or abs(a['sd']/b['sd']-1)>.05:continue
                pairs.append(dict(h=h,a=str(dates[i].date()),b=str(dates[j].date()),
                                  contrast=abs(a['downside_share']-b['downside_share']),A=a,B=b))
        pairs.sort(key=lambda x:-x['contrast'])
        rows.extend(pairs[:3])
    return rows

def robustness(s,pair):
    h=pair['h'];out=[]
    for w in [126,252,504]:
        a=stats(distribution(s,pair['a'],h,w));b=stats(distribution(s,pair['b'],h,w))
        out.append(dict(test=f'window_{w}',a=a['downside_share'],b=b['downside_share']))
    # Each phase is a disjoint sample; do not pool them as independent evidence.
    for phase in range(h):
        aa=distribution(s,pair['a'],h).iloc[phase::h]
        bb=distribution(s,pair['b'],h).iloc[phase::h]
        a=stats(aa);b=stats(bb)
        out.append(dict(test=f'nonoverlap_phase_{phase}',a=a['downside_share'],b=b['downside_share'],n=len(aa)))
    for d in [pair['a'],pair['b']]:
        x=s.loc[:d].copy();start=x.index[-(252+h-1)]; ix=x.loc[start:].abs().idxmax()
        base=stats(distribution(x,d,h));x.loc[ix]=0
        out.append(dict(test='one_day_zeroed_DIAGNOSTIC_NOT_CORRECTION',date=d,shock_date=str(ix.date()),
                        shock_pp=s.loc[ix],original=base['downside_share'],perturbed=stats(distribution(x,d,h))['downside_share']))
    return out

def load_companies(calendar):
    # Stop at 2022: respect the source project's sealed research period.
    fs=sorted(f for f in SRC.glob('IWV_*.parquet') if 2013<=int(f.stem[-4:])<=2022)
    parts=[pd.read_parquet(f) for f in fs]
    d=pd.concat(parts,ignore_index=True)
    original_rows=len(d)
    ambiguous=d.duplicated(['as_of','ticker'],keep=False)
    ambiguous_rows=int(ambiguous.sum())
    # Quarantine all ambiguous date/ticker rows rather than silently select last.
    # This is date-local, not a future-informed blacklist of entire companies.
    d=d.loc[~ambiguous & d.ticker.notna() & d.ticker.ne('-')].copy()
    assert not d.duplicated(['as_of','ticker']).any()
    P=d.pivot(index='as_of',columns='ticker',values='price').reindex(calendar)
    Q=d.pivot(index='as_of',columns='ticker',values='quantity').reindex(calendar)
    W=d.pivot(index='as_of',columns='ticker',values='weight_pct').reindex(calendar)
    S=d.pivot(index='as_of',columns='ticker',values='sector').reindex(calendar)
    P=P.where(P>0);Q=Q.where(Q>0)
    pr=P/P.shift();qr=Q/Q.shift()
    # Conservative sensitivity, NOT a verified corporate-action database. Quantity
    # is ETF holdings, so fund trades/flows can mimic or conceal a split.
    candidate=((pr>1.15)|(pr<1/1.15)) & ((pr*qr-1).abs()<=.05)
    ratios=np.array([.01,.02,.025,.04,.05,.1,.125,.2,.25,1/3,.5,2/3,.75,.8,
                     1.25,4/3,1.5,2,3,4,5,8,10,20,25,40,50,100])
    fac=np.ones(pr.shape); rr=pr.to_numpy(); cand=candidate.to_numpy()
    x=rr[cand];nearest=ratios[np.argmin(abs(np.log(x[:,None]/ratios)),axis=1)]
    accepted=(abs(x/nearest-1)<=.05)
    vals=np.ones(len(x));vals[accepted]=nearest[accepted];fac[cand]=vals
    factor=pd.DataFrame(fac,index=P.index,columns=P.columns)
    action_audit=[]
    for date,name,ratio,url in VERIFIED_ACTIONS:
        action_audit.append(dict(date=date,ticker=name,price_ratio=float(pr.loc[date,name]),
                                 quantity_ratio=float(qr.loc[date,name]),previous_factor=float(factor.loc[date,name]),
                                 verified_factor=ratio,source=url))
        factor.loc[date,name]=ratio
    R=(pr/factor-1)*100;raw=(pr-1)*100
    # Chain only through available consecutive prices. No forward filling of returns.
    # Split factors used for ranking are restricted to each formation interval below.
    audit=dict(files=len(fs),rows=len(d),original_rows=original_rows,ambiguous_rows_quarantined=ambiguous_rows,tickers=P.shape[1],first=str(d.as_of.min().date()),last=str(d.as_of.max().date()),
               candidate_actions=int(candidate.sum().sum()),accepted_actions=int((factor!=1).sum().sum()),
               returns_over_45pct_retained=int((R.abs()>.45*100).sum().sum()),
               missing_calendar_dates=int(P.isna().all(axis=1).sum()),
               verified_action_overrides=action_audit,
               fingerprints={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in fs})
    return P,W,S,R,raw,factor,audit

def build_book(P,W,S,R,raw,factor):
    # Monthly historical membership, formation ends one month before rebalance.
    # Use last snapshot strictly before the first holding date: one-session lag.
    blocks=[];rawblocks=[];meta=[];secblocks=[];namesblocks=[]
    calendar=P.index
    for month in pd.period_range('2014-01','2022-12',freq='M'):
        hold=calendar[calendar.to_period('M')==month]
        if not len(hold):continue
        previous=calendar[calendar<hold[0]]
        if not len(previous):continue
        asof=previous[-1]
        if P.loc[asof].notna().sum()<500:continue
        end=calendar[calendar<=asof-pd.DateOffset(months=1)][-1]
        starts=calendar[calendar<=asof-pd.DateOffset(months=12)]
        if not len(starts):continue
        start=starts[-1]
        formation=P.loc[start:end]
        adjustment=factor.loc[(calendar>start)&(calendar<=end)].prod()
        mom=P.loc[end]/P.loc[start]/adjustment-1
        eligible=P.loc[asof].notna() & (formation.notna().mean()>=.95) & mom.notna()
        mom=mom[eligible];n=len(mom)
        if n<500:continue
        k=n//10;win=mom.nlargest(k).index;los=mom.nsmallest(k).index
        for weighting in ['equal','fund_weight']:
            w=pd.Series(0.,index=P.columns)
            for names,sign in [(win,1),(los,-1)]:
                weights=pd.Series(1.,index=names) if weighting=='equal' else W.loc[asof,names].clip(lower=0)
                w.loc[names]=sign*weights/weights.sum()
            # Fixed daily-reset weights. Missing contributions remain NaN, never
            # silently become zero. Require >=95% observed weight PER LEG and
            # explicitly renormalize to the observed subset on usable dates.
            contrib=pd.DataFrame(0.,index=hold,columns=P.columns)
            cr=contrib.copy();coverage=pd.Series(True,index=hold)
            for names,sign in [(win,1),(los,-1)]:
                weights=w.loc[names].abs();obs=R.loc[hold,names].notna().mul(weights).sum(axis=1)
                obsraw=raw.loc[hold,names].notna().mul(weights).sum(axis=1)
                coverage &= obs.ge(.95)&obsraw.ge(.95)
                contrib.loc[:,names]=R.loc[hold,names].mul(w.loc[names]).div(obs,axis=0)
                cr.loc[:,names]=raw.loc[hold,names].mul(w.loc[names]).div(obsraw,axis=0)
            # Missing names have zero weight in the explicitly renormalized subset.
            contrib=contrib.fillna(0.);cr=cr.fillna(0.)
            contrib.loc[~coverage]=np.nan;cr.loc[~coverage]=np.nan
            sectors=S.loc[asof].fillna('Unknown')
            grouped=contrib.T.groupby(sectors).sum(min_count=1).T
            total=contrib.sum(axis=1,min_count=1)
            long=contrib.loc[:,win].sum(axis=1,min_count=1);short=contrib.loc[:,los].sum(axis=1,min_count=1)
            np.testing.assert_allclose((long+short).dropna(),total.dropna(),atol=1e-10)
            key=(str(month),weighting)
            blocks.append(pd.DataFrame(dict(net=total,long=long,short=short,weighting=weighting)))
            rawblocks.append(pd.DataFrame(dict(raw=cr.sum(axis=1,min_count=1),weighting=weighting)))
            namesblocks.append((weighting,contrib))
            secblocks.append((weighting,grouped))
            meta.append(dict(month=str(month),weighting=weighting,n_members=int(P.loc[asof].notna().sum()),
                             eligible=n,n_leg=k,usable_days=int(coverage.sum()),days=len(hold)))
    books={}
    for wt in ['equal','fund_weight']:
        b=pd.concat([x.drop(columns='weighting') for x in blocks if x.weighting.iloc[0]==wt]).sort_index()
        b['raw']=pd.concat([x.raw for x in rawblocks if x.weighting.iloc[0]==wt]).sort_index()
        names=pd.concat([x for w,x in namesblocks if w==wt]).sort_index()
        secs=pd.concat([x for w,x in secblocks if w==wt]).sort_index().fillna(0.)
        secs.loc[b.net.isna()]=np.nan
        books[wt]=(b,names,secs)
    return books,meta

def anatomy(book,date,h):
    b,names,secs=book
    idx=b.loc[:date].index[-252:]
    # Reindex to full market calendar already done outside, so gaps invalidate windows.
    r=b.rolling(h,min_periods=h).sum().reindex(idx)
    valid=r.net.dropna()
    if len(valid)<180:return dict(status='insufficient valid trailing windows',valid_windows=len(valid))
    tail=valid.index[valid<=valid.quantile(.15)]
    nc=names.rolling(h,min_periods=h).sum().reindex(tail).mean()
    sc=secs.rolling(h,min_periods=h).sum().reindex(tail).mean()
    neg=-nc[nc<0].sort_values();totalneg=neg.sum()
    # Top-name shares refer to gross negative average contributions, not net loss.
    active=(names.reindex(idx).abs()>0).any()
    return dict(status='exploratory price-return proxy',valid_windows=len(valid),tail_windows=len(tail),
                distribution=stats(valid),tail_net=r.loc[tail,'net'].mean(),
                tail_long=r.loc[tail,'long'].mean(),tail_short=r.loc[tail,'short'].mean(),
                top5_share_gross_negative=neg.head(5).sum()/totalneg,
                top10_share_gross_negative=neg.head(10).sum()/totalneg,
                negative_names=int((nc<0).sum()),active_names=int(active.sum()),
                top_negative_names=nc.nsmallest(8).to_dict(),sector_contributions=sc.sort_values().to_dict(),
                name_additivity_error=float(nc.sum()-r.loc[tail,'net'].mean()),
                raw_distribution=stats(r.raw.dropna()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s=french('momentum.zip').Mom;legs=french('portfolios.zip')
    pairs=pair_scan(s);result=dict(pair_selection='post-hoc maximum shape contrast among matched mean/SD; no holdout or inference',pairs=pairs)
    previous=OUT/'results.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if 'verified_action_overrides' not in old['data_audit']:
            result['before_three_verified_actions']={k:old[k] for k in ['reconciliation','cases']}
        elif 'before_three_verified_actions' in old:
            result['before_three_verified_actions']=old['before_three_verified_actions']
    # Deterministic invariance and bookkeeping checks, not inferential validation.
    x=distribution(s,'2020-11-30',20)
    np.testing.assert_allclose(stats(x)['downside_share'],stats(x*2+7)['downside_share'])
    modified=s.copy();modified.loc[modified.index>'2020-11-30']=999.
    pd.testing.assert_series_equal(x,distribution(modified,'2020-11-30',20))
    result['tests']=['positive affine shape invariance','future-return perturbation invariance','daily long+short additivity','ambiguous ticker-date quarantine','no return forward-fill']
    for p in pairs:p['robustness']=robustness(s,p)
    print('MATCHES',json.dumps(pairs,default=str),flush=True)
    calendar=s.loc['2013':'2022'].index
    P,W,S,R,raw,factor,audit=load_companies(calendar)
    print('DATA AUDIT',audit,flush=True)
    books,meta=build_book(P,W,S,R,raw,factor)
    result['data_audit']=audit;result['book_coverage']=meta
    result['reconciliation']={}
    for wt,(b,names,secs) in books.items():
        b=b.reindex(calendar);names=names.reindex(calendar);secs=secs.reindex(calendar)
        books[wt]=(b,names,secs)
        j=b.join(s.rename('french')).dropna(subset=['net','french'])
        result['reconciliation'][wt]=dict(n=len(j),corr=j.net.corr(j.french),raw_corr=j.raw.corr(j.french),
            sd_ratio=j.net.std()/j.french.std(),worst_days=j.nsmallest(5,'net')[['net','french']].reset_index().astype(str).to_dict('records'))
    chosen=[next(p for p in pairs if p['h']==h) for h in [5,20]]
    cases=sorted(set([(p[d],p['h']) for p in chosen for d in ['a','b']]+[('2020-11-30',5),('2020-11-30',20)]))
    result['cases']=[]
    for date,h in cases:
        row=dict(date=date,h=h,french=stats(distribution(s,date,h)),company={})
        fl=(legs['SMALL HiPRIOR']+legs['BIG HiPRIOR'])/2
        fs=-(legs['SMALL LoPRIOR']+legs['BIG LoPRIOR'])/2
        t=distribution(s,date,h);ix=t[t<=t.quantile(.15)].index
        row['french_joint_tail']=dict(net=t.loc[ix].mean(),long=distribution(fl,date,h).loc[ix].mean(),short=distribution(fs,date,h).loc[ix].mean())
        for wt,b in books.items():
            row['company'][wt]=anatomy(b,date,h)
            available=b[0].net.rolling(h,min_periods=h).sum().reindex(t.index).dropna().index
            row['company'][wt]['french_same_available_windows']=stats(t.reindex(available))
        result['cases'].append(row)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for row,p in enumerate(chosen):
        for d,color in [(p['a'],'#176b87'),(p['b'],'#b84a3a')]:
            x=distribution(s,d,p['h']).dropna();z=(x-x.mean())/x.std()
            for col,v in enumerate([x,z]):
                a=axes[row,col];v=np.sort(v);a.plot(v,np.arange(1,len(v)+1)/len(v),label=d,color=color)
                a.axhline(.15,color='grey',ls=':',lw=.7);a.legend();a.grid(alpha=.2)
                a.set_title(f"{p['h']}-day trailing sums: {'raw' if col==0 else 'mean/SD removed'}")
                a.set_xlabel('Percentage-point fixed-notional P&L' if col==0 else 'Within-window standard deviations')
                a.set_ylabel('Empirical cumulative probability')
    fig.suptitle('Similar means and volatility, different trailing distributions\nExploratory selected examples; overlapping windows are not independent',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'distribution_comparison.png',dpi=160);plt.close(fig)
    def convert(x):
        if isinstance(x,(np.integer,np.floating)):return x.item()
        if isinstance(x,pd.Timestamp):return str(x.date())
        raise TypeError(type(x))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=convert))
    print('RESULTS',json.dumps({k:result[k] for k in ['reconciliation','cases']},default=convert),flush=True)

if __name__=='__main__':main()
