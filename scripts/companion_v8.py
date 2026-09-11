"""Inspectable numerical and plotting helpers for notebook 08. No network access."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from distribution_panel import tail_weights,profile
from final_distribution_checks import path_windows,change,weekly
from risk_core import MomentumRiskCore

BLUE='#236b8e';ORANGE='#be653d';INK='#243342';MUTED='#6e7781'

class Companion:
    def __init__(self,root):
        self.root=Path(root);self.folder=self.root/'research/notebook_v8'
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        for name,value in self.manifest['hashes'].items():
            assert hashlib.sha256((self.root/name).read_bytes()).hexdigest()==value,f'Stale input: {name}'
        self.daily=pd.read_parquet(self.folder/'daily.parquet')
        self.evaluation=pd.read_parquet(self.folder/'evaluation.parquet')
        self.scenarios=json.loads((self.folder/'scenarios.json').read_text())
        self.audit=json.loads((self.folder/'validation/results.json').read_text())
        for name,value in self.audit['source_hashes'].items():
            assert hashlib.sha256((self.root/name).read_bytes()).hexdigest()==value,f'Stale validation: {name}'
        self.checks=json.loads((self.root/'research/final_distribution_checks/results.json').read_text())
        self.panel=json.loads((self.root/'research/distribution_panel/results.json').read_text())
        self.atlas=pd.read_parquet(self.root/'research/distribution_panel/monthly_profiles.parquet')
        self.snapshots={r['requested_date']:r for r in self.checks['rehearsal']}
        self.dates=list(self.snapshots);self.models={}
        self.legs=self.daily[['long','short']]
        plt.rcParams.update({'figure.dpi':110,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                             'axes.titleweight':'semibold','axes.labelcolor':INK,'text.color':INK,'axes.titlesize':12})

    def actual(self,date):return self.daily.loc[:date].index[-1]

    def sample(self,date,h=5,window=252,book='broad'):
        cols=['long','short'] if book=='broad' else [book+'_long',book+'_short']
        a=self.daily.loc[:date,cols].rolling(h,min_periods=h).sum().tail(window).copy()
        a.columns=['long','short'];return a

    @staticmethod
    def decorate(ax,xlabel=None,ylabel=None):
        ax.grid(alpha=.15)
        if xlabel:ax.set_xlabel(xlabel)
        if ylabel:ax.set_ylabel(ylabel)

    def exposure(self,date,years=(1975,2022)):
        f=self.daily.loc[str(years[0]):str(years[1])]
        fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
        ax[0].plot(f.index,f.net.cumsum(),color=INK,lw=1)
        ax[0].set_title('A long-winner / short-loser exposure—not a funded wealth index')
        ax[0].set_ylabel('Accumulated fixed-notional P&L (pp)')
        ax[1].plot(f.index,f.net,color=BLUE,lw=.5)
        ax[1].set_ylabel('Daily P&L (pp)')
        for a in ax:
            a.axvline(self.actual(date),color=ORANGE,lw=1.3,label='Selected historical date');a.grid(alpha=.15)
        ax[0].legend(loc='upper left');fig.supxlabel('Full-history context includes dates after the cursor; not an as-of input')
        return fig

    def path(self,date,h=5):
        x=self.daily.loc[:date,'net'].tail(h);p=np.r_[0,x.cumsum().to_numpy()]
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained');t=np.arange(h+1)
        ax.plot(t,p,marker='o',ms=4,color=BLUE);i=int(p.argmin())
        ax.scatter([i,h],[p[i],p[-1]],color=[ORANGE,INK],s=55,zorder=4)
        ax.axhline(0,color=MUTED,lw=.8);ax.axhline(-2.16,color=ORANGE,ls=':',label='Fixed illustrative loss level −2.16 pp')
        ax.annotate(f'Low {p[i]:+.2f}',(i,p[i]),xytext=(5,-18),textcoords='offset points',color=ORANGE)
        ax.annotate(f'Endpoint {p[-1]:+.2f}\nRecovery {p[-1]-p[i]:.2f}',(h,p[-1]),xytext=(-105,12),textcoords='offset points')
        ax.set_title(f'{h} completed sessions ending {x.index[-1]:%Y-%m-%d}: daily-close path')
        ax.set_xlabel('Session since starting at zero');ax.set_ylabel('Fixed-notional cumulative P&L (pp)');ax.legend(loc='best');ax.grid(alpha=.15)
        return fig

    def core(self,date):
        year=pd.Timestamp(date).year
        if year not in self.models:
            cutoff=pd.Timestamp(year-1,12,31);w=weekly(self.daily.Mom.loc[:cutoff]);w=w.loc[w.index<=cutoff]
            self.models[year]=MomentumRiskCore().fit(w,cutoff,daily_returns=self.daily.Mom)
        model=self.models[year];sigma=self.daily.sigma.loc[:date].iloc[-1]
        return model,sigma,model.forecast(sigma)

    def baseline(self,date):
        model,sigma,f=self.core(date);x=self.daily.loc[:date].tail(126)
        fig,ax=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
        ax[0].plot(x.index,x.Mom,color=MUTED,lw=.6,label='Daily P&L')
        ax[0].plot(x.index,x.sigma,color=BLUE,label='EWMA scale')
        ax[0].set_title('1. Observe current scale');ax[0].tick_params(axis='x',rotation=35);ax[0].legend(fontsize=8)
        shocks=np.sort(model.residuals);cdf=np.arange(1,len(shocks)+1)/len(shocks)
        ax[1].step(shocks,cdf,where='post',color=BLUE);ax[1].set_title('2. Freeze 260 weekly shocks')
        ax[1].set_xlabel('Weekly P&L / preceding scale');ax[1].set_ylabel('Empirical cumulative fraction')
        outcomes=shocks*sigma*np.sqrt(5)
        ax[2].step(outcomes,cdf,where='post',color=BLUE)
        # On a CDF probability is a HEIGHT, not area under the curve.
        ax[2].plot([-2.16,-2.16],[0,f['probability']],color=ORANGE,lw=2)
        ax[2].scatter([-2.16],[f['probability']],color=ORANGE,s=35,zorder=5)
        ax[2].axvline(-2.16,color=ORANGE,ls='--',label='Loss threshold')
        ax[2].axvline(f['q05_pp'],color=INK,ls=':',label='5th percentile')
        ax[2].set_title('3. Rescale to selected date');ax[2].set_xlabel('Next-five-session P&L proxy (pp)');ax[2].legend(fontsize=8)
        fig.suptitle(f'As of {self.actual(date):%Y-%m-%d} | P(loss < −2.16) = {f["probability"]:.1%} | q05 = {f["q05_pp"]:.2f} pp')
        for a in ax:a.grid(alpha=.15)
        return fig

    def validation(self,state='all'):
        f=self.evaluation.copy()
        if state=='bear':f=f[f.bear]
        elif state=='nonbear':f=f[~f.bear]
        fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        ax[0].plot(f.end,f.y,color=MUTED,alpha=.55,lw=.6,label='Realised weekly P&L')
        ax[0].plot(f.end,f.q_fhs5y_ewma,color=BLUE,lw=.8,label='Forecast q05')
        bad=f.y<f.q_fhs5y_ewma;ax[0].scatter(f.loc[bad,'end'],f.loc[bad,'y'],s=8,color=ORANGE,label='q05 breach')
        ax[0].legend(fontsize=8);ax[0].set_ylabel('P&L (pp)');ax[0].set_title(f'q05 breaches: {bad.mean():.1%} (nominal 5%)')
        prob=f.p_fhs5y_ewma.to_numpy();y=f.event.to_numpy(float)
        # Fixed bins on the full evaluation sample, unchanged by state control.
        edges=np.unique(np.quantile(self.evaluation.p_fhs5y_ewma,[0,.2,.4,.6,.8,1]));edges[0]=-np.inf;edges[-1]=np.inf
        bins=np.digitize(prob,edges[1:-1]);rng=np.random.default_rng(20260912);draws=[];n=len(self.evaluation)
        # Resample chronological ALL-week blocks, then apply state filter. Filtering
        # before sampling would incorrectly make separated bear dates adjacent.
        full=self.evaluation;group=np.digitize(full.p_fhs5y_ewma.to_numpy(),edges[1:-1])
        for _ in range(400):
            starts=rng.integers(0,n, int(np.ceil(n/26)));ix=((starts[:,None]+np.arange(26))%n).ravel()[:n]
            use=np.ones(len(ix),dtype=bool) if state=='all' else full.bear.to_numpy()[ix]==(state=='bear')
            ix=ix[use];draws.append([full.event.to_numpy()[ix][group[ix]==b].mean() if np.any(group[ix]==b) else np.nan for b in range(len(edges)-1)])
        draws=np.asarray(draws)
        for b in range(len(edges)-1):
            m=bins==b
            if not m.any():continue
            px=prob[m].mean();py=y[m].mean();lo,hi=np.nanquantile(draws[:,b],[.025,.975])
            ax[1].plot([px,px],[lo,hi],color=BLUE)
            ax[1].scatter(px,py,color=BLUE,label=f'Bin {b+1}: {m.sum()} weeks / {int(y[m].sum())} events')
            ax[1].annotate(str(b+1),(px,py),xytext=(5,6),textcoords='offset points',fontsize=8)
        populated=draws[:,np.isfinite(draws).any(axis=0)]
        upper=min(1.,max(.35,float(prob.max())*1.1,float(np.nanquantile(populated,.975,axis=0).max())*1.1))
        ax[1].plot([0,upper],[0,upper],ls=':',color=MUTED);ax[1].set_xlim(0,upper);ax[1].set_ylim(0,min(1,upper))
        ax[1].set_xlabel('Mean forecast loss probability');ax[1].set_ylabel('Observed event fraction')
        ax[1].set_title('Calibration: 26-week block-bootstrap bands')
        ax[1].legend(loc='lower right',fontsize=8,title='Bin support',title_fontsize=9)
        fig.suptitle(f'FIXED evaluation 1984–2022 | {state} | {len(f)} weeks, {int(f.event.sum())} events')
        for a in ax:a.grid(alpha=.15)
        return fig

    def scores(self):
        rows=[]
        for name,pcol,qcol in [('Historical frequency','p_frequency','q_historical'),('126-session volatility logit','p_logit_rv126',None),('Matched EWMA logit','p_logit_ewma',None),('Retained empirical/volatility','p_fhs5y_ewma','q_fhs5y_ewma')]:
            for state in ['all','bear','nonbear']:
                f=self.evaluation if state=='all' else self.evaluation[self.evaluation.bear==(state=='bear')]
                p=f[pcol].clip(1e-8,1-1e-8);y=f.event.astype(float)
                rows.append(dict(model=name,state=state,weeks=len(f),events=int(y.sum()),predicted=p.mean(),observed=y.mean(),
                    brier=np.mean((p-y)**2),log_loss=-np.mean(y*np.log(p)+(1-y)*np.log(1-p)),
                    q05_breach=np.mean(f.y<f[qcol]) if qcol else np.nan))
        return pd.DataFrame(rows)

    def distributions(self,a='2015-04-30',b='2018-11-30',h=5,w=252,standard=False):
        fig,ax=plt.subplots(1,2,figsize=(11.5,4.7),layout='constrained')
        rows=[]
        for date,color in [(a,BLUE),(b,ORANGE)]:
            sample=self.sample(date,h,w).sum(axis=1);v=sample.to_numpy();raw=v.copy()
            if standard:v=(v-v.mean())/v.std(ddof=1)
            q=np.quantile(v,[.15,.5,.85]);tail=tail_weights(v);tailmean=tail@v/tail.sum()
            order=np.sort(v);ax[0].step(order,np.arange(1,len(v)+1)/len(v),where='post',label=date,color=color)
            ax[0].scatter(q,[.15,.5,.85],color=color,s=35)
            rows.append([date,*[f'{x:.2f}' for x in [raw.mean(),raw.std(ddof=1),np.quantile(raw,.15),np.median(raw),tail_weights(raw)@raw/(.15*len(raw))]]])
            ax[0].axvline(tailmean,color=color,ls=':',alpha=.65)
        ax[0].axhline(.15,color=MUTED,lw=.8);ax[0].legend();ax[0].grid(alpha=.15)
        ax[0].set_xlabel('Within-window standard deviations' if standard else 'Trailing P&L (pp)')
        ax[0].set_ylabel('Empirical cumulative fraction');ax[0].set_title('Dots: q15 / median / q85; dotted line: lower-tail mean')
        ax[1].axis('off');table=ax[1].table(cellText=rows,colLabels=['As of','Mean','SD','q15','Median','Tail mean'],loc='center',cellLoc='center',colWidths=[.25,.14,.14,.14,.14,.18])
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.9)
        ax[1].set_title('Original units, even when the curve is standardised')
        fig.suptitle(f'{h}-session outcomes; {w} ending observations | selected comparisons, not held-out evidence')
        return fig

    def leg_view(self,date,h=5,w=252,book='broad',selected=0):
        a=self.sample(date,h,w,book);v=a.to_numpy();y=v.sum(axis=1);weights=tail_weights(y);mask=weights>0
        z=weights/weights.sum();con=z@v;indices=np.flatnonzero(mask);idx=indices[min(selected,len(indices)-1)]
        fig,ax=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
        ax[0].bar(['Winner long','Loser short','Net'],[con[0],con[1],con.sum()],color=[BLUE,ORANGE,INK]);ax[0].axhline(0,color=MUTED,lw=.8)
        ax[0].set_ylabel('Average contribution on same net-tail windows (pp)');ax[0].set_title('Contributions reconcile exactly')
        ax[1].scatter(v[:,0],v[:,1],s=12,alpha=.2,color=MUTED,label='Other windows')
        ax[1].scatter(v[mask,0],v[mask,1],s=20,color=ORANGE,label='Net-tail windows')
        ax[1].scatter(v[idx,0],v[idx,1],s=90,facecolors='none',edgecolors=INK,linewidths=1.5)
        limit=max(abs(v).max()*1.08,1);x=np.array([-limit,limit])
        for level in [0,np.quantile(y,.15)]:ax[1].plot(x,level-x,color=MUTED,ls=':',lw=.7)
        ax[1].axhline(0,color=MUTED,lw=.7);ax[1].axvline(0,color=MUTED,lw=.7)
        ax[1].set_xlim(-limit,limit);ax[1].set_ylim(-limit,limit)
        ax[1].set_xlabel('Winner-long contribution (pp)');ax[1].set_ylabel('Short-loser contribution (pp)')
        ax[1].set_title(f'Circled window ends {a.index[idx]:%Y-%m-%d}');ax[1].legend(fontsize=8)
        fig.suptitle(f'{date} | {book} | {h} sessions / {w} observations; point selection does not redefine the tail')
        return fig

    def episodes(self,date,h=5,w=252,chosen=1,exclude=True):
        a=self.sample(date,h,w);y=a.sum(axis=1).to_numpy();weights=tail_weights(y);indices=np.flatnonzero(weights>0);groups=[]
        for i in indices:
            if not groups or i-h+1>groups[-1][-1]:groups.append([int(i)])
            else:groups[-1].append(int(i))
        pick=min(chosen,len(groups))-1;p=profile(a.to_numpy(),h)
        fig,ax=plt.subplots(2,1,figsize=(11,5.8),layout='constrained',gridspec_kw={'height_ratios':[1.1,1]})
        ax[0].plot(a.index,y,color=MUTED,lw=.7)
        for j,g in enumerate(groups):
            ax[0].scatter(a.index[g],y[g],s=25 if j==pick else 10,color=ORANGE if j==pick else BLUE,alpha=1 if j==pick else .5)
        ax[0].set_ylabel('Trailing P&L (pp)');ax[0].set_title(f'{len(groups)} connected episodes | highlighted {pick+1}; overlapping starts, not independent crashes')
        vals=[p['long'],p['short']];x=np.arange(2)
        ax[1].bar(x-.17,vals,.34,label='Original joint tail',color=BLUE)
        if exclude and p['loo_long'] is not None:ax[1].bar(x+.17,[p['loo_long'],p['loo_short']],.34,label='Largest episode omitted (sensitivity)',color=ORANGE)
        ax[1].set_xticks(x,['Winner long','Loser short']);ax[1].axhline(0,color=MUTED,lw=.7);ax[1].legend(fontsize=8)
        share=p['largest_episode_gross_loss_share']
        ax[1].set_title(f'Largest episode: {share:.0%} of adverse loss mass' if share is not None else 'No negative adverse loss mass')
        ax[1].set_ylabel('Contribution (pp)');fig.suptitle(f'{date} | {h} sessions / {w} observations')
        return fig

    def sizes(self,date,h=5,w=252):
        rows=[]
        for book in ['broad','small','big']:
            p=profile(self.sample(date,h,w,book).to_numpy(),h);rows.append([book,p['tail_mean'],p['long'],p['short'],p['episodes']])
        return pd.DataFrame(rows,columns=['Book','Lower15% mean','Long contribution','Short contribution','Episodes'])

    def atlas_plot(self):
        f=self.atlas[(self.atlas.book=='broad')&(self.atlas.window==252)]
        fig,ax=plt.subplots(2,1,figsize=(11,5.5),sharex=True,layout='constrained')
        for a,h in zip(ax,[5,20]):
            s=f[f.h==h];dates=pd.to_datetime(s.date)
            a.stackplot(dates,s.winner_loss_share,s.loser_rally_share,s.both_lose_share,s.neither_loses_share,
                        colors=[BLUE,ORANGE,INK,'#c6cbd0'],labels=['Winner loss / short offsets','Loser rally / long offsets','Both lose','Neither loses'])
            a.set_ylim(0,1);a.set_ylabel(f'{h}-day tail mass share')
        ax[0].legend(loc='upper left',ncol=2,fontsize=8);fig.suptitle('Full calendar sequence: mechanism mixtures, not pure regimes')
        return fig

    def changes(self,prior,current,h=5):
        old=self.sample(prior,h);new=self.sample(current,h)
        # The decomposition requires overlapping samples, not arbitrary distant dates.
        if len(old.index.intersection(new.index))<30:raise ValueError('Choose dates less than about one trading year apart.')
        c=change(old,new);fig,ax=plt.subplots(1,3,figsize=(12,4.8))
        fig.subplots_adjust(top=.78,bottom=.2,wspace=.3)
        for j,title in enumerate(['Tail mean','Winner-long contribution','Short-loser contribution']):
            v=[c['old'][j],c['entering'][j],c['leaving'][j],c['new'][j]]
            bottom=[0,v[0],v[0]+v[1],0]
            ax[j].bar(['Previous','Entering','Leaving','Current'],v,bottom=bottom,color=[INK,BLUE,ORANGE,INK])
            for i in range(4):
                ax[j].annotate(f'{v[i]:+.2f}',(i,bottom[i]+v[i]),xytext=(0,7 if v[i]>=0 else -13),
                               textcoords='offset points',ha='center',fontsize=8,
                               bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.8))
            extents=[0,*bottom,*[bottom[i]+v[i] for i in range(4)]]
            lower,upper=min(extents),max(extents);span=max(upper-lower,1.)
            ax[j].set_ylim(lower-.18*span,upper+.18*span)
            ax[j].axhline(0,color=MUTED,lw=.7);ax[j].set_title(title);ax[j].tick_params(axis='x',rotation=25)
        fig.suptitle(f'{prior} → {current} | {h} sessions: exact window accounting, not market causation',y=.97,fontsize=11)
        return fig

    def scenario(self,date,bear=True,move=5.):
        year=str(pd.Timestamp(date).year);meta=self.scenarios[year];f=pd.DataFrame(meta['rows']);f=f[f.bear==bear]
        selected=f.loc[np.isclose(f.market,move)].iloc[0]
        fig,ax=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
        for c,label,color in [('long','Winner long',BLUE),('short','Loser short',ORANGE),('net','Net mean',INK)]:
            ax[0].plot(f.market,f[c],label=label,color=color,lw=1.8)
        ax[0].fill_between(f.market,f.mean_lo,f.mean_hi,color=INK,alpha=.1,label='95% pointwise mean CI (percentile block bootstrap)')
        ax[0].axvline(move,color=MUTED,ls=':');ax[0].axhline(0,color=MUTED,lw=.7);ax[0].legend(fontsize=8)
        ax[0].set_xlabel('Specified weekly market move (%)');ax[0].set_ylabel('Conditional mean factor P&L (pp)')
        ax[0].set_title('Historical association—not a causal or unconditional forecast')
        ax[1].bar(f.market,f.nearby,width=.4,color=[ORANGE if n<20 else BLUE for n in f.nearby]);ax[1].axhline(20,color=MUTED,ls=':')
        ax[1].set_xlabel('Market move; nearby = ±1 percentage point');ax[1].set_ylabel('Same-state training weeks')
        ax[1].set_title(f'Selected: {selected.nearby:.0f} nearby; {selected.nearby_last10y:.0f} in last ten years')
        fig.suptitle(f'{date}: hypothetical {"bear" if bear else "non-bear"} state | fit through {meta["fit_cutoff"]} | net {selected.net:+.2f} pp')
        return fig

    def assessment(self,date,bear=True,move=5.):
        """One PM-facing record; keep forecasts, descriptions and scenarios separate."""
        r=self.snapshots[date];p=r['historical_trailing_description']['5']
        model,scale,f=self.core(date)
        warnings='; '.join(r['description_warnings']['5']) or 'The dominant leg agrees across the tested lookbacks and survives largest-episode omission; no concentration flag fired. These checks do not establish stability or safety.'
        meta=self.scenarios[str(pd.Timestamp(date).year)]
        scenario=next(x for x in meta['rows'] if x['bear']==bear and np.isclose(x['market'],move))
        leg='winner-long' if p['long']<p['short'] else 'short-loser'
        return f'''### Historical PM assessment — {date}

Public US equity momentum factor; data through **{r['last_observation']}**. Not today's reading or a return estimate for your actual portfolio. **Exposure convention:** one dollar long and one dollar short, reset daily. **1 pp means one cent of P&L on that paired position**, not per dollar of gross exposure or fund NAV.

**1. Risk level — weekly downside baseline.** Estimated probability of loss below −2.16 pp: **{f['probability']:.1%}**. Model fifth-percentile outcome: **{f['q05_pp']:.2f} pp**. Five-session scaling approximates a calendar week; q05 is not a worst-case bound. Annual fit through {model.fitted_asof:%Y-%m-%d}. This is predominantly a volatility-driven estimate, not a forecast of a catalyst or first reversal.

**2. Exposure to investigate — completed historical losses.** The worst 15% of the latest **252 overlapping five-session windows**, ending on trading days over roughly one year, averaged **{p['tail_mean']:.2f} pp**: winner-long **{p['long']:+.2f}**, short-loser **{p['short']:+.2f}**. The **{leg} leg** contributed more adversely. There are **{p['episodes']} connected episodes**: selected windows sharing underlying trading sessions are grouped together, including chains of overlaps—not counted as independent crises. Qualification: {warnings}

**3. Optional historical scenario — not another forecast.** For a **{move:+.1f}% weekly market move** in a hypothetical preceding **{'bear' if bear else 'non-bear'}** state, the regression's net mean is **{scenario['net']:+.2f} pp**. Pointwise 95% mean interval: **[{scenario['mean_lo']:+.2f}, {scenario['mean_hi']:+.2f}] pp**. Support: **{scenario['nearby']}** nearby same-state training weeks, **{scenario['nearby_last10y']}** in the last ten years. Treat this relationship as context, not a dependable stress-loss limit or an assertion of the actual state.

**Next investigation, not a trade instruction:** verify whether the PM's current book shares the {leg} exposure, then seek dated company/catalyst and positioning evidence. Prices identify the contributing leg, not the cause. Do not infer a squeeze or size a hedge from this page alone.
'''

    def snapshot(self,date):
        r=self.snapshots[date];f=r['separate_forward_research_estimate'];p=r['historical_trailing_description']['5']
        s=r['historical_trailing_description']['20'];warnings=r['description_warnings']
        prior=str((pd.Timestamp(date)-pd.offsets.MonthEnd(1)).date());c=change(self.sample(prior),self.sample(date))
        return f'''### Historical research snapshot — {date}

Data through **{r['last_observation']}**. Not a current reading or actual-book estimate.

**Ahead — existing five-session model:** probability below −2.16 pp **{f['probability']:.1%}**; q05 **{f['q05_pp']:.2f} pp**. These are exploratory model estimates; earlier evaluation found bear-state probability underprediction.

**Recently — trailing five-session distribution:** lower-15% mean **{p['tail_mean']:.2f} pp**, comprising winner long **{p['long']:+.2f}** and short loser **{p['short']:+.2f}**. This is historical attribution, not a forecast. The tail contains **{p['episodes']} connected episodes**.

**Five-session qualification:** {('; '.join(warnings['5'])) or 'No rule-based sensitivity flags; this does not establish confidence or safety.'}

**Change since {prior}:** tail mean changed **{c['delta'][0]:+.2f} pp**; entering observations **{c['entering'][0]:+.2f}**, leaving observations **{c['leaving'][0]:+.2f}**. Order-averaged measurement accounting.

**Longer-path qualification:** the twenty-session tail has **{s['episodes']} connected episodes**. {('; '.join(warnings['20'])) or 'No rule-based sensitivity flags; this does not establish confidence or safety.'}

**Applicability:** latest cached historical vintage; no verified mapping to current holdings. Trailing lower-tail averages, forward q05 and conditional scenario means answer different questions.
'''
