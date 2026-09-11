"""Dated economic vulnerability views; no downloads, text, or trading rules."""
from functools import lru_cache
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from economic_extension import OUT, BASE, daily_features, design, SEED
from companion_v8 import Companion
from distribution_panel import profile


class EconomicCompanion:
    def __init__(self, root):
        self.core = Companion(root)
        self.daily = self.core.daily
        self.panel = pd.read_parquet(OUT/'panel.parquet')
        self.features = {w: daily_features(self.daily,w) for w in [126,252,504]}
        self.predictions = pd.read_parquet(OUT/'conditional_252.parquet')
        self.fits = {}

    def state(self,date,window=252):
        date=self.daily.loc[:date].index[-1]
        row=self.features[window].loc[date].copy()
        row['bear']=bool(self.daily.loc[date,'market_2y']<0)
        row['scale']=np.sqrt(5)*self.daily.loc[date,'sigma']
        row['log_scale']=np.log(row['scale'])
        row['date']=date
        return row

    def training(self,year):
        cutoff=self.panel.loc[self.panel.start.dt.year.eq(year),'cutoff'].min()
        return self.panel.loc[self.panel.end<=cutoff]

    def fitted(self,year):
        if year not in self.fits:
            tr=self.training(year); x=design(tr,True); y=tr[['long','short']].to_numpy()
            coef=np.linalg.lstsq(x,y,rcond=None)[0]
            rng=np.random.default_rng(SEED+year); n=len(tr); boots=[]
            for _ in range(300):
                starts=rng.integers(0,n,size=int(np.ceil(n/26)))
                ix=(starts[:,None]+np.arange(26)).ravel()[:n]%n
                if np.linalg.matrix_rank(x[ix])==8:
                    boots.append(np.linalg.lstsq(x[ix],y[ix],rcond=None)[0])
            assert len(boots)>=270
            self.fits[year]=(coef,np.stack(boots))
        return self.fits[year]

    def scenarios(self,date,moves=(-2.,0.,2.)):
        state=self.state(date); year=state.date.year; tr=self.training(year)
        coef,boots=self.fitted(year)
        q=pd.DataFrame([state.to_dict() for _ in moves]); q['market']=list(moves)
        x=design(q,True); fit=x@coef
        boot=np.einsum('ij,bjk->bik',x,boots).sum(axis=2)
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        rows=[]
        for i,m in enumerate(moves):
            near=tr.market.between(m-1,m+1)&tr.bear.eq(state.bear)
            joint=near & (tr.log_scale-state.log_scale).abs().le(.35) & (tr.beta_net-state.beta_net).abs().le(.3) & (tr.directional_gap-state.directional_gap).abs().le(.75)
            recent=tr.end.ge(state.date-pd.DateOffset(years=10))
            rows.append(dict(market=float(m),long=float(fit[i,0]),short=float(fit[i,1]),net=float(fit[i].sum()),
                             mean_lo=float(lo[i]),mean_hi=float(hi[i]),nearby_market_state=int(near.sum()),
                             nearby_joint=int(joint.sum()),nearby_joint_recent=int((joint&recent).sum())))
        result=pd.DataFrame(rows)
        assert np.allclose(result.long+result.short,result.net)
        return result


    def exposure_plot(self,date):
        fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
        f=self.features[252].loc['1984':date]
        axes[0].plot(self.daily.sigma.loc[f.index],color='#68757c',label='EWMA daily scale')
        axes[0].set_ylabel('Scale (pp)');axes[0].legend(loc='upper left')
        axes[1].plot(f.beta_long,label='Winner beta',color='#236b8e')
        axes[1].plot(-f.beta_short,label='Loser stock beta (not short P&L)',color='#be653d')
        axes[1].legend(loc='upper left');axes[1].set_ylabel('Market beta')
        axes[2].plot(f.beta_net,label='Net linear beta',color='#243342')
        axes[2].plot(f.up_beta_net,label='Net positive-market slope',alpha=.6,color='#be653d')
        axes[2].axhline(0,color='gray',lw=.8);axes[2].set_ylabel('Net sensitivity');axes[2].legend(loc='upper left')
        for ax in axes:ax.grid(alpha=.15)
        fig.suptitle('Risk magnitude and risk direction are different: all inputs trailing')
        return fig

    def scenario_plot(self,date):
        sc=self.scenarios(date,tuple(np.arange(-5,5.01,.5)))
        s=self.state(date)
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for col,color in [('long','#236b8e'),('short','#be653d'),('net','#243342')]:
            axes[0].plot(sc.market,sc[col],label=col,color=color)
        axes[0].fill_between(sc.market,sc.mean_lo,sc.mean_hi,color='#243342',alpha=.12,label='Net mean CI, not outcome band')
        axes[0].axhline(0,lw=.7,color='gray');axes[0].legend(fontsize=8)
        axes[0].set(title=f'Dated {s.date:%Y-%m-%d}: supplied market move',xlabel='Market return (%)',ylabel='Conditional mean P&L (pp)')
        axes[1].plot(sc.market,sc.nearby_market_state,label='Market + bear state only')
        axes[1].plot(sc.market,sc.nearby_joint,label='Also similar scale and exposures')
        axes[1].plot(sc.market,sc.nearby_joint_recent,label='Joint, last ten years')
        axes[1].set(title='Does history support this combination?',xlabel='Supplied market return (%)',ylabel='Nearby training weeks')
        axes[1].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.15)
        return fig

    def performance_plot(self):
        p=self.predictions
        e_dynamic=abs(p.dynamic-p.net);e_vol=abs(p.vol_response-p.net)
        fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
        axes[0].plot(e_dynamic.rolling(156).mean(),label='Exposure-conditioned',color='#236b8e')
        axes[0].plot(e_vol.rolling(156).mean(),label='Volatility-aware conditional benchmark',color='#be653d')
        axes[0].set(title='Conditional response errors through time: 156-week trailing MAE',ylabel='Absolute error (pp)');axes[0].legend(fontsize=8)
        annual=(e_dynamic-e_vol).groupby(p.end.dt.year).mean()
        axes[1].bar(annual.index,annual,color=np.where(annual<0,'#236b8e','#be653d'))
        axes[1].axhline(0,color='gray',lw=.8);axes[1].set(title='Annual paired difference: negative favours exposure model',ylabel='MAE difference (pp)',xlabel='Evaluation year')
        for ax in axes:ax.grid(alpha=.15)
        return fig

    def distribution_plot(self,date):
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for h in [1,5,10,20]:
            x=self.daily.net.loc[:date].rolling(h).sum().tail(252).dropna()
            z=np.sort((x-x.mean())/x.std());prob=np.arange(1,len(z)+1)/len(z)
            axes[0].step(z,prob,label=f'{h} sessions')
            q=x.quantile([.15,.5,.85]);axes[1].plot([.15,.5,.85],q,marker='o',label=f'{h} sessions')
        axes[0].set(title='Trailing shape after removing mean and SD',xlabel='Standard deviations within each horizon',ylabel='Empirical cumulative fraction')
        axes[1].set(title='Actual trailing quantiles retain horizon severity',xlabel='Percentile',ylabel='Summed P&L (pp)')
        for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
        return fig
