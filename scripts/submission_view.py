"""Presentation selection only; frozen forecasts and research fits remain unchanged."""
import numpy as np
import matplotlib.pyplot as plt

class SubmissionView:
    def __init__(self,e):self.e=e

    def rolling(self,date,move):
        s=self.e.state(date);tr=self.e.training(s.date.year)
        return {leg:float((tr[leg]-tr['beta_'+leg]*tr.market).mean()+s['beta_'+leg]*move)
                for leg in ['long','short']}

    def assessment(self,date):
        from submission_narrative import assessment
        return assessment(self,date)


    def verify(self):
        for year in [1984,2009,2022]:
            te=self.e.predictions.loc[self.e.predictions.start.dt.year.eq(year)].iloc[0]
            tr=self.e.training(year)
            for leg in ['long','short']:
                value=(tr[leg]-tr['beta_'+leg]*tr.market).mean()+te['beta_'+leg]*te.market
                assert np.isclose(value,te['rolling_'+leg])

    def scenario_plot(self,date):
        moves=np.arange(-5,5.01,.5);sc=self.e.scenarios(date,tuple(moves))
        fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for leg,color in [('long','#236b8e'),('short','#be653d'),('net','#243342')]:
            y=[self.rolling(date,m) for m in moves]
            vals=[sum(v.values()) if leg=='net' else v[leg] for v in y]
            ax[0].plot(moves,vals,label='Selected rolling beta: '+leg,color=color,lw=2)
        ax[0].plot(moves,sc.net,'--',color='gray',label='Nonlinear challenger: net',alpha=.8)
        ax[0].set(title='Selected reference: supplied market scenario',xlabel='Market move (%)',ylabel='Conditional mean P&L (pp)')
        ax[0].axhline(0,color='gray',lw=.5);ax[0].legend(fontsize=8)
        ax[1].plot(moves,sc.nearby_joint,label='Challenger joint neighbours')
        ax[1].plot(moves,sc.nearby_joint_recent,label='Of which in last ten years')
        ax[1].set(title='Support diagnostic for nonlinear challenger',xlabel='Market move (%)',ylabel='Historical weeks');ax[1].legend(fontsize=8)
        for a in ax:a.grid(alpha=.15)
        return fig

    def performance_plot(self):
        p=self.e.predictions;fig,ax=plt.subplots(2,1,figsize=(11,6),layout='constrained')
        for col,label,color in [('rolling','Selected rolling beta','#243342'),('dynamic','Nonlinear challenger','#236b8e'),('vol_response','Volatility-aware comparator','#be653d')]:
            error=abs(p[col]-p.net)
            ax[0].plot(p.end,error.rolling(156).mean(),label=label,color=color)
        ax[0].set(title='Conditional error: 156-week trailing MAE',ylabel='Absolute error (pp)');ax[0].legend(fontsize=8)
        annual=(abs(p['rolling']-p.net)-abs(p.vol_response-p.net)).groupby(p.end.dt.year).mean()
        ax[1].bar(annual.index,annual);ax[1].axhline(0,color='gray')
        ax[1].set(title='Selected reference minus volatility comparator; negative is better',xlabel='Evaluation year',ylabel='MAE difference (pp)')
        return fig
