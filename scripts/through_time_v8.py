"""Descriptive temporal diagnostics of existing predictions; never fit a model here."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BLUE='#236b8e'; ORANGE='#be653d'; INK='#243342'

def weekly_frame(evaluation,logit_col='p_logit_rv126'):
    f=evaluation.sort_values('end').reset_index(drop=True).copy()
    assert f.end.is_unique and (f.cutoff<f.start).all()
    assert np.array_equal(f.event.astype(bool),f.y.lt(-2.16))
    f['event']=f.event.astype(int)
    f['breach']=f.y.lt(f.q_fhs5y_ewma).astype(int)
    f['comparison_logit']=f[logit_col]
    for label,col in [('retained','p_fhs5y_ewma'),('frequency','p_frequency'),('logit',logit_col)]:
        f['brier_'+label]=(f[col]-f.event)**2
    return f

def rolling_frame(evaluation,window=156,logit_col='p_logit_rv126'):
    """At each week end include only that week and preceding completed outcomes."""
    f=weekly_frame(evaluation,logit_col)
    cols=['p_fhs5y_ewma','comparison_logit','event','breach','brier_retained','brier_frequency','brier_logit']
    r=f[cols].rolling(window,min_periods=window).mean()
    r['events']=f.event.rolling(window,min_periods=window).sum()
    r['end']=f.end
    return r

def performance_plot(evaluation,window=156,logit_col='p_logit_rv126'):
    r=rolling_frame(evaluation,window,logit_col)
    logit_label='matched EWMA logit' if logit_col=='p_logit_ewma' else '126-session volatility logit'
    fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True,layout='constrained')
    ax[0].plot(r.end,100*r.p_fhs5y_ewma,label='Mean forecast: retained',color=BLUE)
    ax[0].plot(r.end,100*r.comparison_logit,label='Mean forecast: '+logit_label,color=INK,ls=':')
    ax[0].plot(r.end,100*r.event,label='Realised severe-loss fraction',color=ORANGE)
    ax[0].set_ylabel('Percent');ax[0].set_title('1. Did predicted risk match subsequent event frequency?')
    ax[1].plot(r.end,100*r.breach,color=BLUE,label='Realised q05 breach fraction')
    ax[1].axhline(5,color=ORANGE,ls='--',label='Nominal 5%')
    ax[1].set_ylabel('Percent');ax[1].set_title('2. How often did losses exceed the model’s downside quantile?')
    for other,color in [('frequency',ORANGE),('logit',INK)]:
        ax[2].plot(r.end,r.brier_retained-r['brier_'+other],label='Retained minus '+(logit_label if other=='logit' else other),color=color)
    ax[2].axhline(0,color='grey',lw=.8)
    ax[2].set_ylabel('Brier loss difference');ax[2].set_title('3. Negative = retained model better on these same weeks')
    ax[3].plot(r.end,r.events,color=INK,label='Severe-loss weeks in the rolling window')
    ax[3].set_ylabel('Event count');ax[3].set_xlabel('Window ending date (evaluation, not a forecast)')
    ax[3].set_title('4. How much tail evidence supports the comparison?')
    for a in ax:a.grid(alpha=.15);a.legend(fontsize=8,loc='upper left')
    fig.suptitle(f'Performance through time | trailing {window} completed weeks | no model refitting or selection here')
    return fig

def episode_ledger(evaluation):
    """All consecutive runs of event weeks, not economic or independent crashes."""
    f=weekly_frame(evaluation); event=f.event.to_numpy(bool)
    starts=np.flatnonzero(event & ~np.r_[False,event[:-1]])
    ends=np.flatnonzero(event & ~np.r_[event[1:],False]);rows=[]
    for ident,(a,b) in enumerate(zip(starts,ends),1):
        before=f.iloc[max(0,a-4):a]; during=f.iloc[a:b+1];after=f.iloc[b+1:b+5]
        rows.append(dict(episode=ident,first=a,last=b,start=str(f.start.iloc[a].date()),end=str(f.end.iloc[b].date()),
            weeks=b-a+1,total_pp=float(during.y.sum()),worst_week_pp=float(during.y.min()),
            first_week_p=float(f.p_fhs5y_ewma.iloc[a]),first_week_logit=float(f.p_logit_rv126.iloc[a]),
            prior4_mean_p=float(before.p_fhs5y_ewma.mean()),prior_weeks=len(before),
            during_mean_p=float(during.p_fhs5y_ewma.mean()),after4_mean_p=float(after.p_fhs5y_ewma.mean()),after_weeks=len(after)))
    result=pd.DataFrame(rows)
    assert result.weeks.sum()==f.event.sum()
    return result

def alarm_table(evaluation,threshold=.20):
    f=weekly_frame(evaluation); alarm=f.p_fhs5y_ewma.ge(threshold);event=f.event.eq(1)
    episodes=episode_ledger(evaluation)
    return pd.DataFrame([
        {'Question':'Elevated forecast weeks', 'Count':int(alarm.sum()),'Denominator':len(f)},
        {'Question':'Elevated forecast + severe loss','Count':int((alarm&event).sum()),'Denominator':int(alarm.sum())},
        {'Question':'Elevated forecast without severe loss','Count':int((alarm&~event).sum()),'Denominator':int(alarm.sum())},
        {'Question':'Severe loss below elevated threshold','Count':int((~alarm&event).sum()),'Denominator':int(event.sum())},
        {'Question':'Event runs elevated before first loss week','Count':int(episodes.first_week_p.ge(threshold).sum()),'Denominator':len(episodes)}])

def episode_plot(evaluation,ident=1,logit_col='p_logit_rv126'):
    f=weekly_frame(evaluation,logit_col);ledger=episode_ledger(evaluation);ep=ledger.loc[ledger.episode==ident].iloc[0]
    a,b=int(ep['first']),int(ep['last']);s=f.iloc[max(0,a-8):min(len(f),b+9)]
    fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    ax[0].plot(s.end,100*s.p_fhs5y_ewma,'o-',color=BLUE,label='Retained probability known before week')
    ax[0].plot(s.end,100*s.comparison_logit,color=INK,ls=':',label=('Matched EWMA' if logit_col=='p_logit_ewma' else '126-session volatility')+' logit probability')
    ax[0].axhline(20,color=ORANGE,ls='--',label='Illustrative elevated level: 20%')
    ax[0].set_ylabel('Probability (%)');ax[0].legend(fontsize=8)
    ax[1].bar(s.end,s.y,width=4,color=[ORANGE if v else BLUE for v in s.event])
    ax[1].axhline(-2.16,color=INK,ls='--',label='Severe-loss threshold: −2.16 pp')
    ax[1].set_ylabel('Realised weekly P&L (pp)');ax[1].set_xlabel('Outcome week ending date; probabilities were available at preceding cutoff')
    ax[1].legend(fontsize=8)
    for axis in ax:
        axis.axvspan(pd.Timestamp(ep.start),pd.Timestamp(ep.end),color='grey',alpha=.15)
        axis.grid(alpha=.15)
    fig.suptitle(f'Event run {ident}: {ep.start} to {ep.end} | first-week probability {ep.first_week_p:.1%}\nGrey = consecutive severe-loss weeks, selected retrospectively; not an economic crisis label')
    return fig

def scenario_time_plot(predictions,window=156):
    fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for name,color in [('linear',ORANGE),('asymmetric',INK),('state asymmetric',BLUE)]:
        f=predictions[predictions.model==name].sort_values('end')
        dates=pd.to_datetime(f.end)
        ax[0].plot(dates,f.error.abs().rolling(window,min_periods=window).mean(),color=color,label=name)
        ax[1].plot(dates,100*f.covered.rolling(window,min_periods=window).mean(),color=color,label=name)
    ax[0].set_ylabel('Mean absolute error (pp)');ax[0].set_title('Conditional mean error: lower is better')
    ax[1].set_ylabel('Outcome coverage (%)');ax[1].set_xlabel('Trailing 156-week evaluation window ending date')
    ax[1].axhline(90,color='grey',ls='--',label='Nominal 90% outcome band')
    ax[1].set_title('Separate residual outcome bands—not the scenario mean-confidence shading')
    for a in ax:a.grid(alpha=.15);a.legend(fontsize=8)
    fig.suptitle('Scenario relationship through time | realised market move supplied, not forecast')
    return fig

def self_test(evaluation):
    f=weekly_frame(evaluation);r=rolling_frame(f)
    assert r.event.iloc[:155].isna().all()
    np.testing.assert_allclose(r.event.iloc[155],f.event.iloc[:156].mean())
    for stop in [200,800,1500]:
        pd.testing.assert_frame_equal(rolling_frame(f.iloc[:stop]),r.iloc[:stop])
    ledger=episode_ledger(f)
    for row in ledger.itertuples():
        assert f.event.iloc[row.first:row.last+1].eq(1).all()
        assert row.first==0 or f.event.iloc[row.first-1]==0
        assert row.last==len(f)-1 or f.event.iloc[row.last+1]==0
        assert row.first_week_p==f.p_fhs5y_ewma.iloc[row.first]
    return 'PASS: trailing-only rolling prefix replay; initial-window arithmetic; exhaustive event runs and pre-outcome probability alignment.'
