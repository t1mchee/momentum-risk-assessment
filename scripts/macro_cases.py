"""Two sourced historical interpretations of saved forecasts; no new fits."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat as nb

CASES=[('2009-03-06','2009-03-13','Rebound exposure: March 2009'),('2020-03-13','2020-03-20','A loss without a prior warning: March 2020')]

def case(e,p,index):
    cutoff,end,title=CASES[index];t=pd.Timestamp(cutoff);end=pd.Timestamp(end)
    ev=e.core.evaluation.loc[end];row=e.predictions.loc[end];state=e.state(t)
    assert ev.cutoff==t and row.cutoff==t and ev.sessions==5
    future=e.daily.loc[(e.daily.index>t)&(e.daily.index<=end)]
    assert len(future)==5
    for leg in ['long','short','net']:
        np.testing.assert_allclose(future[leg].sum(),row[leg],atol=1e-9)
    fit=p.rolling(t,row.market)
    np.testing.assert_allclose([fit['long'],fit['short']],[row.rolling_long,row.rolling_short],atol=1e-9)
    up=p.rolling(t,2);down=p.rolling(t,-2)
    before=f'''### {title}

**Assessment cutoff: {t:%d %B %Y}, after the close.** I selected this case from the existing loss-week results {'to examine a rebound with an elevated prior probability' if index==0 else 'because a loss occurred when its prior estimated probability was below the illustrative 20% warning threshold'}. Both cases use the next five trading days for the outcome. The selection is retrospective.

| Information at the cutoff | Value |
| --- | ---: |
| Prior two-year market return | {e.daily.loc[t,'market_2y']:.1%} |
| Winner-stock beta | {state.beta_long:.2f} |
| Loser-stock beta | {-state.beta_short:.2f} |
| Net beta | {state.beta_net:+.2f} |
| Estimated probability of next-week loss beyond 2.16 pp | {ev.p_fhs5y_ewma:.1%} |
| Estimated net response to a +2% market move | {sum(up.values()):+.2f} pp |
| Estimated net response to a −2% market move | {sum(down.values()):+.2f} pp |

'''
    if index==0:
        before+='''**Contemporaneous context.** On 3 March, Bernanke described a sharp economic contraction, some easing of funding strains, and continued financial stress. He discussed recovery as conditional on stabilising financial conditions. [Federal Reserve testimony, 3 March 2009](https://www.federalreserve.gov/newsevents/testimony/bernanke20090303a.htm).

**Interpretation using information through the cutoff.** A weak economy and an adverse rebound exposure can coexist: asset prices can respond to improving financing conditions or reduced downside expectations before activity recovers. The higher loser-stock beta suggests that a rally could hurt the short leg more than it helped the long leg. The statement does not establish that a rally was imminent.

I would examine whether the beta difference was concentrated in financial or other financing-sensitive stocks, and whether their credit spreads and earnings expectations were changing. A sector concentration or a few unusually responsive stocks could also explain the aggregate exposure. Reliable dated holdings and credit data would be needed to distinguish these possibilities.
'''
    else:
        before+='''**Contemporaneous context.** On 3 March, the FOMC cut its target range by half a percentage point, citing coronavirus risks to economic activity. That action and the market losses already observed were available by this cutoff. [FOMC statement, 3 March 2020](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200303a.htm).

**Interpretation using information through the cutoff.** The negative net beta suggests some protection against a broad market decline in the fitted relationship. I would be cautious about extending that relationship to an abrupt change in company cash-flow prospects. Different exposures to disrupted revenues, refinancing needs or liquidity could change the relative performance of winners and losers.

I would check the sector and company contributions before assuming that the short leg would offset losses on the long leg. Distinguishing changed business exposures from selling pressure would require company information, holdings and flow data that the aggregate factor does not provide. These are possible explanations to examine, not findings from the model.
'''
    after=f'''**Subsequent outcome: {future.index[0]:%d %B}–{end:%d %B %Y}.** The market returned {row.market:+.2f}%. Momentum returned {row.net:+.2f} pp (published rounded factor: {ev.y:+.2f} pp).

| Contribution, pp | Conditional estimate using the subsequently realised market move | Realised | Realised minus estimate |
| --- | ---: | ---: | ---: |
| Long | {fit['long']:+.2f} | {row.long:+.2f} | {row.long-fit['long']:+.2f} |
| Short | {fit['short']:+.2f} | {row.short:+.2f} | {row.short-fit['short']:+.2f} |
| Net | {sum(fit.values()):+.2f} | {row.net:+.2f} | {row.net-sum(fit.values()):+.2f} |

This table uses the realised market move only to evaluate the conditional relationship; that move was not known at the cutoff.

'''
    if index==0:
        after+='''The short leg lost more than the long leg gained, consistent with the direction suggested by the trailing betas. The regression nevertheless understated the net loss substantially, mainly through the short leg. The 29.3% weekly loss probability was already above the illustrative 20% threshold before the rebound, following earlier volatility. That does not establish that the model identified a turning point or the source of the rally.
'''
    else:
        after+='''The fitted response had the wrong net sign. Gains on the short leg did not offset the long-leg loss as much as the regression implied. The 14.8% probability was below the illustrative 20% warning level before this loss; it rose to 16.3% at the next cutoff. A 14.8% estimate allows losses to occur, so this observation alone cannot establish miscalibration. It does show a missed threshold warning and a large conditional-response error. The beta relationship was insufficient to explain the realised leg contributions.
'''
    lo=t-pd.Timedelta(weeks=6);hi=end+pd.Timedelta(weeks=3)
    d=e.daily.loc[lo:hi];cumulative=d[['long','short','net']].cumsum()
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for name,color in [('long','#236b8e'),('short','#be653d'),('net','#243342')]:
        axes[0].plot(cumulative.index,cumulative[name],label=name.title(),color=color)
    axes[0].set(ylabel='Cumulative contribution (pp)',title='Returns summed from the first plotted day');axes[0].legend()
    x=e.core.evaluation;w=x.loc[x.cutoff.between(d.index.min(),d.index.max())]
    axes[1].step(w.cutoff,100*w.p_fhs5y_ewma,where='post',label='Historical-return model')
    axes[1].step(w.cutoff,100*w.p_logit_ewma,where='post',label='Same-volatility logit',alpha=.7)
    axes[1].axhline(20,color='grey',ls=':',label='Illustrative 20% warning level')
    axes[1].set(ylabel='Next-week loss probability (%)',xlabel='Date; probability plotted at its information cutoff');axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(t,color='black',ls='--',lw=1)
        ax.axvspan(t,end,color='grey',alpha=.12);ax.grid(alpha=.15)
    fig.suptitle(title+' | dashed line: assessment; shading: subsequent five trading days')
    return before,fig,after

def insert(cells):
    heading=nb.v4.new_markdown_cell('''## Historical cases: exposure, macro context and subsequent returns

I use two selected cases to examine how the exposure estimates relate to the economic
backdrop and where they miss the outcome. The sources below were published before
each cutoff. I selected and read them retrospectively; these paragraphs are not records
of forecasts I made at the time. The figures include later returns, separated by the
assessment marker. The earlier similar-loss comparison remains a separate historical
description of which leg contributed to the loss.''')
    code=nb.v4.new_code_cell('''from macro_cases import case
for case_index in [0,1]:
    before,figure,after=case(E,P,case_index)
    display(Markdown(before))
    show_inline(figure)
    display(Markdown(after))''')
    i=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Company-level data:'))
    cells[i:i]=[heading,code]
    return cells
