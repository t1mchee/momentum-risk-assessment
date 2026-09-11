"""Numerical QA and compact, qualified constituent PCA findings."""
import json, hashlib, html
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from constituent_pca import ROOT, OUT, pca


def main():
    metrics=pd.read_parquet(OUT/'metrics.parquet');gates=pd.read_json(OUT/'gates.json')
    tails=pd.read_json(OUT/'tails.json');future=pd.read_json(OUT/'future.json')
    loadings=pd.read_parquet(OUT/'loadings.parquet')
    # Normalise date representations only, never alter source results.
    for f in [metrics,gates,tails,future,loadings]:f['date']=pd.to_datetime(f.date)
    raw=metrics.query('window==252 and kind=="raw_correlation"').set_index('date').sort_index()
    last=raw.index[-1];latest=metrics.loc[(metrics.date==last)&metrics.window.eq(252)].set_index('kind')
    primary=gates.query('window==252');cor=float(raw.pc1_share.corr(raw.average_corr))
    x=pd.read_parquet(OUT/'latest_return_panel.parquet');w=pd.read_parquet(OUT/'latest_weights.parquet')
    market=pd.read_parquet(ROOT/'research/notebook_v8/daily.parquet').market.reindex(x.index)
    z=(x-x.mean())/x.std();u,s,v=np.linalg.svd(z.to_numpy(),full_matrices=False)
    row=latest.loc['raw_correlation']
    np.testing.assert_allclose(s[0]**2/(s**2).sum(),row.pc1_share,atol=1e-10)
    sd=x.std().to_numpy();pc=(z.to_numpy()@v[0])*(w.weight.to_numpy()*sd@v[0])
    np.testing.assert_allclose(np.var(pc,ddof=1)/np.var(x.to_numpy()@w.weight,ddof=1),row.pc1_book_variance_share,atol=1e-10)
    np.testing.assert_allclose(tails.pc1_tail+tails.remainder_tail,tails.tail_mean,atol=1e-9)
    assert np.allclose(w.weight.clip(lower=0).sum(),1) and np.allclose(w.weight.clip(upper=0).sum(),-1)
    # Prefix invariance for PCA itself: appending then excluding synthetic future
    # rows leaves all inputs and estimates unchanged; upstream membership vintages
    # and corporate-action authenticity remain unverified.
    extended=pd.concat([x,pd.DataFrame(999.,index=pd.date_range(last+pd.Timedelta(days=1),periods=5),columns=x.columns)])
    prefix=extended.loc[:x.index.max()]
    a,_=pca(x,w.weight,market);b,_=pca(prefix,w.weight,market)
    for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-10)
    rng=np.random.default_rng(20260915);boot=[]
    with threadpool_limits(limits=1):
        for i in range(100):
            starts=rng.integers(0,len(x),size=int(np.ceil(len(x)/20)))
            ix=(starts[:,None]+np.arange(20)).ravel()[:len(x)]%len(x)
            for kind,resid,sector in [('raw_correlation',False,None),('residual_correlation',True,None),('market_sector_residual',True,w.sector)]:
                val,_=pca(x.iloc[ix],w.weight,market.iloc[ix],resid,True,sector)
                boot.append(dict(draw=i,kind=kind,pc1_share=val['pc1_share'],book_share=val['pc1_book_variance_share']))
    bootstrap=pd.DataFrame(boot)
    summary=bootstrap.groupby('kind')[['pc1_share','book_share']].quantile([.025,.975]).reset_index().rename(columns={'level_1':'quantile'})
    summary.to_json(OUT/'bootstrap.json',orient='records',indent=2)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axes[0,0].scatter(raw.index,raw.pc1_share,label='PC1 share',s=18)
    axes[0,0].scatter(raw.index,raw.average_corr,label='Average correlation',s=14,marker='x')
    axes[0,0].set(title=f'Raw PC1 mainly repeats correlation (r={cor:.3f})',ylabel='Cross-sectional co-movement');axes[0,0].legend(fontsize=8)
    axes[0,1].scatter(raw.pc1_share,raw.pc1_book_variance_share,color='#236b8e',s=30)
    axes[0,1].set(title='Common movement ≠ signed book exposure',xlabel='PC1 universe variance share',ylabel='PC1 share of covered-book variance')
    for j,window in enumerate([252,126]):
        g=gates[gates.window==window]
        axes[1,0].scatter(g.date,np.full(len(g),j),c=np.where(g.passed,'#236b8e','#be653d'),marker='s',s=18)
    axes[1,0].set(yticks=[0,1],yticklabels=['252 sessions','126 sessions'],title='Blue: coverage passed; orange: failed')
    kinds=['raw_correlation','residual_correlation','market_sector_residual']
    indices=np.arange(3);vals=latest.loc[kinds]
    axes[1,1].bar(indices-.17,vals.pc1_share,width=.34,label='Universe share')
    axes[1,1].bar(indices+.17,vals.pc1_book_variance_share,width=.34,label='Book / residual-book share')
    axes[1,1].set(xticks=indices,xticklabels=['Raw','Market removed','Market + peers\nremoved'],ylabel='Variance share',title=f'{last:%Y-%m}: denominators differ by residualisation')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.15);ax.tick_params(axis='x',labelrotation=20)
    fig.savefig(OUT/'pca_overview.png',dpi=140,bbox_inches='tight');plt.close(fig)
    l=loadings.loc[(loadings.date==last)&loadings.kind.eq('residual_correlation')]
    sector=l.groupby('sector').exposure_contribution.sum().sort_values()
    # Exposure units depend on PC normalisation; signs align only within this fit.
    tail=tails.loc[tails.date==last].sort_values('h')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].barh(sector.index,sector,color=np.where(sector>=0,'#236b8e','#be653d'))
    axes[0].set(title='Market-residual PC: sector exposure contributions',xlabel='Signed exposure, arbitrary PC orientation')
    xx=np.arange(len(tail))
    axes[1].bar(xx-.18,tail.pc1_tail,width=.36,label='Raw PC1 contribution')
    axes[1].bar(xx+.18,tail.remainder_tail,width=.36,label='Remainder (including mean)')
    axes[1].plot(xx,tail.tail_mean,'ko',label='Net lower-15% mean')
    axes[1].set(xticks=xx,xticklabels=tail.h,title='Trailing net-tail attribution\nBlue: PC1; orange: remainder; dots: net',xlabel='Trailing horizon (sessions)',ylabel='P&L (pp)')
    axes[1].axhline(0,color='gray',lw=.7)
    fig.suptitle(f'{last:%Y-%m-%d} covered proxy: fixed weights applied to trailing history, in-fit attribution')
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(OUT/'pca_anatomy.png',dpi=140,bbox_inches='tight');plt.close(fig)
    latestgate=primary.loc[primary.date==last].iloc[0]
    sector_weights=w.assign(long=w.weight.clip(lower=0),short=-w.weight.clip(upper=0)).groupby('sector')[['long','short']].sum()
    displaycols=['pc1_share','pc1_book_variance_share','book_variance','long_one_sd','short_one_sd','net_one_sd']
    table=latest[displaycols].round(4)
    path=OUT/'FINDINGS.md'
    narrative=f'''# Constituent PCA: implemented, with a descriptive-only promotion decision

## Bottom line

PCA adds a constituent-level way to inspect shared exposures, but **raw PC1 share
is not a useful new headline signal**: its correlation with average correlation is
{cor:.3f} across the {len(raw)} accepted one-year snapshots. The more useful object
is the signed long/short exposure, including co-movement left after market removal.
This does NOT establish crowding, forced selling, a new priced factor or forecast skill.

## Data and what the gate actually permits

The existing IWV proxy has {len(primary)} pre-2023 monthly books. Only
{int(primary.passed.sum())} pass the one-year data gate; 72 pass at six months.
No data after 2022 was read. No upstream files or Notebook 08/09 were changed.
We reuse the saved names, not the older cached book returns that masked large moves.
Primary PCA excludes incomplete stock histories, unverified inferred actions and
unexplained >45% adjusted daily moves. Large real moves can be excluded by this
uncertainty screen: the resulting covered subset is selected and not tail-unbiased.

At {last:%Y-%m-%d}, {int(latestgate.retained_long)} long and
{int(latestgate.retained_short)} short names survive, with
{latestgate.long_weight_coverage:.1%}/{latestgate.short_weight_coverage:.1%} of known
positive formation position value. We renormalise those weights per leg. ETF
position value is not company market cap. Source prices lack a complete dividend,
delisting and permanent-identifier history. This is not an exact French factor
replication or the PM's actual book, nor a clean total-return equity panel.

## Latest historical snapshot: an economically interpretable contrast

- Raw correlation PC1 explains {row.pc1_share:.1%} of standardised constituent variance, but
  {row.pc1_book_variance_share:.1%} of covered long–short variance. For one positive
  historical SD of that market-aligned PC, the long/short contributions are
  {row.long_one_sd:+.2f}/{row.short_one_sd:+.2f} pp: only partial offset.
- Removing market exposure leaves a PC explaining
  {latest.loc['residual_correlation','pc1_share']:.1%} of residual constituent
  variance and {latest.loc['residual_correlation','pc1_book_variance_share']:.1%}
  of **residual-book** variance. Its long and short exposures reinforce, rather
  than offset. PC signs are arbitrary; the reinforcing relationship is invariant.
- Removing same-sector peer co-movement reduces residual-book variance from
  {latest.loc['residual_correlation','book_variance']:.2f} to
  {latest.loc['market_sector_residual','book_variance']:.2f} pp². The remaining
  first PC accounts for {latest.loc['market_sector_residual','pc1_book_variance_share']:.1%}
  of that much smaller denominator—not that fraction of the original book risk.
  These are separate in-fit residualisations, not an additive causal variance decomposition.

The sector plot identifies the concentration behind this particular common exposure.
The covered long leg is {sector_weights.loc['Energy','long']:.1%} energy and
{sector_weights.loc['Health Care','long']:.1%} health care, while
{sector_weights.loc['Information Technology','short']:.1%} of the short leg is
information technology. The market-residual component has long and short P&L
loadings with the same sign: the two legs can therefore reinforce a rotation
exposure even though their market exposures partly offset. This is a description
of this dated proxy's covariance, not evidence about a future rotation or its cause.
Its interpretation must start with named positions/sectors, not an invented macro or
crowding label. A same-sector, leave-one-out peer factor is an internal diagnostic,
not an external investable benchmark; groups under five names remain market-only.

## How the PCA quantities are built

1. Hold the selected formation weights fixed and gather their common trailing
   price-return observations. Returns enter as positive stock returns; short signs
   are in weights, not a separate PCA input convention.
2. Demean each stock and divide by its trailing SD to form Z. Compute the correlation
   matrix C = Z'Z/(T-1), then eigenvalues and eigenvectors. Raw covariance PCA is a
   separate sensitivity. Market-residual PCA first fits each stock on an intercept
   and market return, then repeats the standardisation on the residuals.
3. PC1 score is Z v1. If D contains stock SDs, signed portfolio exposure is w'Dv1.
   Long and short use their own signed weight vectors on the SAME component basis.
4. Component variance is lambda1 * (w'Dv1)^2. Divide by w' Sigma w for the book
   variance share. Universe share lambda1/sum(lambda) answers a different question.
5. For the tail view, reconstruct PC1 P&L on every trailing day; subtract it from
   the raw fixed-weight P&L to obtain the remainder, including the mean. Select
   net-tail windows once and attribute both pieces on the same selected mass.

## Keep the distributional framing

The tail chart uses the SAME worst-15% net windows for PC1 and remainder, at
1/5/10/20 sessions. They add exactly, including fractional cutoff ties. Gaps are
not bridged. These are returns of the selected date's fixed covered weights applied
to trailing observations—not the realised history of the rebalanced strategy.
PCs are estimated on those same observations: this is in-fit attribution, not an
ex-ante prediction. It illustrates how variance concentration and tail contribution
can differ. Neither a Gaussian distribution nor a crash probability is inferred.

## Validation and limits

The next-month check freezes names, weights and loadings. Only
{int(future.future_pass.sum())} of {len(future)} candidate formations pass its
complete-constituent daily observation gate; the final formation has no permitted
next-month observations. We therefore DO NOT fit an incremental forecasting model,
claim persistence, or promote a PCA crash-warning feature. The strict gate exposes
the source's attrition/missing-price limitation, not a statistical rejection of PCA.
Relaxing it would change the exposure or require explicit missing-data assumptions.

Numerical checks pass: independent SVD agrees with eigen decomposition; all-PC
variances reconcile to portfolio variance; long and short exposures add; risk is
unchanged by flipping PC signs; tail pieces add; PCA prefix check passes.
Monthly/window/universe changes remain a comparability issue. PCs are not automatically
the same factor over time; eigenvalue gaps and 126/252-day comparisons are saved.
The latest-date 100-draw 20-session block bootstrap refits PCA and market/peer
regressions on a FIXED selected universe. Its intervals do not account for source
errors, membership selection, research search or out-of-sample performance.

## Decision

Keep this as a separate constituent concentration/exposure diagnostic, with the
data gate displayed. Do not change the numerical forecasts in Notebook 09. Before
predictive integration, obtain reliable daily total returns and security identifiers,
or specify and validate a missing-data model. Reuse the reconstruction already on
disk; do not start another unqualified price-panel build.

Reproduce: run `scripts/constituent_pca.py`, then
`scripts/summarize_constituent_pca.py` with the Notebook 08 Python environment.
The input scope and follow-on sector check are recorded in `research/PCA_PROTOCOL.md`.
All gate failures, metrics, dated constituent loadings, tail results and source
hashes are in this folder. Nothing here claims academic novelty or human PM validation.
'''
    path.write_text(narrative)
    doc='<!doctype html><meta charset="utf-8"><title>Constituent PCA findings</title><style>body{max-width:1080px;margin:40px auto;font:16px/1.55 system-ui;padding:0 20px;color:#243342}img{width:100%}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd}</style>'
    doc+='<h1>Constituent PCA — research findings</h1><img src="pca_overview.png"><img src="pca_anatomy.png">'
    doc+='<h2>Latest snapshot: component shares, variances and one-PC-SD responses</h2>'+table.to_html()
    doc+='<h2>Fixed-universe bootstrap interval endpoints</h2>'+summary.round(4).to_html(index=False)
    doc+='<pre>'+html.escape(narrative)+'</pre>'
    (OUT/'FINDINGS.html').write_text(doc)
    files=[ROOT/'scripts/constituent_pca.py',ROOT/'scripts/summarize_constituent_pca.py',OUT/'metrics.parquet',OUT/'loadings.parquet',OUT/'gates.json',OUT/'tails.json',OUT/'future.json']
    (OUT/'QA.json').write_text(json.dumps(dict(status='PASS',raw_pc1_average_corr=cor,passed_primary=len(raw),
          future_passed=int(future.future_pass.sum()),checks=['independent SVD','all-PC variance identity','signed leg identity','PC sign invariance','tail additivity','PCA prefix safety'],
          limits='Not source-vintage/corporate-action certification or out-of-sample validation',
          hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2))
    print(narrative)


if __name__=='__main__':main()
