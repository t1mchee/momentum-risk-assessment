"""Small, explicitly separate constituent slot for the economic companion."""
from pathlib import Path
import json,hashlib


class BookDiagnostics:
    def __init__(self,root):
        self.root=Path(root);folder=self.root/'research/book_level'
        qa=json.loads((folder/'QA.json').read_text())
        for name,value in qa['hashes'].items():
            assert hashlib.sha256((self.root/name).read_bytes()).hexdigest()==value,f'Stale book diagnostics: {name}'
        self.snapshots={r['date']:r for r in json.loads((folder/'snapshots.json').read_text())}

    def card(self,date):
        date=str(date)[:10];r=self.snapshots.get(date)
        title='### Reconstructed IWV portfolio: dated calculations'
        separation='These calculations use approximate winner/loser portfolios reconstructed from IWV holdings. Their constituents differ from those of the public momentum factor.'
        if r is None or not r['gate']['passed']:
            reason=r['gate']['reason'] if r else 'no approved formation at this date'
            return f'{title}\n\n{separation}\n\n**Unavailable for {date}:** {reason}. No stale book is carried forward and no missing constituent return is treated as zero.'
        g=r['gate'];t=next(x for x in r['tails'] if x['h']==5)
        raw=next(x for x in r['offsets'] if x['k']==3 and not x['market_removed'])
        residual=next(x for x in r['offsets'] if x['k']==3 and x['market_removed'])
        b=r.get('bridge')
        if b and b['passed']:
            change=(f"Using stocks with adequate data at both dates, estimated variance changed **{b['total_change']:+.2f} pp²** from {b['previous']}: membership/weight updates contributed **{b['weight_effect']:+.2f}**, and covariance changes contributed **{b['covariance_effect']:+.2f}**.")
        else:
            change='Rebalance attribution unavailable: '+(b['reason'] if b else 'no consecutive accepted preceding formation')+'.'
        names=', '.join(t['top_names'])
        om=t['omit_episode']
        omission=(f"After removing the largest connected episode, the top-five share recomputed across all names is **{om['top5_share']:.1%}**." if om and om['top5_share'] is not None else 'Largest-episode omission leaves insufficient adverse mass for a concentration comparison.')
        boundary=' After removing the market, the third and fourth components are close in strength, so the selected three-component set may be unstable.' if residual['boundary_gap']<.1 else ''
        return f'''{title}

{separation}

**Portfolio date {date}; price data through {g['price_asof']}.** The data cover {g['long_weight_coverage']:.1%} of known long-leg value and {g['short_weight_coverage']:.1%} of known short-leg value. Dividends, delistings and unresolved corporate actions limit the reconstruction.

1. **What changed?** {change}
2. **Stock contributions.** In the selected five-trading-day loss windows, the five largest negative contributors ({names}) account for **{t['top5_share']:.1%} of the sum of negative average stock contributions**. {omission} The denominator excludes positive contributions, so this percentage differs from a share of net loss.
3. **Do the legs offset?** Across the first three principal components, covariance between the legs reduces their combined variance by **{raw['offset']:+.0%}** relative to the sum of the two leg variances. Net component SD is **{raw['net_component_sd']:.2f} pp/day**. After removing market returns, the reduction is **{residual['offset']:+.0%}**, and SD is **{residual['net_component_sd']:.2f} pp/day**. A negative reduction means the covariance increases risk.{boundary}

I would compare these stock and sector exposures with actual holdings before applying the result to another portfolio. The research files contain the 1-, 5-, 10- and 20-trading-day calculations and omission checks.
'''
