"""Dated assessment text; numerical inputs are calculated by the companion."""
def assessment(view,date):
    from distribution_panel import profile
    s=view.e.state(date);_,_,risk=view.e.core.core(date)
    tail=profile(view.e.core.sample(date,5,252).to_numpy(),5)
    up=view.rolling(date,2);down=view.rolling(date,-2)
    interpretation=''
    if str(s.date.date())=='2022-12-30':
        interpretation='''
At this date, I would first examine the short leg: it accounts for most of the
historical tail loss, and the negative market beta suggests vulnerability to a
market rebound. I would check whether the actual portfolio shares that exposure;
different holdings could change this priority. These estimates do not establish
that a rebound is imminent.

The probability mainly reflects recent volatility. The scenarios are average
historical responses, not stress-loss limits. In the two crisis examples below,
net scenario errors are roughly 6–8 pp, including one wrong-sign estimate.
'''
    return f'''## Example output: {s.date:%d %B %Y}

These calculations use the public US momentum factor and its long and short legs.
Returns are percentage points (pp) on the fixed-notional factor position.

| Calculation | Result |
| --- | --- |
| Estimated probability of losing more than 2.16 pp over approximately the next week | {risk['probability']:.1%} |
| Estimated fifth-percentile weekly outcome | {risk['q05_pp']:+.2f} pp |
| Average of the worst 15% of historical five-trading-day windows | {tail['tail_mean']:+.2f} pp: long {tail['long']:+.2f}, short {tail['short']:+.2f} |
| Conditional response if the market rises 2% | Long {up['long']:+.2f}, short {up['short']:+.2f}, net {sum(up.values()):+.2f} pp |
| Conditional response if the market falls 2% | Long {down['long']:+.2f}, short {down['short']:+.2f}, net {sum(down.values()):+.2f} pp |

The historical loss calculation uses 252 completed five-trading-day windows.
The scenario entries are estimated average responses to the stated market moves.
The trailing net market beta is {s.beta_net:+.2f}. Applying these estimates to another
portfolio would require its holdings. Forecast and scenario errors are reported below.
{interpretation}
'''
