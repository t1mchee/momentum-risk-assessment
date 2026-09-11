"""Build the dated GARCH/path comparison without changing any selected forecasts."""
import json,hashlib
import numpy as np
import pandas as pd
from arch import arch_model
from garch_path_experiment import ROOT,OUT,simulate
from distribution_anatomy import french

def main():
    snapshot=json.loads((ROOT/'research/local_stress/latest_path_assessment.json').read_text())[0]
    date=pd.Timestamp(snapshot['date'])
    record=next(r for r in json.loads((OUT/'fits.json').read_text()) if r['year']==date.year and r['model']=='garch_t')
    legs=french('portfolios.zip').loc['1964':date]
    daily=(legs['SMALL HiPRIOR']+legs['BIG HiPRIOR']-legs['SMALL LoPRIOR']-legs['BIG LoPRIOR'])/2
    cutoff=pd.Timestamp(record['fit_end']);past=daily.loc[:cutoff]
    params=(record['omega'],record['alpha'],record['beta'],0.,record['nu'])
    fixed=arch_model(past,mean='Zero',vol='GARCH',p=1,q=1,dist='t',rescale=False).fix([params[0],params[1],params[2],params[4]])
    omega,alpha,beta,_,nu=params;r=past.iloc[-1]
    h=omega+alpha*r*r+beta*fixed.conditional_volatility.iloc[-1]**2
    predictions=pd.read_parquet(OUT/'predictions.parquet');checks=0
    for t,r in daily.loc[daily.index>cutoff].items():
        h=max(omega+alpha*r*r+beta*h,1e-12)
        if t in predictions.index:
            np.testing.assert_allclose(h,predictions.loc[t,'initial_variance_garch_t'],rtol=1e-9);checks+=1
    p,ep=simulate(h,params,date)
    rows=[dict(model='Historical paths',date=str(date.date()),path_probability=snapshot['path_breach_probability'],endpoint_probability=snapshot['terminal_breach_probability']),dict(model='GARCH-t simulated paths',date=str(date.date()),path_probability=p,endpoint_probability=ep)]
    (OUT/'assessment_comparison.json').write_text(json.dumps(rows,indent=2))
    (OUT/'comparison_QA.json').write_text(json.dumps(dict(status='PASS',date=str(date.date()),fit_end=str(cutoff.date()),annual_variance_replays=checks,endpoint_below_path=ep<=p,method='Saved annual parameters; observations through assessment cutoff only',predictions_sha256=hashlib.sha256((OUT/'predictions.parquet').read_bytes()).hexdigest()),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
