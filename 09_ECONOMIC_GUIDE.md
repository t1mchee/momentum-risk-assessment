# Execution and source-data requirements

The main notebook contains the core assessment and its validation, followed by
extensions. The compact appendix records 21 experiment sections. The optional
supplementary archive preserves the full saved evidence. None requires reading the
earlier development notebooks.

## Execute the submission from prepared results

Use Python 3.11 or 3.12. From the supplied project directory:

```sh
python3 -m venv .venv-submission
.venv-submission/bin/python -m pip install -r requirements.txt
.venv-submission/bin/python scripts/execute_economic_notebook.py
.venv-submission/bin/python scripts/execute_economic_notebook.py --static
```

On Windows, use `.venv-submission\Scripts\python.exe` instead. The first execution
updates the main notebook and HTML. The second tests the optional-widget fallback
without overwriting the delivered notebook. There should be ten figures and no
execution errors. The supplied scripts and prepared `research/` files are required;
the `.ipynb` alone is not an executable package. No source data are downloaded by
this notebook execution. A Parquet engine is included in `requirements.txt`.

The notebook calculates displays and reads saved model predictions and scores.
Executing it does **not** refit every research model or independently authenticate
the original source data. Recent-period tables read `research/recent_validation/`;
their calculation is in `scripts/recent_validation.py`.

## Rebuild the original research

This is a separate operation with additional input requirements. The experiment
sections in `09_research_appendix` identify their scripts and saved evidence.
`FILE_MANIFEST.json` records packaged file hashes. In particular:

- Public-factor and scenario calculations require the cached daily French factor
  and leg histories. `scripts/economic_extension.py` and
  `scripts/economic_macro_scenarios.py` rebuild those extensions; they can download
  FRED histories if the relevant cache is absent.
- Stock reconstruction requires the original IWV holdings, prices and membership
  archive. Paths marked `SOURCE_DATA_ROOT` must be configured to the location of
  those external archives. Those archives are not required to execute
  the submission from prepared results. The membership cache's generating code
  version remains unverified, so this is not a fully reproducible source-to-portfolio
  construction.
- Options preparation requires the original MTUM quote snapshots and a configured
  input path. The notebook reads the prepared audit and figure instead.
- `scripts/garch_path_experiment.py` refits the path experiment;
  `scripts/prepare_path_comparison.py` prepares the dated comparison. This is
  substantially more work than displaying their saved results.

Source histories are current-vintage, not authenticated historical publication
archives. Macro lags are assumptions, not a substitute for release-vintage data.
Check providers' redistribution terms before sharing underlying archives; public
download access does not establish redistribution permission.

## Date controls

The historical selector updates the dated core assessment, scenario curve and
distribution plots, not the evaluation results. The separate path comparison stays
at its labelled date. Optional live widgets require `ipywidgets` and a compatible
Jupyter frontend. Without widgets, change `DATE` and rerun the relevant cells.
The saved HTML contains static outputs, not a live Python service.
