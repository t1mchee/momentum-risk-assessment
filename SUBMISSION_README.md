# Equity momentum risk — reading and reproduction

## Main documents and optional evidence

1. **Main assessment:** `09_economic_vulnerability.html` — conclusion, exposure and
   units, baseline construction and validation, economic interpretation, worked
   historical assessment, limitations and production/text-AI path.
2. **Compact research appendix:** `09_research_appendix.html` — results from twenty-one
   experiment sections. The appendix links the reported findings to methods and
   selected evidence.
3. **Optional supplementary archive:** `SUPPLEMENTARY_RESEARCH_ARCHIVE.html` — all
   saved executed figures, full tables and expandable calculation code. About 9 MB;
   explicitly not required reading. Provide separately if detailed verification is wanted.

The main HTML embeds its figures and tables; the compact appendix is intentionally
lightweight. Earlier research notebooks are not
required to read the evidence. The notebook equivalents have the same basenames
with `.ipynb` extensions. The appendix is a documentation notebook with expandable
calculation code and embedded saved results; it does not rerun model searches.

## What is and is not implemented

Two historical cases connect the saved estimates to contemporaneous macro context:
March 2009 rebound exposure and a March 2020 missed threshold warning. Each separates
information through the cutoff from subsequent outcomes and shows errors by leg.
They are selected retrospective illustrations, reproduced by `scripts/macro_cases.py`.

The statistical estimates and research comparisons are implemented. Their worked
assessment is historical, not live. The main analysis uses data through December
2022, with a separately labelled frozen-specification recent evaluation through July
2026 (not an untouched holdout). Run `scripts/recent_validation.py` to reproduce it;
the protocol and results are under `research/recent_validation` and
`research/RECENT_VALIDATION_PROTOCOL.md`. Recent path probabilities overpredict
breaches; the extension does not validate a new signal. A separately labelled 2026 ETF-options snapshot is a pricing/data-quality
example, not an input to the historical forecast. The public factor, reconstructed
constituent proxy and long-only ETF are different exposures.

One dated issuer-text example demonstrates retrospective corporate-action verification
linked to a numerical correction. It is manually checked, not an automated catalyst
model or historical text-pipeline validation. Broader text evidence, authenticated production feeds and human review form a proposed
production workflow, not a claim of an already integrated AI prediction service.
The purpose is risk investigation for a discretionary PM, not a trading strategy.

## Reproduce calculations

The path section displays historical simulation and GARCH separately. Reproduce the
GARCH experiment with `scripts/garch_path_experiment.py`, then prepare its dated
comparison with `scripts/prepare_path_comparison.py` (requires `arch` and a Parquet
engine; tested using `.venv/bin/python`). These write to `research/garch_path`.
The main notebook reads the saved results; it does not refit GARCH when opened.

Executing the main notebook requires the accompanying `scripts/` helpers and prepared
`research/` inputs, plus Python dependencies in `requirements.txt`. It runs from the
project root without downloading new data. Rebuilding original research calculations
has additional source-data requirements detailed in `09_ECONOMIC_GUIDE.md`.
The notebook file by itself is not a standalone software package. For a clean
installation and the distinction between displaying saved research and refitting
models, follow [the execution guide](09_ECONOMIC_GUIDE.md).

Packaged files and hashes are recorded in `FILE_MANIFEST.json`. The appendix embeds saved evidence; its build QA
does not imply that every historical fit was rerun. Model evaluation is chronological
but exploratory after model search, not an untouched final holdout. Provider
redistribution permissions must be checked before sharing underlying data archives.
