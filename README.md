# US equity momentum risk — Tim Chee

This is the classical statistical implementation and evaluation accompanying the
previously reviewed AI-system design.

## Read

1. [Main notebook](09_economic_vulnerability.ipynb): assessment, calculations, validation and extensions.
2. [Research appendix](09_research_appendix.ipynb): methods and findings from all 21 experiment sections.
3. [Supplementary archive](SUPPLEMENTARY_RESEARCH_ARCHIVE.ipynb): optional saved figures, tables and calculation
   extracts. Development narratives are omitted; methods and limitations are in the appendix.

The corresponding HTML files are included and open locally without Python. GitHub
can preview the notebooks; download or clone the repository to use the HTML copies.

## Run the main notebook

From this folder, with Python 3.12:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python scripts/execute_economic_notebook.py
```

The tested environment was macOS with Python 3.12. On Windows, substitute
`.venv\Scripts\python.exe` and use `requirements.txt` if a pinned package is
platform-specific. This executes all main notebook
cells and exports the HTML. Alternatively, open the notebook in Jupyter or Cursor
using that environment and run its cells from this folder. Optional live selectors
require `ipywidgets`; the default installation provides the static fallback.

The prepared inputs and runtime helpers are included, so notebook execution needs
no downloads. It was tested from this folder in a fresh Python 3.12 environment
installed from the declared dependencies. `requirements-tested.txt` records the
versions used in that test; `requirements.txt` contains the broader dependency ranges.

## Reproduction scope

Execution recalculates the dated displays and reads saved predictions and evaluation
tables. It does not refit every experiment. Source code for the principal calculations
and additional calculation extracts are included. The recent-factor and GARCH
experiments use the included cached public factor files. Stock and options preparation
needs additional source archives; paths marked `SOURCE_DATA_ROOT` refer to those
archives, which are not bundled. The stock-membership lineage limitation remains
unresolved. See `09_ECONOMIC_GUIDE.md` for details.

The archived figures and tables retain the original numerical evidence. Runtime
source has been separated from notebook-drafting code and unused legacy displays.
Packaged source hashes reflect those source-only changes; factor returns and saved
forecast rows are unchanged. `FILE_MANIFEST.json` lists the packaged file hashes.

AI assistance with implementation, testing and drafting is disclosed in the main
notebook. The estimates use classical statistical methods, without text-derived
forecast inputs. The worked output is historical, not a live service.

Source attribution and data limitations are retained in the documents. Underlying
source data remain subject to their providers' terms.
