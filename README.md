# Trump tariff threats and the US Treasury yield curve

Event study of US Treasury yields around the tariff announcements of the two Trump administrations
(2017--2020 and 2025). Python lab of the course *Advanced Fixed Income and Credit* (EDHEC, MSc in
Financial Engineering), Part 2 of a group assignment.

For each event date the notebook measures the change of the constant-maturity yields (3M to 30Y) over
$[-30,+30]$ trading days and compares it with the average over all non-event days (the *abnormal yield
change*). It then looks at level, slope and curvature, at the reaction by targeted country, at the first
versus the second mandate, and at the limits of the method (alternative benchmarks, declustered events,
principal components, placebo inference).

## Structure

```
module_1/pylab_trump_tariff_threats_assignment.ipynb   completed and executed notebook (Q1.1 - Q2.1)
module_1/figures/                                      figures produced by the notebook (PNG and PDF)
functions_library/module_1/event_study.py              event-study functions: alignment, abnormal changes,
                                                       placebo test, curve components, PCA, plots
functions_library/module_1/fred.py                     course script to download the data from FRED
data/fred/yield_curve_snapshot.csv                     FRED yields used (2015-01-02 to 2026-10-21)
make_tables.py                                         generates the LaTeX tables of the report from the same code
```

## Run

```bash
pip install polars numpy scipy matplotlib plotly requests nbformat jupyter
jupyter notebook module_1/pylab_trump_tariff_threats_assignment.ipynb     # run it from the repository
python make_tables.py [output_folder]                                      # optional, default: tables/
```

Keep the folder structure: the notebook imports `functions_library` and reads the data from `data/fred`.
No API key is needed, since the notebook uses the saved snapshot (`REFRESH_DATA = False`). To download the
data again, get a free key at <https://fred.stlouisfed.org/docs/api/api_key.html>, save it in
`data/fred/api_key.py` as `FRED_API_KEY = "<your key>"` (this file is git-ignored) and set
`REFRESH_DATA = True`. The placebo draws use a fixed seed, so results are identical across runs.

## Data

Daily Treasury constant-maturity yields (series `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2`, `DGS3`, `DGS5`, `DGS7`,
`DGS10`, `DGS20`, `DGS30`), H.15 release of the Board of Governors of the Federal Reserve System, retrieved
from FRED (Federal Reserve Bank of St. Louis). The event dates come from the course notebook.

## Use of AI tools

The code and the notebook were written with the help of an AI assistant (Claude, Anthropic), as declared in
the report that goes with this assignment.
