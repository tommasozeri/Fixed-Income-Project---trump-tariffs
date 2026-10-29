"""Generate the LaTeX tables and numbers of the Part 2 report.

It uses the same functions (functions_library/module_1/event_study.py), the same data
(data/fred/yield_curve_snapshot.csv) and the same event lists (read from the notebook) as the
notebook, so the numbers in the report are identical to the notebook output.

Usage, from the FIN2764 folder:   python make_tables.py [output_folder]     (default: tables/)
"""
import sys, os, json, ast
sys.path.insert(0, os.getcwd())
import numpy as np, polars as pl, nbformat
from functions_library.module_1.event_study import *

OUT = sys.argv[1] if len(sys.argv) > 1 else "tables"
os.makedirs(OUT, exist_ok=True)

# ---- same inputs as the notebook -------------------------------------------
raw = pl.read_csv('data/fred/yield_curve_snapshot.csv', try_parse_dates=True)
yields = prepare_yields(raw); dates = yields['date']
nb = nbformat.read('module_1/pylab_trump_tariff_threats_assignment.ipynb', as_version=4)
src = next(c.source for c in nb.cells if c.get('id') == 'timeline-setup')
ns = {}; exec(src, ns)
actions = build_event_table(ns['trump_actions'], 'Trump action', dates)
responses = build_event_table(ns['country_responses'], 'Country response', dates)
all_idx = np.union1d(actions['idx'].to_numpy(), responses['idx'].to_numpy())
COLS = MATURITIES + COMPONENTS
END_DATE = '2026-10-22'

res_a = event_study(yields, actions['idx'], COLS, exclude_idx=all_idx)
res_r = event_study(yields, responses['idx'], COLS, exclude_idx=all_idx)

def stars(p): return '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.10 else ''
def num(x, p=None, d=1):
    s = f"{x:+.{d}f}".replace('-', '$-$').replace('+', '')
    if p is not None and stars(p):
        s = s.replace('$-$', '-')  # keep math mode together
        return f"${'-' if x < 0 else ''}{abs(x):.{d}f}{stars(p)}$"
    return s

def write(name, body):
    with open(f"{OUT}/{name}.tex", 'w', encoding='utf-8') as f: f.write(body)

# ---- Table: abnormal change at horizons + p-values (Q1.1) --------------------
taus = (-30, -10, -1, 1, 5, 10, 30)
pos = [int(np.where(res_a['taus'] == t)[0][0]) for t in taus]
rows = []
for c in COLS:
    k = res_a['columns'].index(c)
    cells = [num(res_a['abnormal'][p, k], res_a['pval'][p, k]) for p in pos]
    rows.append(f"{c} & " + " & ".join(cells) + r" \\")
    if c == '30Y': rows.append(r"\midrule")
hdr = " & ".join([f"$\\tau={t:+d}$" for t in taus])
write('tab_q11_horizons', "\\begin{tabular}{l" + "r"*len(taus) + "}\n\\toprule\nSeries & " + hdr +
      " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- generic window table ---------------------------------------------------
def window_table(results, columns, windows, name, first='Sample'):
    head1 = first + " & $N$"
    head2 = " & "
    for a, b in windows:
        head1 += f" & \\multicolumn{{{len(columns)}}}{{c}}{{$[{a:+d},{b:+d}]$}}"
        head2 += " & " + " & ".join(columns)
    cm = " ".join(f"\\cmidrule(lr){{{3+i*len(columns)}-{2+(i+1)*len(columns)}}}" for i in range(len(windows)))
    lines = []
    for label, res in results.items():
        cells = []
        for a, b in windows:
            eff, pv = window_effect(res, a, b)
            for c in columns:
                k = res['columns'].index(c); cells.append(num(eff[k], pv[k]))
        lines.append(f"{label} & {res['n_events']} & " + " & ".join(cells) + r" \\")
    body = ("\\begin{tabular}{l r " + "r"*(len(columns)*len(windows)) + "}\n\\toprule\n" + head1 +
            " \\\\\n" + cm + "\n" + head2 + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write(name, body)

W4 = ((-30, 0), (-1, 1), (0, 5), (0, 30)); W3 = ((-1, 1), (0, 5), (0, 30))
window_table({'Trump actions': res_a}, ['3M', '2Y', '5Y', '10Y', '30Y'], W4, 'tab_q11_windows')
window_table({'Trump actions': res_a, 'Country responses': res_r}, COMPONENTS, W4, 'tab_q12')

REGIONS = ['China', 'EU', 'North America', 'Other countries']
rr = {r: event_study(yields, responses.filter(pl.col('regions').list.contains(r))['idx'], COLS, exclude_idx=all_idx) for r in REGIONS}
ra = {r: event_study(yields, actions.filter(pl.col('regions').list.contains(r))['idx'], COLS, exclude_idx=all_idx) for r in REGIONS + ['Global / other']}
window_table({'All responses': res_r, **rr}, ['2Y', '10Y', 'Slope'], W3, 'tab_q13_responses', first='Responding country')
window_table({'All actions': res_a, **ra}, ['2Y', '10Y', 'Slope'], W3, 'tab_q13_actions', first='Targeted country')

TERMS = {'First term': ('2017-01-20', '2021-01-19'), 'Second term': ('2025-01-20', END_DATE)}
rt, rts = {}, {}
for cat, tab in [('Actions', actions), ('Responses', responses)]:
    for term, period in TERMS.items():
        idx = tab.filter(pl.col('term') == term)['idx']
        rt[f'{cat} -- {term.lower()}'] = event_study(yields, idx, COLS, exclude_idx=all_idx)
        rts[f'{cat} -- {term.lower()}'] = event_study(yields, idx, COLS, exclude_idx=all_idx, baseline_period=period)
window_table(rt, ['2Y', '10Y', '30Y', 'Slope'], W3, 'tab_q14_all')
window_table(rts, ['2Y', '10Y', '30Y', 'Slope'], W3, 'tab_q14_same')

# ---- Q2.1 diagnostics + specifications -------------------------------------
def n_overlaps(idx, others, w=30):
    others = np.asarray(others); return np.array([np.sum((np.abs(others - i) <= w) & (others != i)) for i in idx])
ov = n_overlaps(actions['idx'].to_numpy(), all_idx)
FOMC = ['2017-02-01','2017-03-15','2017-05-03','2017-06-14','2017-07-26','2017-09-20','2017-11-01','2017-12-13',
        '2018-01-31','2018-03-21','2018-05-02','2018-06-13','2018-08-01','2018-09-26','2018-11-08','2018-12-19',
        '2019-01-30','2019-03-20','2019-05-01','2019-06-19','2019-07-31','2019-09-18','2019-10-30','2019-12-11',
        '2020-01-29','2020-03-03','2020-03-15','2020-04-29','2020-06-10','2020-07-29','2020-09-16','2020-11-05','2020-12-16',
        '2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-10-29','2025-12-10']
fomc_idx = np.searchsorted(dates.to_numpy(), np.array(FOMC, dtype='datetime64[D]'))
post = np.array([np.sum((fomc_idx > i) & (fomc_idx <= i + 30)) for i in actions['idx']])
ann = np.array([np.sum(np.abs(fomc_idx - i) <= 1) for i in actions['idx']])
act_idx = actions['idx'].to_numpy()
specs = {
    'A. Assignment benchmark (all non-event days)': res_a,
    'B. Zero benchmark (raw changes)': event_study(yields, act_idx, COLS, exclude_idx=all_idx, baseline='zero'),
    'C. Local pre-event drift $[-280,-31]$': event_study(yields, act_idx, COLS, exclude_idx=all_idx, baseline='local'),
    'D. Declustered events (gap $>$ 30 days)': event_study(yields, decluster_events(act_idx, 30), COLS, exclude_idx=all_idx),
    'E. Excluding FOMC within $\\pm1$ day': event_study(yields, act_idx[ann == 0], COLS, exclude_idx=all_idx),
}
window_table(specs, ['2Y', '10Y', 'Slope'], W3, 'tab_q21_specs', first='Specification')

loadings, evr, scores = pca_factors(yields, n=3)
ypc = yields.with_columns([pl.Series(f'PC{j+1}', scores[:, j]) for j in range(3)])
rpc = {'Trump actions': event_study(ypc, act_idx, ['PC1','PC2','PC3'], exclude_idx=all_idx),
       'Country responses': event_study(ypc, responses['idx'], ['PC1','PC2','PC3'], exclude_idx=all_idx)}
window_table(rpc, ['PC1', 'PC2', 'PC3'], W3, 'tab_q21_pca')

# ---- Q1.3 re-estimated with benchmarks (B) zero and (C) local drift ---------------
def region_idx(r):
    return responses.filter(pl.col('regions').list.contains(r))['idx']
groups = {'All responses': responses['idx']}
groups.update({r: region_idx(r) for r in REGIONS})
bench = {'(A) all non-event days': None, '(B) zero': 'zero', '(C) local drift': 'local'}
cols3 = ['2Y', '10Y', 'Slope']
res_bench = {}
for g, idx in groups.items():
    for b, base in bench.items():
        kw = {} if base is None else {'baseline': base}
        res_bench[(g, b)] = event_study(yields, idx, COLS, exclude_idx=all_idx, **kw)
head1 = "Responding country & $N$"
for b in bench:
    head1 += f" & \\multicolumn{{3}}{{c}}{{{b}}}"
cm = " ".join(f"\\cmidrule(lr){{{3+i*3}-{5+i*3}}}" for i in range(len(bench)))
head2 = " &  & " + " & ".join(" & ".join(cols3) for _ in bench)
lines = []
for g in groups:
    cells = []
    for b in bench:
        r_ = res_bench[(g, b)]
        eff, pv = window_effect(r_, 0, 30)
        for c in cols3:
            k = r_['columns'].index(c); cells.append(num(eff[k], pv[k]))
    lines.append(f"{g} & {res_bench[(g, list(bench)[0])]['n_events']} & " + " & ".join(cells) + r" \\")
write('tab_q21_countries', "\\begin{tabular}{l r " + "r"*(3*len(bench)) + "}\n\\toprule\n" + head1 + " \\\\\n" + cm + "\n" +
      head2 + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

from scipy.stats import t as student_t
p30 = int(np.where(res_a['taus'] == 30)[0][0]); k10 = res_a['columns'].index('10Y')
tst = res_a['tstat'][p30, k10]; pn = 2 * (1 - student_t.cdf(abs(tst), df=res_a['n_events'] - 1))

# normal-day 30-day drift at 2Y (used in the text of Q2.1)
k2 = res_a['columns'].index('2Y')
drift2y = res_a['normal_mean'][p30, k2]

# ---- macros with the diagnostic numbers used in the text --------------------
macros = f"""% numbers used in the text, generated by make_tables.py
\\newcommand{{\\nActions}}{{{res_a['n_events']}}}
\\newcommand{{\\nResponses}}{{{res_r['n_events']}}}
\\newcommand{{\\nAllEventDays}}{{{len(all_idx)}}}
\\newcommand{{\\nNormalDays}}{{{res_a['n_normal']}}}
\\newcommand{{\\nTradingDays}}{{{yields.shape[0]}}}
\\newcommand{{\\lastDate}}{{{dates.max()}}}
\\newcommand{{\\overlapMean}}{{{ov.mean():.1f}}}
\\newcommand{{\\overlapMedian}}{{{np.median(ov):.0f}}}
\\newcommand{{\\overlapMax}}{{{ov.max()}}}
\\newcommand{{\\cleanWindows}}{{{int(np.sum(ov == 0))}}}
\\newcommand{{\\fomcShare}}{{{100*np.mean(post > 0):.0f}}}
\\newcommand{{\\fomcAnn}}{{{int(ann.sum())}}}
\\newcommand{{\\nDeclustered}}{{{len(decluster_events(act_idx, 30))}}}
\\newcommand{{\\tenyAYC}}{{{res_a['abnormal'][p30, k10]:.1f}}}
\\newcommand{{\\tenyT}}{{{tst:.2f}}}
\\newcommand{{\\tenyPnaive}}{{{pn:.3f}}}
\\newcommand{{\\tenyPplacebo}}{{{res_a['pval'][p30, k10]:.3f}}}
\\newcommand{{\\driftTwoY}}{{{drift2y:.1f}}}
\\newcommand{{\\pcaVarOne}}{{{100*evr[0]:.0f}}}
\\newcommand{{\\pcaVarTwo}}{{{100*evr[1]:.0f}}}
\\newcommand{{\\pcaVarThree}}{{{100*evr[2]:.0f}}}
"""
write('numbers', macros)

# ---- appendix: event tables ---------------------------------------------------
def esc(s): return s.replace('&', r'\&').replace('%', r'\%').replace('$', r'\$').replace('#', r'\#').replace('"', "''")
for name, tab in [('tab_events_actions', actions), ('tab_events_responses', responses)]:
    lines = []
    for r in tab.iter_rows(named=True):
        lines.append(f"{r['trade_date']} & {', '.join(r['regions'])} & {esc(r['description'])} \\\\")
    body = ("\\begin{longtable}{l p{2.8cm} p{8.6cm}}\n\\toprule\nTrading day & Region & Event \\\\\n\\midrule\n\\endhead\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n")
    write(name, body)

print('tables written to', OUT)
print(f"N={res_a['n_events']}/{res_r['n_events']}, normal={res_a['n_normal']}, overlap {ov.mean():.1f}, clean {np.sum(ov==0)}, fomc {np.mean(post>0):.0%}/{ann.sum()}, 10Y {res_a['abnormal'][p30,k10]:.1f} t={tst:.2f} p={pn:.3f}/{res_a['pval'][p30,k10]:.3f}, drift2y {drift2y:.1f}, evr {evr}")
