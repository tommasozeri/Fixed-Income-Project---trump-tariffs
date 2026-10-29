"""
Event-study toolkit for the FIN2764 group assignment
("US Treasury Yield Curve Analysis: Trump Tariff Threats Impact").

Conventions used throughout
---------------------------
* Yields are Treasury constant-maturity (par) yields in percent, as published
  by FRED (DGSxx series) / the US Treasury. All changes are reported in basis
  points (bp).
* Event time tau is measured in *trading days* (rows of the yield dataset),
  tau = -30, ..., +30.
* Yield change in an event window, relative to the event date t_e:
      dY_e(tau) = Y(t_e + tau) - Y(t_e)            (so dY_e(0) = 0)
* Abnormal yield change (AYC), as required by the assignment:
      AYC(tau) = mean_e dY_e(tau) - mean_{d in non-event dates} dY_d(tau)
  i.e. the average path around events minus the average path around every
  other ("normal") trading day.
"""

import datetime
import io
import os
import re

import numpy as np
import polars as pl
import requests
import matplotlib.pyplot as plt


MATURITIES = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
MATURITY_YEARS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
COMPONENTS = ['Level', 'Slope', 'Curvature']

# Start of Trump's second mandate: used to split the sample in two terms
SECOND_TERM_START = '2021-01-20'


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

## Fallback data source: US Treasury daily par yield curve (the raw source of
## the FRED DGSxx series). Used only when no FRED API key is available.
TREASURY_COLUMNS = {
    '3 Mo': '3M', '6 Mo': '6M', '1 Yr': '1Y', '2 Yr': '2Y', '3 Yr': '3Y',
    '5 Yr': '5Y', '7 Yr': '7Y', '10 Yr': '10Y', '20 Yr': '20Y', '30 Yr': '30Y',
}


def fetch_yield_curve_treasury(start_date='2015-01-01', end_date=None):
    """
    Download the daily Treasury par yield curve from home.treasury.gov, one
    calendar year at a time, and return it in the same layout as
    fred.fetch_yield_curve_data (columns: date, 3M, ..., 30Y).
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4]) if end_date else datetime.date.today().year

    url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           "daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve"
           "&field_tdr_date_value={y}&page&_format=csv")

    frames = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching Treasury par yield curve for {year}...")
        r = requests.get(url.format(y=year), timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        raw = pl.read_csv(io.StringIO(r.text), infer_schema_length=0)  # read as strings
        df = raw.select(
            [pl.col('Date').str.to_date('%m/%d/%Y').alias('date')]
            + [pl.col(src).cast(pl.Float64, strict=False).alias(dst)
               for src, dst in TREASURY_COLUMNS.items() if src in raw.columns]
        )
        frames.append(df)

    data = pl.concat(frames, how='diagonal').sort('date')
    data = data.filter(pl.col('date') >= pl.lit(start_date).str.to_date())
    if end_date:
        data = data.filter(pl.col('date') <= pl.lit(end_date).str.to_date())
    print(f"\n✅ Treasury data: {data['date'].min()} to {data['date'].max()}, shape {data.shape}")
    return data


def prepare_yields(yield_df):
    """
    Clean the raw yield panel and add the three yield-curve components.

    * Rows where every maturity is missing (FRED market holidays) are dropped;
      isolated missing values are forward-filled (assumption: no change).
    * Components (in % like the yields):
        Level     = average of the 10 maturities (3M ... 30Y)
        Slope     = 10Y - 3M
        Curvature = 2 x 5Y - 2Y - 10Y   (2s5s10s butterfly)
    """
    df = (yield_df
          .select(['date'] + MATURITIES)
          .filter(~pl.all_horizontal([pl.col(m).is_null() for m in MATURITIES]))
          .sort('date')
          .with_columns([pl.col(m).forward_fill() for m in MATURITIES])
          .drop_nulls())
    df = df.with_columns([
        pl.mean_horizontal(MATURITIES).alias('Level'),
        (pl.col('10Y') - pl.col('3M')).alias('Slope'),
        (2 * pl.col('5Y') - pl.col('2Y') - pl.col('10Y')).alias('Curvature'),
    ])
    return df


# ---------------------------------------------------------------------------
# 2. Events
# ---------------------------------------------------------------------------

## Keyword rules used to tag the country/region targeted by (or responding to)
## each event. An event can belong to several groups (e.g. "Canada, Mexico,
## China condemn tariffs"); events with no match are tagged 'Global / other'.
REGION_PATTERNS = {
    'China': r'China|Chinese|Hong Kong|yuan',
    'EU': r'\bEU\b|European',
    'North America': r'Canada|Mexico|USMCA',
    'Other countries': r'India|Japan|Korea|Brazil|Vietnam|Turkey|Russia|Malaysia|New Zealand|14 countries|Asian',
}


def tag_regions(description):
    regions = [name for name, pat in REGION_PATTERNS.items() if re.search(pat, description)]
    return regions if regions else ['Global / other']


def build_event_table(events, category, trading_dates):
    """
    Turn a list of (date_str, description) tuples into an event table.

    Each calendar event date is mapped to the first trading day on or after
    it (assumption: announcements made on weekends/holidays are priced on the
    next trading day). Events falling on the same trading day are merged.

    Returns a polars DataFrame with columns:
        trade_date, idx (row in the yield panel), category, term,
        regions (list[str]), description, calendar_dates
    """
    td = np.asarray(trading_dates, dtype='datetime64[D]')
    rows = []
    for date_str, desc in events:
        d = np.datetime64(date_str, 'D')
        i = int(np.searchsorted(td, d, side='left'))
        if i >= len(td):
            continue  # event after the end of the sample
        rows.append({
            'trade_date': td[i].astype(object),
            'idx': i,
            'calendar_date': date_str,
            'description': desc,
        })
    ev = pl.DataFrame(rows)
    ev = (ev.group_by(['trade_date', 'idx'], maintain_order=True)
            .agg([pl.col('calendar_date').alias('calendar_dates'),
                  pl.col('description').str.join(' || ').alias('description')])
            .sort('idx'))
    ev = ev.with_columns([
        pl.lit(category).alias('category'),
        pl.when(pl.col('trade_date') < pl.lit(SECOND_TERM_START).str.to_date())
          .then(pl.lit('First term')).otherwise(pl.lit('Second term')).alias('term'),
        pl.col('description').map_elements(tag_regions, return_dtype=pl.List(pl.Utf8)).alias('regions'),
    ])
    return ev.select(['trade_date', 'idx', 'category', 'term', 'regions', 'description', 'calendar_dates'])


def decluster_events(event_idx, min_gap=30):
    """
    Keep only the first event of each cluster: an event is dropped if it
    occurs fewer than `min_gap` trading days after the previously kept event.
    Used as a robustness check against overlapping event windows.
    """
    kept = []
    for i in sorted(event_idx):
        if not kept or i - kept[-1] > min_gap:
            kept.append(i)
    return np.array(kept, dtype=int)


# ---------------------------------------------------------------------------
# 3. Event-study engine
# ---------------------------------------------------------------------------

def window_changes(values, centre_idx, pre=30, post=30):
    """
    values: (T, K) array of yields (%). centre_idx: (N,) row indices.
    Returns dY with shape (N, pre+post+1, K) in bp, dY[n, tau+pre, k] =
    100 * (values[c+tau, k] - values[c, k]). Out-of-sample points are NaN.
    """
    T = values.shape[0]
    taus = np.arange(-pre, post + 1)
    rows = np.asarray(centre_idx)[:, None] + taus[None, :]            # (N, L)
    valid = (rows >= 0) & (rows < T)
    out = np.full((len(centre_idx), len(taus), values.shape[1]), np.nan)
    out[valid] = values[rows[valid]]
    out = 100.0 * (out - values[np.asarray(centre_idx)][:, None, :])
    return out


def event_study(yields, event_idx, columns, pre=30, post=30,
                exclude_idx=None, baseline_period=None,
                baseline='non_event', est_window=(-280, -31),
                n_placebo=2000, seed=42):
    """
    Average and abnormal yield changes around a set of events.

    Parameters
    ----------
    yields         : polars DataFrame from prepare_yields
    event_idx      : row indices of the event trading days
    columns        : maturities / components to analyse
    exclude_idx    : indices that may not be used as 'normal' days (default:
                     the events themselves). Pass the union of all event sets
                     to keep every tariff-related day out of the benchmark.
    baseline_period: optional (start, end) date strings restricting the pool
                     of normal days (e.g. same presidential term).
    baseline       : 'non_event' -> AYC = mean event path - mean normal path
                                    (assignment definition, default)
                     'zero'      -> AYC = raw mean event path (random walk)
                     'local'     -> constant-drift model estimated for each
                                    event on its own pre-event window
                                    est_window (in trading days)
    n_placebo      : number of placebo draws (random samples of normal days of
                     the same size as the event sample) used for inference.

    Returns a dict with taus, the event/normal/abnormal paths (L, K) in bp,
    cross-sectional t-stats, placebo 95% bands and two-sided placebo p-values.
    """
    values = yields.select(columns).to_numpy()
    dates = yields['date'].to_numpy()
    T = len(values)
    taus = np.arange(-pre, post + 1)
    event_idx = np.asarray(event_idx, dtype=int)

    dY_ev = window_changes(values, event_idx, pre, post)                # (N, L, K)

    # pool of normal days: full window available, not an excluded day
    excl = set(event_idx.tolist()) | (set(np.asarray(exclude_idx).tolist()) if exclude_idx is not None else set())
    pool = np.arange(pre, T - post)
    pool = pool[~np.isin(pool, list(excl))]
    if baseline_period is not None:
        lo, hi = (np.datetime64(baseline_period[0]), np.datetime64(baseline_period[1]))
        pool = pool[(dates[pool] >= lo) & (dates[pool] <= hi)]
    dY_pool = window_changes(values, pool, pre, post)                   # (P, L, K)
    normal = dY_pool.mean(axis=0)

    if baseline == 'non_event':
        expected_ev = np.broadcast_to(normal, dY_ev.shape)
        expected_pool = normal
    elif baseline == 'zero':
        normal = np.zeros_like(normal)
        expected_ev = np.zeros_like(dY_ev)
        expected_pool = 0.0
    elif baseline == 'local':
        # expected change = tau x average daily change over the estimation window
        a, b = est_window
        def drift(idx):
            lo_, hi_ = np.clip(idx + a, 0, T - 1), np.clip(idx + b, 0, T - 1)
            return 100.0 * (values[hi_] - values[lo_]) / np.maximum(hi_ - lo_, 1)[:, None]
        expected_ev = drift(event_idx)[:, None, :] * taus[None, :, None]
        expected_pool = drift(pool)[:, None, :] * taus[None, :, None]
        normal = expected_ev.mean(axis=0)
    else:
        raise ValueError(baseline)

    ab_ev = dY_ev - expected_ev                                         # per-event abnormal
    ab_pool = dY_pool - expected_pool
    N = np.sum(~np.isnan(ab_ev[:, -1, 0]))
    mean_ev = np.nanmean(dY_ev, axis=0)
    abnormal = np.nanmean(ab_ev, axis=0)
    se = np.nanstd(ab_ev, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(ab_ev), axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        tstat = abnormal / se

    # placebo inference: distribution of the mean abnormal change over
    # random samples of normal days with the same number of events
    rng = np.random.default_rng(seed)
    n = len(event_idx)
    draws = np.empty((n_placebo,) + abnormal.shape)
    for b in range(n_placebo):
        draws[b] = ab_pool[rng.integers(0, len(pool), n)].mean(axis=0)
    draws -= draws.mean(axis=0)                                         # centre under H0
    lo_band, hi_band = np.percentile(draws, [2.5, 97.5], axis=0)
    pval = (np.abs(draws) >= np.abs(abnormal)[None]).mean(axis=0)

    return {
        'taus': taus, 'columns': list(columns), 'n_events': int(n), 'n_normal': int(len(pool)),
        'event_mean': mean_ev, 'normal_mean': normal, 'abnormal': abnormal,
        'se': se, 'tstat': tstat, 'band_lo': lo_band, 'band_hi': hi_band, 'pval': pval,
        'event_paths': dY_ev, 'draws': draws,
    }


def summary_table(res, taus=(-30, -10, -1, 1, 5, 10, 30), stat='abnormal'):
    """Abnormal change (bp) at selected horizons, one row per column, with
    placebo p-values in brackets as a separate frame."""
    pos = [int(np.where(res['taus'] == t)[0][0]) for t in taus]
    tab = {'series': res['columns']}
    pv = {'series': res['columns']}
    for t, p in zip(taus, pos):
        tab[f'tau={t:+d}'] = np.round(res[stat][p], 1)
        pv[f'tau={t:+d}'] = np.round(res['pval'][p], 3)
    return pl.DataFrame(tab), pl.DataFrame(pv)


def window_effect(res, start, end):
    """
    Abnormal change between tau=start and tau=end (bp), e.g. (-1, +1), for
    every column, with its placebo p-value. Returns (effect (K,), pval (K,)).
    """
    i0 = int(np.where(res['taus'] == start)[0][0])
    i1 = int(np.where(res['taus'] == end)[0][0])
    eff = res['abnormal'][i1] - res['abnormal'][i0]
    null = res['draws'][:, i1] - res['draws'][:, i0]
    pval = (np.abs(null) >= np.abs(eff)[None]).mean(axis=0)
    return eff, pval


def _stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


def compare_table(results, columns, windows=((-30, 0), (-1, 1), (0, 5), (0, 30))):
    """
    One row per event sample (dict label -> result), one column per
    (series, window): abnormal change in bp with placebo significance stars
    (*** 1%, ** 5%, * 10%). Window (-30, 0) is the pre-event drift
    Y(t) - Y(t-30) in excess of normal days.
    """
    rows = []
    for label, res in results.items():
        row = {'sample': label, 'N': res['n_events']}
        for a, b in windows:
            eff, pv = window_effect(res, a, b)
            for c in columns:
                k = res['columns'].index(c)
                row[f'{c} [{a:+d},{b:+d}]'] = f"{eff[k]:.1f}{_stars(pv[k])}"
        rows.append(row)
    return pl.DataFrame(rows)


def pca_factors(yields, n=3):
    """
    Principal components of daily yield changes (bp) across the 10 maturities.
    Returns (loadings (10, n), explained variance ratio (n,), factor scores as
    cumulative levels (T, n) so they can be fed to event_study like yields).
    Signs are normalised so that PC1 loads positively on 10Y (level),
    PC2 is long-minus-short (slope) and PC3 is belly-minus-wings (curvature).
    """
    Y = yields.select(MATURITIES).to_numpy()
    dY = np.diff(Y, axis=0) * 100
    dY_c = dY - dY.mean(axis=0)
    U, S, Vt = np.linalg.svd(dY_c, full_matrices=False)
    load = Vt[:n].T
    i10, i3m, i5 = MATURITIES.index('10Y'), MATURITIES.index('3M'), MATURITIES.index('5Y')
    if load[i10, 0] < 0: load[:, 0] *= -1
    if n > 1 and load[i10, 1] - load[i3m, 1] < 0: load[:, 1] *= -1
    if n > 2 and load[i5, 2] < 0: load[:, 2] *= -1
    evr = (S[:n] ** 2) / np.sum(S ** 2)
    scores = np.vstack([np.zeros(n), np.cumsum(dY_c @ load, axis=0)]) / 100  # back to % units
    return load, evr, scores


# ---------------------------------------------------------------------------
# 4. Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, savepath):
    """Save a figure as PNG (for the notebook) and as vector PDF (for the report)."""
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(savepath)[0] + '.pdf', bbox_inches='tight')


def plot_paths(res, columns=None, title='', ncols=3, savepath=None):
    """Abnormal change path over event time, with 95% placebo band, one
    panel per maturity / component."""
    columns = columns or res['columns']
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, columns):
        k = res['columns'].index(c)
        ax.fill_between(res['taus'], res['band_lo'][:, k], res['band_hi'][:, k],
                        color='grey', alpha=0.25, label='95% placebo band')
        ax.plot(res['taus'], res['event_mean'][:, k], color='tab:orange', lw=1, ls='--', label='Event avg.')
        ax.plot(res['taus'], res['normal_mean'][:, k], color='tab:green', lw=1, ls=':', label='Normal avg.')
        ax.plot(res['taus'], res['abnormal'][:, k], color='tab:blue', lw=2, label='Abnormal')
        ax.axvline(0, color='k', lw=0.8)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(c)
        ax.set_xlabel('Trading days relative to event')
        ax.set_ylabel('bp')
    for ax in list(axes.flat)[len(columns):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{title} (N = {res['n_events']} events)", fontsize=13)
    fig.tight_layout()
    if savepath:
        _save(fig, savepath)
    return fig


def plot_curve_change(results, tau=30, title='', savepath=None):
    """Abnormal change of the whole yield curve at a given horizon, for one or
    several event samples (dict label -> result). Error bars: placebo band."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, res in results.items():
        p = int(np.where(res['taus'] == tau)[0][0])
        ks = [res['columns'].index(m) for m in MATURITIES]
        y = res['abnormal'][p, ks]
        lo = res['band_lo'][p, ks]
        hi = res['band_hi'][p, ks]
        line, = ax.plot(MATURITY_YEARS, y, marker='o', label=f"{label} (N={res['n_events']})")
        ax.fill_between(MATURITY_YEARS, lo, hi, color=line.get_color(), alpha=0.12)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xscale('log')
    ax.set_xticks(MATURITY_YEARS)
    ax.set_xticklabels(MATURITIES)
    ax.set_xlabel('Maturity')
    ax.set_ylabel('Abnormal change (bp)')
    ax.set_title(f"{title} - abnormal yield curve change at tau = {tau:+d}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if savepath:
        _save(fig, savepath)
    return fig


def plot_compare_paths(results, column, title='', savepath=None):
    """Abnormal path of one series for several event samples."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, res in results.items():
        k = res['columns'].index(column)
        line, = ax.plot(res['taus'], res['abnormal'][:, k], lw=2, label=f"{label} (N={res['n_events']})")
        ax.fill_between(res['taus'], res['band_lo'][:, k], res['band_hi'][:, k],
                        color=line.get_color(), alpha=0.10)
    ax.axvline(0, color='k', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Trading days relative to event')
    ax.set_ylabel('Abnormal change (bp)')
    ax.set_title(f"{title} - {column}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if savepath:
        _save(fig, savepath)
    return fig
