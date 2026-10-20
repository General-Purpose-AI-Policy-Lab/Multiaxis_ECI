"""The print months-behind figures and the table rows for the top-2 reading written by run.sh."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / '2_model'))
from multiaxis_eci.analysis.frontier_gap import SCOPES, load_scope, summarize
from multiaxis_eci.viz.frontier_gap_print import lag_fig_print, save_print
spec = importlib.util.spec_from_file_location('plot2', ROOT / '4_diagnostics/2_plot_frontier_gap.py')
plot2 = importlib.util.module_from_spec(spec); sys.modules['plot2'] = plot2; spec.loader.exec_module(plot2)
by_class = plot2.benchmarks_by_class()
S = Path(sys.argv[1])
for kind, scopes in (('country', ['public', 'private']), ('openness', ['all', 'public', 'semi_private', 'private'])):
    d = S / kind
    results = {s: load_scope(d, kind, s) for s in SCOPES if (d / f'frontier_gap_{kind}_{s}_records.csv').exists()}
    scopes = [s for s in scopes if s in results]
    fig = lag_fig_print(results, scopes, today='2026-10-01', label_scope='all' if 'all' in scopes else scopes[0],
                        group=kind, lang='en', counts={s: len(by_class[s]) for s in scopes})
    save_print(fig, d / f'lag_top2_{kind}.png', d / f'lag_top2_{kind}.svg')
    print(f'== {kind}: points since 2024 per scope:', {s: int((results[s].lag_df.release_date >= '2024-01-01').sum()) for s in results})
    for key, label in plot2.SHOWN_ROWS + [('lag_today', 'lag today, trend lines')]:
        row = {s: plot2.fmt(summarize(plot2.DRAWS[key](results[s])), False) if plot2.DRAWS[key](results[s]) is not None else 'n/a' for s in results}
        print(f'   {label[:60]:<60s}', row)
    print('   wrote', d / f'lag_top2_{kind}.png')
