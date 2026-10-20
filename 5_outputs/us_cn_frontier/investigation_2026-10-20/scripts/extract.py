"""Per-draw ECI-H of every candidate of the country cut, saved once per scope for the variant study."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]   # the repository
sys.path.insert(0, str(ROOT / '2_model'))
spec = importlib.util.spec_from_file_location('fg1', ROOT / '4_diagnostics/1_frontier_gap.py')
fg1 = importlib.util.module_from_spec(spec); sys.modules["fg1"] = fg1; spec.loader.exec_module(fg1)
from multiaxis_eci import config
from multiaxis_eci.analysis.frontier_gap import group_labels, linking_members, link_to_reference
from multiaxis_eci.analysis.stats import _release_dates, eci_transform
from multiaxis_eci.analysis.timelines import candidate_mask
from multiaxis_eci.data import PROCESSED_FILE
OUT = Path(sys.argv[1])
model_dates, _ = _release_dates(pd.read_csv(PROCESSED_FILE))
for scope in ['all', 'public', 'private']:
    folder = 'canonical_humanmerge' if scope == 'all' else f'canonical_{scope}_humanmerge'
    rd = config.RESULTS_DIR / folder
    mini, theta0 = fg1.load_matched(rd / 'trace.nc', scope, False)
    names = mini.mlookup.sort_values('model_idx')['model'].tolist()
    keep = candidate_mask(theta0[:, :, None], 0, mini, model_dates) & (mini.n_obs_per_model >= 2)
    if scope == 'all':
        tr = eci_transform(theta0, mini); a, b = tr.a, tr.b
    else:
        ref = pd.read_csv(config.RESULTS_DIR / 'canonical_humanmerge' / 'all_models_eci.csv').set_index('name')['mean']
        members = linking_members(names, keep, mini.n_obs_per_model, 'country')
        a, b, n = link_to_reference(theta0, names, members, ref)
    E = a[:, None] + b[:, None] * theta0
    labels = group_labels('country', names)
    lab = np.array([labels.get(m, '') for m in names])
    dates = np.array([str(pd.Timestamp(model_dates[m]).date()) if m in model_dates.index else '' for m in names])
    np.savez_compressed(OUT / f'E_{scope}.npz', E=E.astype(np.float32), names=np.array(names), keep=keep,
                        label=lab, date=dates, n_obs=mini.n_obs_per_model)
    print(scope, E.shape, 'candidates', keep.sum(), 'US', (keep & (lab == 'US')).sum(), 'CN', (keep & (lab == 'CN')).sum(), flush=True)
