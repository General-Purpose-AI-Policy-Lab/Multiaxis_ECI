# The published K=4 trace (deleted 2026-09-11)

`trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc` (13 GB, 10 chains x 12,000 draws, 5,000 tuning steps, nutpie 0.16.8, sampled 2026-08-27) was deleted to free disk. Its tables, modes file, forecast caches and hand-confirmed `axis_names.json` stay in this folder.

To regenerate it, check out the code that produced it and run the flagship fit on the input snapshot versioned with it:

```bash
git worktree add ../Multiaxis_ECI_frozen origin/blogpost-frozen
cd ../Multiaxis_ECI_frozen
python 2_fit.py --K 4 --human-merge --lineage-prior --lineage-bm --chains 10 --draws 12000 --tune 5000
```

The input is `1_data/processed/benchmarks_merged.csv` on that branch (835 model versions, 100 benchmarks before the curated exclusions; the fit saw 4,923 observations, 829 test-takers, 96 benchmarks), the random seed is `config.SAMPLE_KW["random_seed"] = 42`, and the trace's own `mirt_spec` attribute recorded the full specification. A refit reproduces the posterior up to MCMC and floating-point nondeterminism, not the file byte for byte.
