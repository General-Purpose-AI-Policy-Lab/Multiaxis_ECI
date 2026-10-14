# Notes (Claude)

**2026-10-10 04:25.** smc3 (256 particles, 50x20) at step 41, lambda 0.130, ~400-500 s per temperature
(post-update Spotlight/MediaAnalysis indexing takes ~2 cores); its cap raised from 12 h to 18 h in
jobs.txt (a capped run resumes from its checkpoint with the new cap). The 6-hourly scheduled
check-in stalled at its first Bash call (tool approval needed, nobody to approve): disabled; the
study is followed from the main Claude session instead. The final-report task may stall the same way:
the main session writes REPORT.md itself if it is still alive.

**2026-10-10 07:37 (scheduled check-in).** Nothing finished yet. Runner (pid 5415) and smc3 (pid 5419) alive;
smc3 at step 78, lambda 0.440, 29544 s elapsed, ~280 s per step, acceptance ~0.62, step size 0.018 and falling.
At the current lambda increments (~0.01/step) it needs roughly 40-55 more steps (3-4 h), so it may hit the
running process's 12 h cap (~11:20) just before lambda = 1; the runner then resumes it from ckpt.npz under the
18 h cap in jobs.txt. No change to jobs.txt: the smc4 decision waits for smc3's particles. Timing check: smc3
done ~11:30-13:00 today, phase 2 (~10 anneals + 2 polishes, ~26 h) to ~Oct 11 15:00, smc4 (~14 h) and
crit_s6/s7 to ~Oct 12 10:00, inside the 22:00 limit.

**2026-10-10 11:00 (main session).** smc3 done: lambda = 1 after 119 temperatures (41350 s). Particle loglik
median -14491, max -14302, 26/256 above -14395; per-particle nearest mode A for 212/256 (single-particle
loadings are noisy: one draw of mode A itself matches A at only 0.31, so read this as indicative). Added
after crit_long: smc3r (500 HMC rejuvenation steps on smc3's particles at lambda = 1), smc3_top (export of
its 10 best particles) and pol_smc3 (NUTS polish, one chain per particle). smc.py gained --from-run,
--rejuv, --export-top. smc4 kept for now; to revisit after smc3r (if rejuvenation brings the particles to
the posterior, a 256-particle run with --rejuv may beat a 512-particle one at the same cost).

**2026-10-10 12:55 (main session).** crit_long (critical stages 600 + 100 instead of 300 + 50): 0/10 above
-14395, median -14436, best -14396 / logpost -21153: longer critical stages do not help (base recipe: 3, 2, 2
of 10 over three seeds). pol_crit_long will be skipped (no chain above the threshold). smc3r started.

**2026-10-10 14:20 (main session).** smc3r (500 rejuvenation steps): particle loglik median -14491 -> -14449, max
-14257, 36/256 above -14395, still short of equilibrium. pol_smc3 (NUTS from the 10 best particles, -14272..-14346):
all fell to -14423..-14465 (logpost -21180..-21214), 9/10 nearest A but match <= 0.61: the SMC's highs were
non-equilibrium, like the annealing transients of round 1. smc4 (512 particles) dropped: the limit is mixing near
lambda = 1, not the particle count. Replaced by crit_narrow (critical window 0.04-0.2, denser ladder around the
loglik jump) and two more base-recipe replicates (crit_s8, crit_s9).

**2026-10-10 15:55 (main session).** crit_kb2 (keep-best 2): 2/10 above -14395, median -14432, best chain -14352 / logpost -21083 (D-like, match 0.75): the best log posterior of the study so far. pol_crit_kb2 queued right after crit_resample.

**2026-10-10 17:30 (main session).** crit_resample (damped resampling on stage means in the window): 0/10 above -14395, median -14428, best -14399 / logpost -21146. Resampling, like keep-best 5, does worse than keep-best 2-3. pol_crit_kb2 running.

**2026-10-10 18:10 (main session).** pol_crit_kb2 (10 NUTS chains from crit_kb2 chain 4, -14352): 6/10 above -14395, median -14390, best -14381, logpost -21128 to -21170; all D-like (match 0.70-0.76) except one A-like (0.70) and one C-like. Like pol_crit_select (E-like, median -14389, logpost ~-21140), a non-A family at A's likelihood, its log posterior 20-40 below A's typical chain (-21108) but within A's own chain spread (-21107 to -21171). The 14352 high fell back by ~40 once polished.

**2026-10-10 20:00 (main session).** crit_wide (window 0.02-0.6): 1/10 above -14395 (-14395 exactly), median -14415, best logpost -21143. Not better than the base window 0.03-0.4. Population batches started (pop_b1).

**2026-10-10 22:45 (main session).** pop_finish (best 10 of 30 chains selected at T = 0.4 on stage means, then finished): 0/10 above -14395, median -14433, best -14397 / logpost -21138. The stage mean at T = 0.4 does not predict the final basin; the base recipe's edge (3, 2, 2 of 10 vs the single control's 1/10) may be partly luck. Added three plain-NUTS controls at the same budget (control_nuts2-4, interleaved with the base replicates) so both have several seeds.

**2026-10-11 00:20 (main session).** crit_s4 (base recipe, seed 4): 0/10 above -14395, median -14434, best -14401 / logpost -21143. Base recipe so far 3, 2, 2, 0 of 10 (7/40). control_nuts2 running.

**2026-10-11 02:10 (main session).** control_nuts2 (plain pymc NUTS, 1800 + 300, seed 191000): 5/10 above -14395, 3 at A's level, median -14399, best -14373 / logpost -21112, families D, E, B: better than every annealing run. Controls 6/20 vs base recipe 7/40 so far; large seed-to-seed variance. No short-run chain (100+) has reached A's basin (match_A <= 0.62), while the flagship's long nutpie run put 4/20 there. Added nutpie controls (anneal.py --sampler nutpie; a 20+20 smoke test already reached -14512), a 4000-tune pymc control and more seeds; crit_s6/s7 dropped.

**2026-10-11 04:05 (main session).** crit_s5: 0/10 above -14395 (median -14452): base recipe 7/50 over five seeds. control_nutpie1 (nutpie 1800 + 300): 4/10 above, 3 at A's level, median -14411, best -14374 / logpost -21123; families E, B, D; one chain (-14413) is the first of round 2 to match A's loadings above 0.8 (0.82). Plain samplers so far: pymc NUTS 6/20, nutpie 4/10, vs annealing 7/50.

**2026-10-11 05:45 (main session).** control_nuts3: 2/10 above -14395, median -14417, best -14389 / logpost -21140. pymc NUTS controls: 1, 5, 2 of 10 (8/30). crit_narrow running.

**2026-10-11 07:30 (main session).** crit_narrow (critical window 0.04-0.2, denser ladder): 4/10 above -14395, 3 at A's level, median -14405, best -14366 / logpost -21123: the best annealing run of round 2, on par with the best controls. One seed only: queued crit_narrow2, crit_narrow3 with control_nutpie4 and control_nuts6 alongside.

**2026-10-11 09:05 (main session).** crit_s8: 0/10 above -14395, median -14438. Base recipe 7/60 over six seeds (3, 2, 2, 0, 0, 0). control_nuts4 running.

**2026-10-11 10:45 (main session).** control_nuts4: 7/10 above -14395, 6 at A's level, median -14379, best -14364 / logpost -21105 (above A's mean). pymc NUTS controls 15/40 (1, 5, 2, 7). The case for annealing is gone; the base recipe is at 7/60.

**2026-10-10 12:27 (scheduled check-in).** Nothing new finished since 11:00. Runner (pid 5415) alive; crit_long
(pid 15490, 10 workers) in its final T = 1 stage after 3817 s (T = 0.8 stage: loglik max -15023, median -15118,
not yet comparable), so it should end around 12:45. smc3 vs the jobs.txt rule: median -14491 is far below
-14395 and no particle matched a mode at >= 0.8, which would argue for replacing smc4; but the 11:00 session
deliberately deferred that to smc3r (rejuvenation at lambda = 1) and pol_smc3, both queued next and due by
~15:00, and smc4 is ~30 h away in the queue, so jobs.txt left unchanged. Timing: phase 2 remaining ~22 h
(to ~Oct 11 11:00), smc4 capped at 16 h (to ~Oct 12 03:00), crit_s6/s7 to ~Oct 12 09:00: inside 22:00.
Next check-in: decide smc4 from smc3r/pol_smc3 (replace by two crit-window runs if the polished particles
stay below -14395), and compare crit_long with the base recipe (3, 2, 2 of 10 above; medians -14422/-14398/-14424).

**2026-10-12 22:25 (scheduled check-in).** Queue empty since control_nuts6 ended 2026-10-11 22:58; the runner has been relaunched by the LaunchAgent several times and exited each time after 2 h with no job (normal). No failures. Finished since the 10-11 10:45 note: crit_s9 2/10 above -14395 (median -14424, best -14363 / logpost -21118, B,E); control_long (pymc 4000 tune) 5/10 (-14400, -14354 / -21109, B,D); control_nutpie2 4/10 (-14412, -14370 / -21123, A,B,D); control_nuts5 3/10 (-14410, -14379 / -21125, B,C); control_nutpie3 (4000 tune) 4/10 (-14405, -14370 / -21119, A,C,D); crit_narrow2 2/10 (-14414, -14391 / -21133, B,C); control_nutpie4 3/10 (-14402, -14374 / -21111, B,D); crit_narrow3 1/10 (-14409, -14392 / -21139, C); control_nuts6 3/10 (-14421, -14363 / -21111, B,C,D). Totals: pymc NUTS 1800+300 21/60 (1,5,2,7,3,3); pymc 4000 tune 5/10; nutpie 1800 11/30 (4,4,3); nutpie 4000 4/10; base annealing recipe 9/70 (3,2,2,0,0,0,2); crit_narrow 7/30 (4,2,1): crit_narrow's first 4/10 did not replicate. Best log posterior of round 2 remains crit_kb2's -21083 (D-like, fell back once polished); among plain runs control_nuts4's -21105. Families reaching A's level: mostly B, D, C, E; A only in two nutpie runs. jobs.txt unchanged: past the 22:00 cutoff for new work, and the runner starts nothing after 23:00. Remaining: final REPORT.md (main session).
