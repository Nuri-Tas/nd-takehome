# Figure provenance

| figure | source | pipeline |
|---|---|---|
| fig1_stopping_prior | `numbers_qed_prior.json` | pre-fix models; the comparison is between three positional schemes trained on identical data, so it is internally controlled and the contamination cannot affect it |
| fig2_written_lengths | `numbers_length_probe_hard.json` | pre-fix models, probed on the clean hard pool; same internal-control argument |
| fig3_rounds | `runs/clean_seed0/log.json` | **clean** |
| fig4_length_illusion | pool-construction counts | unaffected (internal pools only) |
| fig5_codec | `numbers_ablation_absrefs.json` | pre-fix data, but absolute vs relative were trained on the *same* data and probed on the *same* clean targets, so the 6-vs-7 frontier comparison is controlled |
| fig6_energy | `numbers_energy_vs_length.json`, `numbers_frontier_vs_k.json` | pre-fix models, measured on the clean hard pool |

The contamination was between Stage-1 training data and the two external
benchmarks (validation_36, test). Every figure above except fig3 is a
*within-pipeline comparison* -- both arms trained on identical data -- so the
contamination shifts both arms equally and cannot produce the effect shown.
Re-measuring Stage 1 clean confirmed this directly: P is 7 either way, and the
validation/test scores moved by less than a point.
| fig7_mechanism | `numbers_energy.json`, `numbers_energy_vs_length.json` | pre-fix models (Stage-1 vs after-RL from the same pipeline), so the before/after comparison is internally controlled; it demonstrates the mechanism, not a headline number |
