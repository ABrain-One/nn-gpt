# TuneNNGen Results Registry

Easy-to-find derived result files for paper analysis. These are mined from logged summaries; they are not new experiments.

## Current Key Files

- `acc_gap_advantage_pairs_20260606.json`: full comparable pair analysis with Pearson correlation and skip counts.
- `acc_gap_advantage_pairs_20260606.csv`: spreadsheet-friendly pair table.
- `family_win_rates_20260606.md`: central per-target-family win-rate table for the revised paper.
- `family_win_rates_20260606.csv`: spreadsheet-friendly family win-rate table.
- `family_win_rates_20260606.json`: machine-readable family win-rate table.
- `paper_expected_outputs_20260621.csv`: canonical reported values for the main, cross-dataset, and `hp_copy` paper tables.

The `acc_gap_advantage_pairs` files aggregate comparable pairs mined across all
logged runs. They are used for family-level analysis and can select a different
run than the frozen main-table protocol. Paper-table values and `hp_copy`
ablations are therefore kept separately in `paper_expected_outputs_20260621.csv`.

## Source Artifact

- `/home/kabir/newws/acc_gap_advantage_analysis.json`
