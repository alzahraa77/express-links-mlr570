# Sparse Sensor Links for Low-Speed Traffic Forecasting

The latest revision expands the introduction to 22 distinct references, expands the related work to 34 distinct references, and clarifies the methodology. See `REVISION_NOTES.md` for the source checks and scope. Open `main.tex` for the clean manuscript or `marked.tex` for the version with this revision's edited prose in blue and bold. The `pdf/` folder contains both compiled versions.

Open `main.tex` as the main document in Overleaf, or compile with `latexmk -pdf main.tex` using a TeX installation with IEEEtran and TikZ. The ZIP includes all figure PDFs and the BibTeX database. Run `python make_paper_figs.py` with NumPy, pandas and Matplotlib to regenerate the plots. The geographic map is retained as a pre-rendered PDF from the submitted v5 source.

## Evidence used in the revision

| Report item | Source |
|---|---|
| Original greedy method and controls | [Lead–lag selection notebook](https://colab.research.google.com/drive/16l3yrAJ5FDYRQm_N_SypcI5QOvHhEA2t) and [graph construction notebook](https://colab.research.google.com/drive/1s8RZUjeLyHxSYYRiQUA1mEHw0kkBCp28) |
| Original real-data comparison | [Consolidated run scores](https://drive.google.com/file/d/1UhrcgbUnSSzK2uP4T3GJgd2d8v7-k08z/view). The package includes the real-data seed and metric columns used for figures and tables. |
| Event tolerance check and missing-input rerun | [Tolerance-check outputs](https://drive.google.com/file/d/1uETNTvKiMd6zQ_waTsU3c1vTSQDRGLUs/view) and [old versus fixed inputs](https://drive.google.com/file/d/1AoLXd06PzQU8RE2I3Q0Mi7pD9TVdke2U/view) |
| Baseline, six-way comparison, graphs and week-swap diagnostic | [GPU results archive](https://drive.google.com/file/d/1jbPhTXgqzZzLsLJDDRRv2n1tEzUzCzJB/view). Relevant CSVs are included in `data/`. |
| Ridge screen and graph construction | [Replication implementation](https://drive.google.com/file/d/14RdLX_C7YBokOOZbeJJAeq2e1PK8agf4/view) |
| Five-donor diagnostic | [Diagnostic JSON](https://drive.google.com/file/d/11YBeSC8u4KaCMlk7wFZOgQvB7x5-UM85/view), also included in `data/` |
| DCRNN and DCGRU | Per-seed `result.json` files from `other_model_output/dcrnn/` and `other_model_output/dcgru/` in the GPU results archive (`7_final_code_and_results/EL_upload.zip`), collected in `data/second_models.csv` |
| Tuning | `tuning_output/METR-LA/decisions/final_result.json` and `final_test_results.csv` in the same archive |

## Revision notes

The notes below record earlier versions. Figure and table numbers in them refer to the version named in each heading, not to the current report. For the current numbering, see the repository README.

The report uses METR-LA and PEMS-BAY. The original greedy selector and the later ridge-based selectors have distinct methods, tables, and interpretations. The supplementary DCGRU check is reported at the end of E6; its 12 run files are in the GPU results archive. Stage 2 ran on a single NVIDIA A10G GPU. No experiments were rerun for this revision.

## Additional results in v8

- Appendix A gives the six-way comparison at 15, 30 and 60 minutes and the mean over all 12 horizons. Each row reports MAE, RMSE and MAPE. The supplementary plot shows the MAE difference from the original graph at all 12 horizons.
- Appendix B explains the original jam-warning results and the aggregate DCRNN and DCGRU error metrics. Regenerate the tables with `python make_supplement.py` and figures with `python make_paper_figs.py`.
- `data/original_study.csv` contains the 42 retained real-data runs, including event and bottleneck-subset metrics. The companion files record the event tolerance and cutoff checks and the METR-LA missing-input rerun. Source is the [Coding folder](https://drive.google.com/drive/folders/1OWVQ7eGW9E88ix5R52rhIKALw5QEToJ_).
- The baseline, pilot and replication CSVs include all 12 horizons, 12-horizon means and all recorded speed bands. The seed summary, selected links and published baseline comparisons are also included.
- `second_models.csv` preserves the fields from the 24 DCRNN and DCGRU JSON files. `dcrnn_comparison.csv` additionally preserves the DCRNN horizon and runtime summaries.
- Tuning files contain the selection decisions, archived horizon outputs, and the per-seed scores and intervals from the locked test.
- The archived single-seed LargeST pilot, its destination check, and the separate hub sweep are in `data/largest_*`. These are exploratory real-data results, with different models or evaluation protocols from the main traffic-speed benchmarks. See the [pilot table](https://drive.google.com/file/d/1EQrQiAFtFraJ7R0nfzx1LufHVcMCxYlt/view) and [hub sweep](https://drive.google.com/file/d/15nzumAY-gQ8ftt4-zS6m85vWliZQwbeB/view).

## Merged revision

This revision combines the expanded related work and all-road jam-recall ranges with the horizon and event appendices and the detailed result exports. Common figures and shared numeric fields were checked against both v8 packages. The method text describes edge weights as conductances because the resistance calculation uses the weighted symmetrized adjacency.

## v10 additions

- Table I (notation) in Section III-A.
- Algorithm 1 (greedy lead–lag selector) and Algorithm 2 (ridge-based Express Links and Forecast-only), written from the project code (`05_leadlag_rho_and_express_links.ipynb` and `run_replication.py`).
- Fig. 1: a worked example of Algorithm 1 on a six-road illustration (two triangles and one bridge). It is the hand-checked example from the Stage 1 notebook, used only to explain the method. It is not a dataset.
- Equation (2): closed form of the drop in total effective resistance used by the greedy selector.
- Fig. 3 (`figs/stage1_tradeoff.pdf`): mean lead–lag strength vs. total resistance reduction for each Stage 1 rule, from `data/stage1_link_properties.csv` (the Stage 1 `versions_summary` files).
- Fig. 7 (`figs/link_properties.pdf`): pair resistance and held-out gain of the links chosen by each ridge-based selector, from `data/selected_links.csv`.
- Both new plots are produced by `make_paper_figs.py`. No experimental numbers changed.
- v10.1 layout fix: the references now come directly after the main text and before the one-column appendices. This removes a page that held only Fig. 11. A float barrier before the Discussion was also removed.

## v11 review

See `CLAIM_AUDIT.md` for the evidence behind the main conclusions and recommended next experiment. The model comparison, tuning heatmap, and final test interval figure now read the included numerical outputs. The other regenerated quantitative figures and appendices also derive from the package data.

## v12 prose revision

This edition uses plainer academic wording throughout the report. It uses the same experimental data, tables, plots, and references as v11.

## v15–v16 figures

Figs. 1–3 were added: the bottleneck motivation, the framework overview and the selector illustration. In v16, Fig. 2 separates the training-period data used for link selection from the 60-minute forecaster input, and labels the evaluation metrics by stage. In v17 it was simplified into two panels: (a) link selection and (b) forecasting and evaluation. Their TikZ sources are in `figs/source/`. No numbers changed.
