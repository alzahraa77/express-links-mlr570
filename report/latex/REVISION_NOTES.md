# v18: experimental setup to appendices

- Removed the calendar dates and clock times from the splits paragraph.
- Compute is described as a single NVIDIA A10G GPU only.
- Rewrote Experimental Setup: data sources for Stage 1 and Stage 2, Stage 1 paper split vs. Stage 2 strict split, model settings, seed use, bootstrap resample counts (1,000 for D1; 10,000 for D2 and the tuning test).
- Every table and figure caption is now one sentence; how to read each one moved into the text just before it.
- Each table and figure is placed right after the paragraph that first mentions it (two-column floats can only sit at the top of a page, so they appear at the top of the next page).
- Fixed: code name `express_fixed` in E1 prose, confusing "fixed-budget variants" note in the trade-off caption, "second addendum committed" jargon in the DCGRU paragraph, repeated common-test-readings sentence in E6.
- Appendix captions shortened the same way (also in make_supplement.py).
- Conclusion expanded to five paragraphs (question, Stage 1, Stage 2, diagnostics, takeaway).
- Removed the test-specification date and the bootstrap seed numbers from the diagnostics method.
- Added Tori et al. (ICLR 2025) and Attali et al. (2026 survey) to the Discussion; one matching sentence (no citation) in the Conclusion.
- Plain-language pass on the method: lead-lag strength, effective resistance, the bottleneck screen, every Stage 1 and Stage 2 graph (including the virtual node), and the diagnostics are now explained where they first appear. Abstract and introduction glossed the same terms.
- Figs. 1-5 captions shortened to one sentence; their explanations moved into the text. Added why DCRNN was run on METR-LA only.
- Wording fixes: abstract Stage 1 description, resistance sentence in the conclusion, "regularized ridge objective" for the Forecast-only shortlist, "shared evaluation roads", and a conditional wording for the rewiring survey.
- Added GitHub link (abstract and new Data Availability section), a paper-organization paragraph, Limitations and Future Work split into two subsections with bold topics.
- Changes are in blue bold in the marked PDF (build marked.tex). No numbers changed and no experiments were run.

# Introduction, related work and methodology revision

The introduction now follows problem, motivation, approach and contributions. It cites 22 distinct sources. The related work has been expanded into thematic comparisons and includes a table explaining the relationship between representative methods and this project. Six references were added: T-GCN, STAEformer, DRew, Graph Diffusion Convolution, Graphormer and GraphGPS.

The methodology now states the forecasting map and graph-augmentation constraint, introduces the two selection procedures before their details, formalizes candidate eligibility, and explains training-only ridge fitting and ranking. It retains the initial greedy method and the later ridge-based variants as distinct procedures. The fixed hub control is explicitly distinguished from Cao et al.'s learned semi-adaptive model. Computational considerations are analytical descriptions, rather than new runtime measurements.

The experimental setup, results, discussion, limitations, conclusion and appendices are retained verbatim in the LaTeX source. Figures, numerical tables, data exports, diagnostic thresholds and tuning decisions were not changed. Reference and equation numbering may change because material was added earlier in the paper. No experiments were run for this revision.

## Writing structure examined

The revision uses the common progression of a task definition, a framework overview, component explanations with equations or algorithms, and a separate experimental setup. The wording is original to this report.

| Paper | Relevant writing structure | Primary source |
|---|---|---|
| DCRNN | Defines traffic forecasting before diffusion and recurrent architecture | https://arxiv.org/pdf/1707.01926 |
| Graph WaveNet | Introduces graph and forecasting definitions before spatial and temporal modules | https://arxiv.org/pdf/1906.00121 |
| GTR / effective resistance | Separates graph quantities, theoretical motivation and the rewiring procedure | https://proceedings.mlr.press/v202/black23a/black23a.pdf |
| Virtual nodes for traffic | Gives problem definitions and related work before the backbone and adjacency construction | https://arxiv.org/html/2501.10048v1 |
| STAEformer | Defines the prediction task, gives an overview, then explains modules before experiments | https://arxiv.org/html/2308.10425v5 |
| DRew | Defines message passing and graph notation before explaining the rewiring mechanism | https://proceedings.mlr.press/v202/gutteridge23a/gutteridge23a.pdf |

## Primary sources checked for expanded comparisons

| Source | Supported comparison | Link |
|---|---|---|
| T-GCN | Graph convolution with a GRU; journal issue in 2020, online publication in 2019 | https://doi.org/10.1109/TITS.2019.2935152 |
| STAEformer | Adaptive embeddings with temporal and spatial Transformer layers | https://doi.org/10.1145/3583780.3615160 |
| DRew | Layer-dependent rewiring with distance-dependent delay | https://proceedings.mlr.press/v202/gutteridge23a.html |
| Graph Diffusion Convolution | Generalized diffusion and a sparsified communication graph | https://proceedings.neurips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html |
| Graphormer | Structural encodings for a graph Transformer | https://proceedings.neurips.cc/paper_files/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html |
| GraphGPS | Positional or structural encodings, local message passing and global attention | https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html |
| MTGNN | Jointly learned directed dependencies | https://arxiv.org/abs/2005.11650 |
| GTS | Distribution over discrete graphs | https://arxiv.org/abs/2101.06861 |
| AGCRN | Adaptive graph generation and node-specific parameters | https://arxiv.org/abs/2007.02842 |
| ASTGCN | Spatial and temporal attention | https://cdn.aaai.org/ojs/3881/3881-13-6940-1-10-20190702.pdf |
| GMAN | Spatial and temporal attention in an encoder-decoder | https://arxiv.org/abs/1911.08415 |
| STFGNN | Temporal-graph construction and fusion with spatial relationships | https://cdn.aaai.org/ojs/16542/16542-13-20036-1-2-20210518.pdf |
| STID | Sensor and time identity with an MLP | https://arxiv.org/abs/2208.05233 |
| PDFormer | Delay-aware transformation and short- and long-range attention | https://arxiv.org/abs/2301.07945 |
| FoSR | Spectral expansion, relational treatment of added edges, classification evaluation | https://arxiv.org/abs/2210.11790 |
| DiffWire | Commute-time and spectral-gap layers | https://proceedings.mlr.press/v198/arnaiz-rodri-guez22a.html |
| Expander Graph Propagation | Sparse expander communication structure | https://proceedings.mlr.press/v198/deac22a.html |
| LASER | Connectivity, locality and sparsity tradeoffs | https://proceedings.iclr.cc/paper_files/paper/2024/hash/7b7db41ea66d624587f211aa15c07e45-Abstract-Conference.html |
| Spatiotemporal over-squashing | Spatial and temporal information-propagation analysis | https://papers.nips.cc/paper_files/paper/2025/file/30cdb8f140341beda5f7a8767a4b102d-Paper-Conference.pdf |
| MPNN with virtual nodes | Approximation relationships with graph Transformers | https://proceedings.mlr.press/v202/cai23b.html |
| STGC-GNNs | Spatiotemporal alignment and Granger testing | https://arxiv.org/abs/2210.16789 |
| ST-CGI | Joint graph and autoregressive-process learning | https://kaichunf.github.io/papers/zhang2022granger.pdf |
| MSA-GCN | Granger testing within a multistage architecture | https://www.mdpi.com/2227-7390/12/21/3338 |

The comparison table identifies literature context separately from implemented controls. Citation counts refer to distinct bibliography keys, not repeated citation occurrences. The added prose does not assert a first-ever contribution, causal discovery, or demonstrated over-squashing reduction.

## Compile

Open `main.tex` in Overleaf for a clean version. Open `marked.tex` to compile a review version with this revision's edited prose in blue and bold. Both use the same methods, data and bibliography. Compile with PDFLaTeX and BibTeX, or with Tectonic.

## v15 additions

- Fig. 1 (Introduction): the motivating bottleneck idea on a small illustrative graph. The caption states that the diagnostics did not establish this mechanism.
- Fig. 2 (Framework overview): the pipeline from input to selector, rewired graph, forecaster, forecast and evaluation.
- Fig. 3 (Framework overview): the kinds of links each rule tends to add. Panel (b) shows one link crossing the bottleneck and one inside an area, because on METR-LA only 3 of the 25 Stage 1 links crossed the screened bottleneck.
- Restored the plain-language summary of the strongest result after the contribution list. It had been dropped in v14.
- The figure sources are in `figs/source/` (TikZ: `common.tex`, `A_before_after.tex`, `B_pipeline.tex`, `C_bottleneck.tex`).
- No numbers, tables or results changed.
