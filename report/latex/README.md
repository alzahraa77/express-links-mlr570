# Sparse Sensor Links for Low-Speed Traffic Forecasting: LaTeX source

MLR 570 Advanced Machine Learning final report, American University of Sharjah.
Authors: Al Zahraa Altayasinah and Mohammed Alawadhi.

## Build

Open `main.tex` in Overleaf (IEEEtran class, pdfLaTeX + BibTeX), or run `latexmk -pdf main.tex`.
`Express_Links_Final_Report.pdf` is the compiled report.

## Contents

- `main.tex`, `sections/`: report text. `supplement_*.tex`: appendix tables.
- `refs.bib`, `main.bbl`: bibliography.
- `figs/`: figure PDFs; `figs/source/` holds the TikZ sources of the illustrations.
- `data/`: the saved results behind every plot and appendix table.
- `make_paper_figs.py`: redraws the plots from `data/` (needs NumPy, pandas, Matplotlib).
- `make_supplement.py`: rebuilds the appendix tables from `data/`.

The METR-LA link map (`figs/links_map.pdf`) is drawn by `report_figures/make_links_map.py` in the code repository:
https://github.com/alzahraa77/express-links-mlr570
