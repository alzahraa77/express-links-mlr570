# Express Links · METR-LA pilot

Open `Express_Links_Step2_Pilot.ipynb` in the PyTorch Jupyter kernel on the lab's GPU node. Choose a writable `PROJECT_DIR`. To reuse an existing dataset cache, change `CACHE` in the configuration cell to that folder. You can also point `MANUAL_DATA_DIR` to the existing HDF5 files. The notebook writes results under `PROJECT_DIR/pilot_output`.

Start with `PROFILE = "smoke"` and run all cells. After it succeeds, use `PROFILE = "full"` for the actual 100-epoch pilot. You can select a subset with `MODES`, though the full comparison requires all six. Leave `SEEDS = [0]` for the first pass. The notebook retrains the original graph as a paired baseline under the same run code. Checkpoints save after each epoch.

The prebuilt graphs and link lists in `METR-LA/` were generated from the Torch Spatiotemporal research mirror used to verify Step 1. The notebook regenerates the graphs from whichever dataset copy it actually loads, logs fingerprints, and saves the new versions in `pilot_output/graphs/METR-LA/`. Use graphs and model results from the same run. Treat smoke scores as pipeline checks.

Read `pilot_output/full/comparison.csv` for 60-minute overall, lowest-speed-band and shared target-sensor MAE. The most informative comparisons are against the original graph and the forecast-only selector. The adapter for GTR and the fixed-weight virtual hub share the Graph WaveNet backbone. They are not exact reproductions of their papers. The hub's arc count is reported separately. The full pilot must finish before we can claim an accuracy gain.
