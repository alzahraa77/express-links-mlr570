"""Load the setup part of run_replication.py without running any experiment.

run_replication.py is executed unchanged up to the line `dataset = args.dataset`, which is where
its own experiment starts. That defines the same data checks, strict preprocessing, batching,
frozen-graph loader, graph statistics and Graph WaveNet code the replication used. Nothing in
run_replication.py is modified, and its experiment section is never run.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "\ndataset = args.dataset\n"


def load(dataset, project_dir, cache, profile="full"):
    path = HERE / "run_replication.py"
    source = path.read_text()
    assert source.count(MARKER) == 1, "Unexpected run_replication.py revision"
    setup = source[: source.index(MARKER)]
    namespace = {"__name__": "replication_setup", "__file__": str(path)}
    # Local CPU testing only: EL_SETUP_PROFILE=smoke skips the GPU requirement of the "full" setup.
    profile = os.environ.get("EL_SETUP_PROFILE", profile)
    saved = sys.argv
    sys.argv = [str(path), "--dataset", dataset, "--profile", profile, "--mode", "original", "--seed", "0",
                "--project-dir", str(project_dir), "--cache", str(cache)]
    try:
        exec(compile(setup, str(path), "exec"), namespace)
    finally:
        sys.argv = saved
    return namespace


def prepare(R, dataset, scratch_dir):
    """Validated source data, strict preprocessing and the frozen graphs, as in the replication."""
    h5, graph_name, *_ = R["SPECS"][dataset]
    source = R["validate_data"](dataset, R["data_dir"] / h5, R["data_dir"] / graph_name)
    prepared = R["prepare_dataset"](dataset, source, "strict", Path(scratch_dir))
    folder = R["OUTPUT"] / "graphs" / dataset
    graphs = R["load_frozen_graphs"](folder, source[-1], prepared["adj"])
    targets = [prepared["ids"][b] for _, b, _ in graphs["express_links"][2]]
    return source, prepared, graphs, targets
