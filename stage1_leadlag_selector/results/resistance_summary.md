# Effective resistance summary (Week 1, step 4)

R computed on the two-way road graph (weight = max of the two directions, self-links removed), each link conducting its road-graph weight. R uses the road map only, no speed data. Exact pseudoinverse per connected part; checked against hand-computed examples, NetworkX and Foster's theorem.

| | METR-LA | PEMS-BAY |
|---|---|---|
| Sensors | 207 | 325 |
| Two-way road links | 1,313 | 2,079 |
| Connected parts | 2 | 7 |
| Sensors in the main part | 206 | 319 |
| R, median | 1.01 | 1.26 |
| R, 90th percentile | 2.82 | 1.95 |
| R, 99th percentile | 6.24 | 5.06 |
| R, maximum | 8.92 | 12.03 |
| 99th percentile / median | 6.2x | 4.0x |
| Road links between two sensors, median / max | 5 / 13 | 5 / 13 |
| Links with bridge score >= 0.5 | 18 | 10 |
| True bridges (score 1) | 0 | 3 |
| Spearman, weighted vs unweighted R | 0.986 | 0.952 |

- **METR-LA**: 1 cut-off sensor(s) with no link to any other sensor (717804); R = inf for them, leave them out of the express-link candidates.
- **PEMS-BAY**: 6 cut-off sensor(s) with no link to any other sensor (400001, 400017, 400240, 400560, 400677, 407202); R = inf for them, leave them out of the express-link candidates.

Bridge score of a link = weight x R across it: the share of the flow between its two ends that goes through that link (1 = a true bridge). Files: `data/processed/<dataset>/resistance/`.