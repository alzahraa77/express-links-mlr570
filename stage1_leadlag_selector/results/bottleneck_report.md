# Bottleneck report: METR-LA and PEMS-BAY (Week 1, step 5)

**Question (proposal 3.2):** are the bottlenecks in each benchmark strong enough for real-data results to carry the conclusion, or does the synthetic data carry it?

## Decision

| | Decision | Why |
|---|---|---|
| **METR-LA** | **Strong, but local** | 9 sensors (4.4%) behind 3 links; R jump 4.17 (2.57 with every link = 1), narrower than a 1-crossing town |
| **PEMS-BAY** | **Weak (borderline)** | closest region (23 sensors): R jump 1.39, but 1.22 with every link = 1; line 1.43 |

## The numbers

| | METR-LA | PEMS-BAY |
|---|---|---|
| Sensors analysed (main part) | 206 of 207 | 319 of 325 |
| R: median / 99th percentile / max | 1.01 / 6.24 / 8.92 | 1.26 / 5.06 / 12.03 |
| Candidate regions (k = 2 to 8) | 15 | 15 |
| Bottleneck / mild / none | 1 / 0 / 14 | 0 / 8 / 7 |
| Strongest region: R jump (weights / every link = 1) | 4.17 / 2.57 | 1.39 / 1.22 |
| Sensors behind a bottleneck | 9 (4.4%) | 0 (0.0%) |
| Road pairs across a bottleneck | 1,773 (8.4%) | 0 (0.0%) |
| Dead ends (single link to 1-3 sensors) | 0 | 3 |
| Same decision under neighbouring rules | 100% of 6 | 67% of 6 |

![Decision scale](../figures/bottleneck_scale.png)

![R distribution](../figures/bottleneck_R_distribution.png)

![Bottleneck map](../figures/bottleneck_map.png)

## What it means for the project

- **METR-LA:** METR-LA has one real bottleneck: 9 sensors sit behind 3 road links, and reaching them costs 4.2x the R that distance explains (2.6x with every link = 1; narrower than a 1-crossing town). It is found at k = 4,5,6,7,8. It covers only 4% of the sensors, so METR-LA can test RQ2 on real traffic, but the "behind the bottleneck" subset is small and its jam recall will be noisy. The toy city stays the main evidence for RQ1, and Sharjah should stay in the core plan.
- **PEMS-BAY:** No region of PEMS-BAY passes the rule. The closest is a region of 23 sensors at k = 7,8: R jump 1.39 with link weights, but 1.22 with every link = 1 (line 1.43). PEMS-BAY has mild bottlenecks at most, comparable to a town with 2 to 4 crossings. As the proposal says, the synthetic data carries the main conclusion; PEMS-BAY shows what express links do where bottlenecks are mild, and a gain as large as on a strong bottleneck would point to a cause other than over-squashing.

## Method (state in the report)

- **R jump** of a region = median R of pairs with one road inside and one outside, divided by median R of pairs on the same side at the same number of road links, averaged over hop distances. It measures the extra cost of crossing, not distance.
- **Regions:** spectral partition of the two-way road graph (normalised Laplacian, k-means with 30 restarts, seed 0) at every k from 2 to 8; regions of 5 or more sensors and at most half the map.
- **Rule:** a region is behind a bottleneck if its R jump is at least that of a town of two 10 x 10 districts joined by 2 crossings (R jump 1.43; 1 crossing 1.80, 4 crossings 1.13, 8 crossings 1.02), with both link weightings, at 2 or more values of k. A benchmark is strong if any region passes; local if under 10% of its sensors are behind it.
- **How the rule was set:** a first pass used one k (the largest eigengap); its robustness check showed PEMS-BAY flipping with k, so the rule was tightened to the multi-scale version before the final numbers were read. The reference-town thresholds were not changed.
- **"Across a bottleneck"** (same-side control, behind-bottleneck subset): two roads are across a bottleneck if they are in different districts, i.e. on different sides of a bottleneck region. Saved as `district` and `behind_bottleneck` in `data/processed/<dataset>/bottleneck/regions.npz`.
- Road map only, no speed data, so no test information is used.