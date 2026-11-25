# Data summary (Week 1, step 2)

Input: past 12 readings (1 hour) + time of day. Output: next 12 readings (1 hour). Split 70/10/20 by time, same procedure as Graph WaveNet. Zero readings are kept as 0 and masked in the loss and metrics.

## METR-LA

- Sensors: 207; timestamps: 34,272 (2012-03-01 00:00 to 2012-06-27 23:55)
- Zero (masked) readings in the raw file: 8.11%
- Last reading training may use (for deseasonalising and rho in week 2): 2012-05-23 07:40

| Split | Samples | First anchor | Last anchor | x shape | y shape | Masked inputs | Masked targets |
|---|---|---|---|---|---|---|---|
| train | 23,974 | 2012-03-01 00:55 | 2012-05-23 06:40 | (23974, 12, 207, 2) | (23974, 12, 207, 2) | 7.13% | 7.13% |
| val | 3,425 | 2012-05-23 06:45 | 2012-06-04 04:05 | (3425, 12, 207, 2) | (3425, 12, 207, 2) | 6.96% | 6.97% |
| test | 6,850 | 2012-06-04 04:10 | 2012-06-27 22:55 | (6850, 12, 207, 2) | (6850, 12, 207, 2) | 12.11% | 12.11% |

## PEMS-BAY

- Sensors: 325; timestamps: 52,116 (2017-01-01 00:00 to 2017-06-30 23:55)
- Zero (masked) readings in the raw file: 0.00%
- Last reading training may use (for deseasonalising and rho in week 2): 2017-05-07 17:35

| Split | Samples | First anchor | Last anchor | x shape | y shape | Masked inputs | Masked targets |
|---|---|---|---|---|---|---|---|
| train | 36,465 | 2017-01-01 00:55 | 2017-05-07 16:35 | (36465, 12, 325, 2) | (36465, 12, 325, 2) | 0.00% | 0.00% |
| val | 5,209 | 2017-05-07 16:40 | 2017-05-25 18:40 | (5209, 12, 325, 2) | (5209, 12, 325, 2) | 0.01% | 0.01% |
| test | 10,419 | 2017-05-25 18:45 | 2017-06-30 22:55 | (10419, 12, 325, 2) | (10419, 12, 325, 2) | 0.01% | 0.01% |

"Anchor" is the last observed reading of a sample: its input covers the 55 minutes before it, its output the 60 minutes after it.