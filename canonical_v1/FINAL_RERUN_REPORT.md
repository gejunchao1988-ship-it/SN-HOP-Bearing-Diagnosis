# SN-HOP canonical reproducible rerun

Verification passed: **True**.

## Main strict DG results

| method        | direction   |   balanced_accuracy_pct |   macro_f1_pct |   task_sd_pp |
|:--------------|:------------|------------------------:|---------------:|-------------:|
| absolute      | 1500_to_900 |                 35.8333 |        26.9753 |     17.1279  |
| absolute      | 900_to_1500 |                 31.4167 |        23.3612 |     15.2856  |
| absolute      | combined    |                 33.625  |        25.1683 |     15.9616  |
| order         | 1500_to_900 |                 45.3333 |        37.425  |      9.8617  |
| order         | 900_to_1500 |                 37.3333 |        23.8603 |      8.75595 |
| order         | combined    |                 41.3333 |        30.6427 |      9.96118 |
| traditional23 | 1500_to_900 |                 52.5833 |        43.1673 |     13.0375  |
| traditional23 | 900_to_1500 |                 42.6667 |        34.1032 |     12.4276  |
| traditional23 | combined    |                 47.625  |        38.6352 |     13.3997  |
| pfn           | 1500_to_900 |                 60.5    |        56.2052 |     14.0447  |
| pfn           | 900_to_1500 |                 50.3333 |        46.8704 |     18.2244  |
| pfn           | combined    |                 55.4167 |        51.5378 |     16.6721  |
| snhop         | 1500_to_900 |                 55.8333 |        51.3512 |     13.4543  |
| snhop         | 900_to_1500 |                 57.5    |        50.8681 |     18.1344  |
| snhop         | combined    |                 56.6667 |        51.1096 |     15.5644  |

## Paired comparisons vs SN-HOP

| baseline      |   gain_pp |   wins |   ties |   losses |       raw_p |      holm_p |
|:--------------|----------:|-------:|-------:|---------:|------------:|------------:|
| absolute      |  23.0417  |     17 |      1 |        2 | 0.000170778 | 0.000683113 |
| order         |  15.3333  |     14 |      2 |        4 | 0.000923315 | 0.00276995  |
| traditional23 |   9.04167 |     15 |      1 |        4 | 0.0320612   | 0.0641224   |
| pfn           |   1.25    |     11 |      0 |        9 | 0.389731    | 0.389731    |

## Ablation

| variant          |   target_ba |   target_f1 |
|:-----------------|------------:|------------:|
| context_only     |     41.8333 |     33.3625 |
| contrast_context |     51.8333 |     46.0878 |
| full             |     56.6667 |     51.1096 |
| harmonics_only   |     53      |     45.5469 |
| peak_context     |     49.5833 |     42.9757 |
| uniform_centers  |     42.3333 |     31.277  |

## UDA

| method   | kind            | variant                |   k |   target_ba |   target_f1 |
|:---------|:----------------|:-----------------------|----:|------------:|------------:|
| absolute | adaptation_size | CORAL                  |   1 |     36.8333 |     32.1599 |
| absolute | adaptation_size | CORAL                  |   3 |     34.6667 |     31.7768 |
| absolute | adaptation_size | CORAL                  |   5 |     36.6667 |     34.6622 |
| absolute | adaptation_size | CORAL                  |  10 |     37.5    |     35.3827 |
| absolute | adaptation_size | CORAL                  |  15 |     37      |     34.7991 |
| absolute | adaptation_size | NoDA                   |   0 |     33.8333 |     25.77   |
| absolute | uda             | CORAL                  |  10 |     37.75   |     35.7663 |
| absolute | uda             | NoDA                   |  10 |     33.3333 |     24.9173 |
| absolute | uda             | target_standardization |  10 |     39.25   |     36.7636 |
| order    | adaptation_size | CORAL                  |   1 |     56.3333 |     53.793  |
| order    | adaptation_size | CORAL                  |   3 |     55.3333 |     53.6383 |
| order    | adaptation_size | CORAL                  |   5 |     55.8333 |     54.2894 |
| order    | adaptation_size | CORAL                  |  10 |     53      |     51.6175 |
| order    | adaptation_size | CORAL                  |  15 |     52.1667 |     50.6644 |
| order    | adaptation_size | NoDA                   |   0 |     41.5    |     30.82   |
| order    | uda             | CORAL                  |  10 |     53.4167 |     51.8527 |
| order    | uda             | NoDA                   |  10 |     42.1667 |     31.5516 |
| order    | uda             | target_standardization |  10 |     52.0833 |     48.6237 |
| pfn      | adaptation_size | CORAL                  |   1 |     62.1667 |     61.1574 |
| pfn      | adaptation_size | CORAL                  |   3 |     60.1667 |     59.5106 |
| pfn      | adaptation_size | CORAL                  |   5 |     61      |     60.084  |
| pfn      | adaptation_size | CORAL                  |  10 |     61.1667 |     60.6638 |
| pfn      | adaptation_size | CORAL                  |  15 |     60.8333 |     60.4879 |
| pfn      | adaptation_size | NoDA                   |   0 |     56      |     52.1468 |
| pfn      | uda             | CORAL                  |  10 |     61.75   |     61.3601 |
| pfn      | uda             | NoDA                   |  10 |     55.8333 |     51.953  |
| pfn      | uda             | target_standardization |  10 |     64.0833 |     62.8357 |
| snhop    | adaptation_size | CORAL                  |   1 |     65.3333 |     63.0244 |
| snhop    | adaptation_size | CORAL                  |   3 |     64.3333 |     62.3056 |
| snhop    | adaptation_size | CORAL                  |   5 |     64.3333 |     61.9452 |
| snhop    | adaptation_size | CORAL                  |  10 |     63.8333 |     61.4436 |
| snhop    | adaptation_size | CORAL                  |  15 |     64      |     61.5809 |
| snhop    | adaptation_size | NoDA                   |   0 |     58.8333 |     53.5399 |
| snhop    | uda             | CORAL                  |  10 |     62.4167 |     60.3562 |
| snhop    | uda             | NoDA                   |  10 |     57.75   |     52.3281 |
| snhop    | uda             | target_standardization |  10 |     62.5    |     61.1654 |

## Noise robustness

| snr   |   ('target_ba', 'mean') |   ('target_ba', 'std') |   ('target_f1', 'mean') |   ('target_f1', 'std') |
|:------|------------------------:|-----------------------:|------------------------:|-----------------------:|
| 0     |                 53.6944 |                13.5064 |                 47.914  |                17.6643 |
| 10    |                 56.8056 |                14.546  |                 51.3839 |                19.204  |
| 20    |                 56.7083 |                15.1898 |                 51.1844 |                19.6647 |
| 3     |                 56.3333 |                13.2263 |                 50.8785 |                17.9601 |
| 6     |                 57.5556 |                13.7829 |                 52.0768 |                18.7598 |
| clean |                 56.6667 |                15.5644 |                 51.1096 |                20.1187 |

## Class recalls

Strict DG recalls (%): Healthy 54.50, Outer 47.25, Inner 68.25.
SN-HOP+CORAL recalls (%): Healthy 61.25, Outer 44.00, Inner 82.00.

## Selected SN-HOP configurations

Half-width counts: {0.2: 14, 0.4: 3, 0.6: 3}
Maximum-order counts: {15: 9, 25: 6, 35: 2, 50: 3}
Feature dimension range: 26-64; median 38.

## Timing

|   halfwidth |   max_order |   feature_dim |   ms_per_600_mean |   ms_per_600_sd |
|------------:|------------:|--------------:|------------------:|----------------:|
|         0.2 |          15 |            26 |           1.94983 |        0.170809 |
|         0.2 |          35 |            48 |           2.84495 |        0.475241 |
|         0.4 |          15 |            26 |           1.99526 |        0.185044 |
|         0.6 |          50 |            64 |           3.4958  |        0.692211 |
|         0.6 |          25 |            38 |           2.32993 |        0.109139 |
|         0.4 |          50 |            64 |           3.35017 |        0.241264 |
|         0.4 |          25 |            38 |           2.29387 |        0.113624 |
|         0.2 |          25 |            38 |           2.27675 |        0.126757 |

## Provenance

This directory is the canonical rerun implementation used for the updated manuscript. It was reconstructed after the earlier Stage G/H execution source was not preserved. Reported updated results are generated by the code and saved artifacts in this package; they are not back-filled from the historical manuscript values.