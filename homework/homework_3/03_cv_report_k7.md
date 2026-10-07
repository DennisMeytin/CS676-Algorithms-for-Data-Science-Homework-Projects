# Homework 03 — K-Fold Cross Validation

7-fold cross validation over 240 observations and 30 features,
of which only 3 actually drive the response. Model: ordinary least
squares, solved in closed form.

## Per-fold results

| fold | train RMSE | validation RMSE |
| --- | --- | --- |
| 1 | 1.9693 | 2.2415 |
| 2 | 1.8912 | 2.5536 |
| 3 | 1.9435 | 2.3351 |
| 4 | 1.9439 | 2.3364 |
| 5 | 1.9228 | 2.5254 |
| 6 | 1.9902 | 2.0132 |
| 7 | 1.9662 | 2.2109 |

## Summary

| quantity | value |
| --- | --- |
| mean validation RMSE | 2.3166 |
| sd across folds | 0.1868 |
| mean training RMSE | 1.9467 |
| resubstitution RMSE (all data) | 1.9734 |
| optimism | 0.3432 |
| noise sd used to generate y | 2.0000 |

## Reading the numbers

Training RMSE is lower than validation RMSE in essentially every fold. That gap
is the model fitting noise it has already seen — including the
27 features that are pure noise by construction.

The resubstitution RMSE, computed by fitting and scoring on the same rows, is the
most optimistic number available and the one most often reported by mistake. The
cross-validated figure is higher and is the honest one.

The spread across folds (0.1868) is worth as much attention as the
mean. A single train/test split would have handed you one draw from that spread
and no way to know how lucky it was.
