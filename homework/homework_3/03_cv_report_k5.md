# Homework 03 — K-Fold Cross Validation

5-fold cross validation over 240 observations and 30 features,
of which only 3 actually drive the response. Model: ordinary least
squares, solved in closed form.

## Per-fold results

| fold | train RMSE | validation RMSE |
| --- | --- | --- |
| 1 | 1.9547 | 2.3163 |
| 2 | 1.8772 | 2.5597 |
| 3 | 1.9156 | 2.3945 |
| 4 | 1.9745 | 2.1465 |
| 5 | 1.9494 | 2.2090 |

## Summary

| quantity | value |
| --- | --- |
| mean validation RMSE | 2.3252 |
| sd across folds | 0.1623 |
| mean training RMSE | 1.9343 |
| resubstitution RMSE (all data) | 1.9734 |
| optimism | 0.3518 |
| noise sd used to generate y | 2.0000 |

## Reading the numbers

Training RMSE is lower than validation RMSE in essentially every fold. That gap
is the model fitting noise it has already seen — including the
27 features that are pure noise by construction.

The resubstitution RMSE, computed by fitting and scoring on the same rows, is the
most optimistic number available and the one most often reported by mistake. The
cross-validated figure is higher and is the honest one.

The spread across folds (0.1623) is worth as much attention as the
mean. A single train/test split would have handed you one draw from that spread
and no way to know how lucky it was.
