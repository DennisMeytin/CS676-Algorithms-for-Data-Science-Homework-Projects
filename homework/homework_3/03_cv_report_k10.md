# Homework 03 — K-Fold Cross Validation

10-fold cross validation over 240 observations and 30 features,
of which only 3 actually drive the response. Model: ordinary least
squares, solved in closed form.

## Per-fold results

| fold | train RMSE | validation RMSE |
| --- | --- | --- |
| 1 | 1.9619 | 2.2751 |
| 2 | 1.9750 | 2.1335 |
| 3 | 1.8906 | 2.7892 |
| 4 | 1.9733 | 2.1669 |
| 5 | 1.9310 | 2.4960 |
| 6 | 1.9691 | 2.1593 |
| 7 | 1.9571 | 2.3171 |
| 8 | 1.9891 | 1.9693 |
| 9 | 1.9674 | 2.1203 |
| 10 | 1.9562 | 2.2944 |

## Summary

| quantity | value |
| --- | --- |
| mean validation RMSE | 2.2721 |
| sd across folds | 0.2304 |
| mean training RMSE | 1.9571 |
| resubstitution RMSE (all data) | 1.9734 |
| optimism | 0.2987 |
| noise sd used to generate y | 2.0000 |

## Reading the numbers

Training RMSE is lower than validation RMSE in essentially every fold. That gap
is the model fitting noise it has already seen — including the
27 features that are pure noise by construction.

The resubstitution RMSE, computed by fitting and scoring on the same rows, is the
most optimistic number available and the one most often reported by mistake. The
cross-validated figure is higher and is the honest one.

The spread across folds (0.2304) is worth as much attention as the
mean. A single train/test split would have handed you one draw from that spread
and no way to know how lucky it was.
