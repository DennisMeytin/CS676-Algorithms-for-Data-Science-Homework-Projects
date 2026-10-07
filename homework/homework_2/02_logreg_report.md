# Homework 02 — Logistic Regression

Fitted 300 observations (two overlapping Gaussian blobs) by batch
gradient descent on the log-loss, learning rate 0.15,
3000 iterations.

## Model

| term | coefficient |
| --- | --- |
| intercept | -0.5156 |
| x1 | +2.5183 |
| x2 | +1.4906 |

## Confusion matrix

| | predicted 0 | predicted 1 |
| --- | --- | --- |
| **actual 0** | 138 | 12 |
| **actual 1** | 11 | 139 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.9233 |
| precision | 0.9205 |
| recall | 0.9267 |
| f1 | 0.9236 |
| majority-class baseline | 0.5000 |
| log-loss, first iteration | 0.6931 |
| log-loss, last iteration | 0.1872 |

## Reading the numbers

Accuracy alone is not enough: always compare against the majority-class baseline
of 0.5000, which is what a model that ignores the features entirely would
score. The remaining errors sit where the two blobs overlap, and no straight line
can fix those — they are a property of the data, not of the fit.
