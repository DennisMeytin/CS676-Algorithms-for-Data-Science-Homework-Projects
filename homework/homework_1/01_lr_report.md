# Homework 01 — Linear Regression

Fitted 200 observations by batch gradient descent
(learning rate 0.05, 2000 iterations), then checked the
result against the closed-form least squares solution.

## Coefficients

| term | true | gradient descent | closed form |
| --- | --- | --- | --- |
| intercept | 3.0000 | 3.0836 | 3.0836 |
| x1 | 2.5000 | 2.4054 | 2.4054 |
| x2 | -1.2000 | -1.0250 | -1.0250 |

## Fit

| metric | value |
| --- | --- |
| MSE | 2.2209 |
| R-squared | 0.7516 |
| loss at first iteration | 18.7537 |
| loss at last iteration | 2.2209 |
| max gap vs closed form | 0.000000 |

## Reading the numbers

The MSE settles near 2.25, which is the variance of the noise we
added. No model can do better than that on this data — the remaining error is not
a modelling failure, it is the noise itself.

The gradient-descent and closed-form coefficients agree to 0.000000, so the loop
is finding the same optimum the algebra does.
