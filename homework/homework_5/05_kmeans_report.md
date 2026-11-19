# Homework 05 — K-Means Clustering

Lloyd's algorithm on 270 unlabelled points, k = 3, run until the
assignments stopped changing (3 passes).

## Clusters found

| cluster | size | centroid |
| --- | --- | --- |
| 0 | 90 | (-2.818, -1.986) |
| 1 | 90 | (+0.438, +3.229) |
| 2 | 90 | (+3.658, -1.406) |

## Convergence

| quantity | value |
| --- | --- |
| inertia, first pass | 409.12 |
| inertia, final pass | 397.60 |
| passes to convergence | 3 |
| purity vs the true blobs | 1.0000 |

## Reading the numbers

Inertia falls on every pass and then stops. That is guaranteed: both steps of the
algorithm can only lower it, so it decreases monotonically until the assignments
repeat. Inertia rising is not a tuning problem — it means assign and update ran in
the wrong order.

Purity of 1.0000 compares the clusters against the blobs the data was generated
from. The algorithm never saw those, and it never sees anything that tells it it
is right — which is what makes this unsupervised.

## Two things to try

Change the seed and re-run. Random initialisation sometimes drops two centroids
into one blob and none into another, and the result is visibly wrong even though
the code is correct. That failure is what k-means++ was invented to avoid.

Then sweep k from 1 to 8 and plot the final inertia against k. It falls every
time, because more clusters always fit better — which is precisely why you cannot
pick k by minimising it.
