# Homework 04 — Decision Tree (fit 1)

Grew a classification tree with the `SimpleDecisionTreeClassifier` from the
course notes on 240 synthetic points in two dimensions, holding out
30% for testing. Class spread for this fit: class 0 sd
0.9, class-1 clumps sd 0.8 (larger means more overlap).

## Results

| quantity | value |
| --- | --- |
| training accuracy | 1.0000 |
| test accuracy | 1.0000 |
| majority-class baseline | 0.5139 |
| leaves | 5 |
| depth | 4 |

## Reading the numbers

Training accuracy is 1.0000. That is not a sign of a good model — this
tree splits until every leaf is pure, so it can always memorise its training set.
Any tree grown without a stopping rule will report roughly 1.0 here. The test
accuracy of 1.0000 is the only number that says anything about
generalisation.

The tree has 5 leaves at depth 4. Each leaf is a rectangle in
feature space, because every split compares one feature against one threshold.
That is why the decision regions have square corners, and why a tree needs a
staircase of splits to approximate a diagonal boundary that logistic regression
would capture with a single line.

## Something to try

Add a stopping rule — refuse to split a node with fewer than, say, five rows, or
cap the depth. Watch training accuracy fall and test accuracy hold steady or
improve. That gap closing is exactly what pruning is for.
