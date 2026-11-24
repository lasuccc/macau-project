# How the sample weights work (numerical example)

Each client's weights count in proportion to its number of training windows, n_k.
New weight = sum(n_k x weight_k) / sum(n_k).

## Toy example
Two models have the value 2 and 8. Model A has 1 window, model B has 3 windows.
(1 x 2 + 3 x 8) / (1 + 3) = 26 / 4 = 6.5.
A plain mean would give 5. The bigger client pulls the result toward its own value.
The check in aggregation_checks.txt confirms the code returns 6.5.

## Real example, round 1 of my 10-round run
Clients picked: 66, 68, 78, 86, 91, 95, 100, 109, 113, 116
Their window counts: 1440, 1440, 1440, 1438, 1438, 1439, 1440, 1440, 1440, 1437
Total windows: 14392
Each client's weight is its count divided by the total. They range from 0.0998 to 0.1001.
A plain mean would give every client 0.1000.
The weights are almost equal because every client has between about 1437 and 1440 windows.
So on this data the weighted average is close to the plain average. The 6.5 test is what shows the weighting is coded correctly.

The weights use the number of unique windows, not the number of batches and not the number of local epochs.
Only the clients picked in that round are in the sum and in the denominator.
