## K-means (KM) 

### Application Introduction

K-means (KM) is a clustering algorithm that is used to learn features in either (semi-)supervised learning or unsupervised learning. It partitions a number of items into k clusters, which minimizes the mean distance between items and cluster centers. In each iteration, the cluster centers will be repositioned until the centers do not change that much. Our KM code is based on the implementation from [“Rodinia: A benchmark suite for heterogeneous computing,” in IISWC, 2009].

### Usage

- using `bash run.sh` to run the application with CPU offset from 0 to 100, spaced by 10.

- kmeans

    -i filename      :file containing data to be clustered

    -f               :cpu offset                            [default=0]

    -m max_nclusters :maximum number of clusters allowed    [default=5]
    
    -n min_nclusters :minimum number of clusters allowed    [default=5]

    -t threshold     :threshold value                       [default=0.001]

    -l nloops        :iteration for each number of clusters [default=1]

    -b               :input file is in binary format

    -r               :calculate RMSE                        [default=off]

    -o               :output cluster center coordinates     [default=off]

    **example:**
    $ ./kmeans -i ../data/204800.txt -f $[i*10]

### Expected Result
run with `-o` to see the result.

```
================= Centroid Coordinates =================
0: 199.48 43.54 396.69 0.02 0.00 0.01 0.00 0.01 0.00 0.00 0.02 0.00 0.00 0.00 0.02 141.01 10.12 0.51 0.50 0.25 0.25 0.31 0.06 0.02 236.65 51.26 0.21 0.10 0.08 0.01 0.50 0.50 0.26 0.25

1: 2.25 2183339.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.75 1.75 0.00 0.00 0.00 0.00 1.00 0.00 0.00 148.25 61.38 0.25 0.05 0.26 0.03 0.12 0.00 0.00 0.00

2: 0.00 521.33 0.57 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.00 0.00 0.00 490.71 490.71 0.00 0.00 0.00 0.00 1.00 0.00 0.00 254.31 254.04 1.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00

3: 29.69 489.77 1137285.25 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.69 2.69 0.04 0.01 0.00 0.00 1.00 0.00 0.01 139.54 72.31 0.33 0.03 0.01 0.05 0.00 0.00 0.02 0.04

4: 0.30 1285.06 32.05 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 485.85 485.89 0.00 0.00 0.00 0.00 1.00 0.00 0.02 248.78 247.80 0.98 0.00 0.96 0.00 0.00 0.00 0.00 0.00

CPU offset: xx
Time: xxx ms
```