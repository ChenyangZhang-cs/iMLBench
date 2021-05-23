## K Nearest Neighbor (KNN)

### Application Introduction

K Nearest Neighbor (KNN) is a non-parametric al- gorithm used for classification and regression in instance- based learning or lazy learning domain. When KNN is used as a classification, objects are classified by its k nearest neighbors’ categories, and the output of the algorithm is the classification result. KNN can also be used as a regression method, and the result is the value of a particular object, determined by the mean values of its k nearest neighbors. The original KNN code is from [“Rodinia: A benchmark suite for heterogeneous computing,” in IISWC, 2009].

This application computes the nearest location to a specific latitude and longitude for a number of hurricanes (data from: http://weather.unisys.com/hurricane/).

### Usage

nn [filename] -o [int] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]

example:
$ ./nn filelist -o 10

- filename     the filename that lists the data input files

- -o [int]     the CPU offset of CPU-GPU co-running. 10 means CPU executes 10% computational task. (default: 0)

- -r [int]     the number of records to return (default: 10)

- -lat [float] the latitude for nearest neighbors (default: 0)

- -lng [float] the longitude for nearest neighbors (default: 0)

- -h, --help   Display the help file

- -q           Quiet mode. Suppress all text output.

- -t           Print timing information.

- -p [int]     Choose the platform (must choose both platform and device)

- -d [int]     Choose the device (must choose both platform and device)


Notes: 1. The filename is required as the first parameter.
       2. If you declare either the device or the platform,
          you must declare both.

### Expected result

```
1980  3 12  0  2 SANDY       7.1   1.2  107   38 --> Distance=7.200694
1985 12 11  0 18 DEBBY       7.5   0.7   93  501 --> Distance=7.532596
1986 10 13 12 24 HELENE      7.5   1.3  113  437 --> Distance=7.611833
1958 10 25  6 20 OSCAR       7.8   0.5   57  540 --> Distance=7.816009
1973  2 26  6 28 DEBBY       7.6   2.7   35  654 --> Distance=8.065358
1976 11  7  0 20 JOYCE       7.6   3.2   79  131 --> Distance=8.246211
1965  3 17  0 14 ERNESTO     8.3   0.3  160  680 --> Distance=8.305420
2000  9 23  6  8 HELENE      8.4   0.2   51  286 --> Distance=8.402381
1994  5 24 18 13 TONY        8.1   2.5   35  761 --> Distance=8.477028
1985 11 24 18 27 KIRK        7.7   3.7  103  537 --> Distance=8.542833
```