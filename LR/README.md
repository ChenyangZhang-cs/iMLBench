## Linear Regression (LR)

### Application Introduction

Linear Regression (LR) is a fundamental supervisedmachine-learning algorithm, which is used to build a linear model to fit the relationship between variables. Since modeling linear relationship has been extensively studied, LR’s results are stable and credible. LR has been widely applied in prediction, error reduction, and variation explanation. Our LR code is based on the implementation from [“Rodinia: A benchmark suite for heterogeneous computing,” in IISWC, 2009].

### Usage

- using `bash run.sh` to run application with CPU offset from 0 to 100, spaced by 10.

- linear [loops] [cpu_offset]
    
    loops is an integer which represents kernel execution time.

    cpu_offset is an integer which represents the CPU offset of CPU-GPU co-running. 10 means CPU executes 10% computational task. (default: 0)

    example:
    $ ./linear 100 10

### Expected Result
```
CPU offset: xx
Time: xxx ms
> TEMPERATURE REGRESSION (16298)

        Parallelized
        --------
        | R squared: 63%
        | Time: xxx ms
        | Equation: y = 0.798x + 0.036

        Iterative
        --------
        | R squared: 63%
        | Time: xxx ms
        | Equation: y = 0.798x + 0.036
```