#!/bin/bash

for i in {0..10}
do
        for j in {1..3}
        do
			./kmeans -i ../kmeansData/204800.txt -f $[i*10]
        done
        sleep 60
done

