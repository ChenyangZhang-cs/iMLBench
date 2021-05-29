#!/bin/bash
for i in {0..10}
do
	./kmeans -i ../data/204800.txt -f $[i*10]
    sleep 60
done

