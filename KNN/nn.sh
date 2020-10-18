#!/bin/bash

for i in {0..10}
do
        for j in {1..3}
        do
			./nn filelist.txt -o $[i*10]
        done
		sleep 60
done

