#!/bin/bash

for i in {0..10}
do
    ./nn filelist.txt -o $[i*10]
    sleep 60
done

