#!/bin/bash

for i in {0..10}
do
	./build/linear 50 $[i*10]
    sleep 10
done