#!/bin/bash

for i in {0..10}
do
        for j in {1..3}
        do
			./build/OpenCLNet MLP /off $[i*10]
                        sleep 30
        done
		sleep 100
done

