#!/bin/bash

for i in {0..10}
do
		./3D $[i*10]
		sleep 60
done

