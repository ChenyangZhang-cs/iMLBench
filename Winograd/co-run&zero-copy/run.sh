#!/bin/bash

for i in {0..10}
do
		./WinogradConv2D 3 $[i*10]
		sleep 60
done
