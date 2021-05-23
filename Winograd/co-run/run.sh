#!/bin/bash

for i in {0..10}
do
		./WinogradConv2D $[i*10]
		sleep 60
done
