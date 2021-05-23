#!/bin/bash
make clean
make

for i in {0..10}
do
	./nn filelist -o $[i*10]
done

