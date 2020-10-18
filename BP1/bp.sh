!/bin/bash

for i in {0..10}
do
        for j in {1..3}
        do
			./backprop $[819200] $[i*10]
			
        done
		sleep 60
done

