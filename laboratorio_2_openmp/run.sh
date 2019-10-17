#!/bin/bash
for threads in {1..50..1}
do
    echo "Iteracion: $threads"
    time ./bin/mandelbrotp -i 500 -a -1 -b -1 -c 1 -d 1 -s 0.001 -f ./results/parallel.raw -t $threads;
    echo  "-----------------"
done