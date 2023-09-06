#!/bin/bash

# launch the first instance
CUDA_VISIBLE_DEVICES=1 python launch.py --validation_folds 1 2 3 &
pid1=$!
sleep 1
# launch the second instance
CUDA_VISIBLE_DEVICES=2 python launch.py --validation_folds 4 5 &
pid2=$!
sleep 1
# launch the third instance
CUDA_VISIBLE_DEVICES=3 python launch.py --validation_folds 6 7 &
pid3=$!

# wait for all instances to finish
wait $pid1
wait $pid2
wait $pid3
