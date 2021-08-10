#!/bin/bash

python main.py --model=LiveRec \
               --dataset=/mnt/localdata/rappaz/twitch/data/v3/full/ \
               --fr_ctx \
               --fr_rep \
               --model_to="test" \
               --device="cuda:2" \
               --caching 
