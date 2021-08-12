#!/bin/bash

python main.py --model=LiveRec \
               --dataset="dataset/" \
               --fr_ctx \
               --fr_rep \
               --model_to="liverec" \
               --device="cuda" \
               --caching 
