#!/usr/bin/env bash

scripts/ComputeWMT21Results.py --separate-domains --task-type Document --csv-file scores/aggregated/all.csv foo | tee ranking.log
