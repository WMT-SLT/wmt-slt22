#!/usr/bin/env bash

python scripts/ComputeWMTSLT22Results.py --separate-domains --task-type Document --csv-file scores/aggregated/all.csv foo | tee ranking.log
