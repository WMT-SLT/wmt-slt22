#!/usr/bin/env bash

APPRAISE_PYTHON=./Appraise/venv/bin/python3
APPRAISE_ROOT=./Appraise

$APPRAISE_PYTHON $APPRAISE_ROOT/manage.py ComputeWMT21Results --task-type Document --csv-file <(cat scores/*.csv) foo | tee ranking.log
