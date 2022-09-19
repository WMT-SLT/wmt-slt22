#!/usr/bin/env bash -x

APPRAISE_PYTHON=./Appraise/venv/bin/python3
APPRAISE_ROOT=./Appraise

mkdir -p batches
prefix=batches/batches.slttest2022.sgg-deu

$APPRAISE_PYTHON scripts/combine.py \
    -i submissions/slttest2022.dsgs-de.refs-nrm-docids.xml \
    -o slttest2022.dsgs-de.all.xml \
    submissions/slttest2022.dsgs-de.dsgs-de.*.xml

$APPRAISE_PYTHON $APPRAISE_ROOT/create_wmt22_tasks.py \
    -f slttest2022.dsgs-de.all.xml -o $prefix -s sgg -t deu --rng-seed 1111 \
    --selected-docs submissions/slttest22-doc-snippets.tsv --static-context 5 --no-qc \
    | tee $prefix.log
