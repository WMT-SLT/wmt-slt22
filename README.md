# WMT22 SignLT Human evaluation

This repository contains data and scripts for reproducing human evaluation
settings and results for WMT22 Sign Language Translation Task.

## Content

Scripts:
* `generate-snippets.sh` - splits documents into 10-segment long chunks
* `generate-batches.sh` - produces batches for Appraise
* `generate-ranking.sh` - computes ranking from scores exported from Appraise

Data:
* `slttest2022.dsgs-de.all.xml` - the official test set
* `submissions/*.xml` - the official submissions to the shared task
* `submissions/slttest22-doc-snippets.tsv` - document chunks
* `batches/*.json` - JSON batches for creating a campaign in Appraise
* `scores/*.csv` - scores exported from Appraise
* `ranking.log` - output of Appraise script for computing system rankings
