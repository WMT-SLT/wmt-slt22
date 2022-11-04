# Findings of the First WMT Shared Task on Sign Language Translation

This repository contains data and scripts for reproducing the evaluation
of the WMT22 Sign Language Translation Task.

## Human evaluation

The files related to the human evaluation reside in the directory `human_evaluation`.

Scripts:
* `generate-batches.sh` - produces batches for Appraise
* `generate-snippets.sh` - splits documents into 10-segment long chunks
* `generate-ranking.sh` - computes ranking from scores exported from Appraise
* `scripts/iaa.py` - for generating intra-annotator agreements
* `scripts/create_histogram.py` - for generating the histogram that appears in the paper

Data:
* `slttest2022.dsgs-de.all.xml` - the official test set
* `submissions/*.xml` - the official submissions to the shared task
* `submissions/slttest22-doc-snippets.tsv` - document chunks
* `batches/*.json` - JSON batches for creating a campaign in Appraise
* `scores/*.csv` - scores exported from Appraise
* `ranking.log` - output of Appraise script for computing system rankings

## Automatic evaluation

The files related to the human evaluation reside in the directory `automatic_evaluation`.

Scripts (directory `tools`):
* `automaticEval.py` - Automatic evaluation with BLEU, chrF++ and BLEURT for WMT-SLT 2022 Confidence intervals obtained via bootstrap resampling
* `corrMetricsHuman.py` - Pearson and Spearman correlations for the automatic metrics
* `plotMetrics.py` - 3D plot of the correlation between the automatic metrics

## Requirements
For running most scripts one needs to create a Python virtual enviroment and install

```commandline
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(The BLEURT requirements include Tensorflow which a heavy thing to download. If you don't need that, feel free to comment out the BLEURT requirement entry to save time and hard disk space).


As an exception, the script `generate-batches.sh` requires one to install Appraise from this repository: 
```commandline
git clone https://github.com/AppraiseDev/Appraise.git
cd Appraise
git checkout 147865c284d340085d1333e1b7ed2a40d52bd703
```


