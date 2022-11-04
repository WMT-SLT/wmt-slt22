./scripts/splits.py srf.0 13 2 12 18 | tee submissions/slttest22-doc-snippets.tsv
./scripts/splits.py focusnews.1 11 4 0 0 | tee -a submissions/slttest22-doc-snippets.tsv
./scripts/splits.py focusnews.2 12 4 0 0 | tee -a submissions/slttest22-doc-snippets.tsv
./scripts/splits.py focusnews.3 11 3 10 1 | tee -a submissions/slttest22-doc-snippets.tsv
./scripts/splits.py focusnews.4 13 1 12 3 | tee -a submissions/slttest22-doc-snippets.tsv
./scripts/splits.py focusnews.5 13 2 12 3 | tee -a submissions/slttest22-doc-snippets.tsv
