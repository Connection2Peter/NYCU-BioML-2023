# 2023 NYCU Biological Machine Learning Final Project

## Download project
- ```git clone https://github.com/ConnectionOuOb/NYCU-2023-BioML.git```

## Download modules
- ```conda install matplotlib xgboost catboost scikit-learn pandas biopython```

## Generate NR sets by CD-Hit
- ```find testKmer/ -name "*.fasta" | xargs -I % bash -c 'cd-hit -i % -o testNR/$(basename % .fasta).nr050.fasta -c 0.5 -n 2 -T 0'```

## Contributors
- Connection
  - Generate Basic & SSE-PSSM feature sets
  - All ML/DL related coding
  - All Experiment
- Ivern
  - Generate iFeature feature sets
  - Some Model test case
  - Independent test
- Brian
  - Generate customize feature sets
  - Dataset pre-processing
  - Independent test

## Reference
1. Chen TR, Juan SH, Huang YW, Lin YC, Lo WC. A secondary structure-based position-specific scoring matrix applied to the improvement in protein secondary structure prediction. PLoS One. 2021 Jul 28;16(7):e0255076. doi: 10.1371/journal.pone.0255076. PMID: 34320027; PMCID: PMC8318245.
2. Zhen Chen, Pei Zhao, Fuyi Li, André Leier, Tatiana T Marquez-Lago, Yanan Wang, Geoffrey I Webb, A Ian Smith, Roger J Daly, Kuo-Chen Chou, Jiangning Song, iFeature: a Python package and web server for features extraction and selection from protein and peptide sequences, Bioinformatics, Volume 34, Issue 14, July 2018, Pages 2499–2502, https://doi.org/10.1093/bioinformatics/bty140
3. Fu L, Niu B, Zhu Z, Wu S, Li W. CD-HIT: accelerated for clustering the next-generation sequencing data. Bioinformatics. 2012 Dec 1;28(23):3150-2. doi: 10.1093/bioinformatics/bts565. Epub 2012 Oct 11. PMID: 23060610; PMCID: PMC3516142.
