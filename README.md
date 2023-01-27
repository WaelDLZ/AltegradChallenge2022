# Protein Cellular Component Ontology Prediction 

Challenge for the 2022 ALTEGRAD course of Master MVA of ENS Paris-Saclay.

For more information, see [challenge-overview](https://www.kaggle.com/competitions/altegrad-2022/overview).

We describe our approach in our [report](https://github.com/WaelDLZ/AltegradChallenge2022/blob/main/report.pdf) and our [presentation](https://github.com/WaelDLZ/AltegradChallenge2022/blob/main/presentation.pdf).

## Abstract
Machine learning for protein engineering has attracted a lot of attention recently. Proteins are composed of sequences of amino acids and have a specific 3D structure. They can be seen as natural language sequences and graphs which motivates the use of graph models in combination with NLP models to solve bioinformatics tasks with proteins. In this paper, we use the sequence and structure of those proteins and classify them into $18$ different classes to predict their Cellular Component Ontology. We compare several graph models, mainly GCNs, GATs and a HGP-SL and rely on the ProBert embeddings of amino acids and proteins. We show promising results, achieving an optimal negative log-likelihood of $0.84$ on the public test set and $0.779$ on the private test set.

## Code
This repository contains our implementation of GCN, GAT, HGPSL and functions to embed proteins with ProtBert.

## Evaluation
The performance of the models is assessed with the logarithmic loss measure. This metric is defined as the negative log-likelihood of the true class labels given a probabilistic classifier’s predictions. Specifically, the multi-class log loss is defined as: $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^C y_{ij}\log p_{ij}$$ where $N$ is the number of samples (i.e. number of proteins), $C$ is the number of classes (i.e. the 18 categories), $y_{ij}$ is $1$ if the sample $i$ belongs to class $j$ and $0$ otherwise, and $p_{ij}$ is the predicted probability that the sample $i$ belongs to class $j$.

## Results
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/64415312/215042512-6fc76aa9-2a8f-41b8-9e3b-40afda46f20a.png">
</p>
 
