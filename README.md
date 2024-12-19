# AccTD

This repository contains the source code of the paper "GPUSCAN++: Efficient Structural Graph
Clustering on GPUs" by 
Long Yuan, Zeyu Zhou, Xuemin Lin, Zi Chen, Xiang Zhao, Fan Zhang. 

## Overview

We provide source codes of `GPUSCAN++`, `GPUSCAN++uvm`, `GPUSCAN++o` for structural graph
clustering. 
The codes are organized in three cmake projects. 
We give the code organization in the following "Algorithms" section. 

## Algorithms

Algorithm | Code Folder |
--- | --- 
Our Optimized Structural Graph Clustering on GPUs | GPUSCAN++
Structural Graph Clustering on GPUs with UVM | GPUSCAN++uvm
Our Optimized Structural Graph Clustering on GPUs | GPUSCAN++
Our new Out-of-core algorithm| GPUSCAN++uvm


## Datasets (Input) Pre-Processing 

We use the converter in [ppSCAN-release](https://github.com/RapidsAtHKUST/ppSCAN/tree/master/ppSCAN-release) 
to transform an `edge list txt file` into our format (two binary files `b_degree.bin` and `b_adj.bin` under a folder). 
These two binary files contain the information for the reconstruction of the Compressed Sparse Row (CSR) format.
Please see [Lijun's datasets format](https://github.com/LijunChang/Cohesive_subgraph_book/tree/master/datasets) for more details.

## Build

We build the three cmake projects separately. 

* build steps:

```
cd GPUSCAN++
make
```

## Run

Suppose we already have two binary files `b_degree.bin` and `b_adj.bin` under a folder `folder_path` , parameters `epsilon` and `miu` as our input. 
Then we run as follows

```zsh
./GPUScan folder_path epsilon miu
```