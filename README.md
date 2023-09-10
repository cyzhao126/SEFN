# SEFN：Segment-aware Evidential Fusion Network for Trustworthy Video Sewer Defect Classification （ICRA 2024）

## Introduction

This is the official repository for *SEFN：Segment-aware Evidential Fusion Network for Trustworthy Video Sewer Defect Classification*. In this repository, we release the splits of training set and test set for $\tau_{cls}$ and $\tau_{ood}$, as well as the source code.

## Installation

1、Create conda environment:
```
$ conda create -n SEFN python=3.7
$ conda activate SEFN
```
2、This code was implemented with python 3.7, torch 1.10.0+cu113
```
$ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
3、 Install other dependency
```
$ pip install -r requirements.txt
```
