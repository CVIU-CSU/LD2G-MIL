## Multiple Instance Learning with Parameter-Efficient Foundation Model Adaptation for Neonatal Retinal Screening

This is the code implementation of Multiple Instance Learning with Parameter-Efficient Foundation Model Adaptation for Neonatal Retinal Screening.
Our code is built on the basis of MMClassification.

MMClassification is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

## Methodological Overview

<img src="./img/framework1.jpg" alt="framework" style="zoom:75%;" />

## Datasets

In this study, 115,621 fundus images of 8886 neonates were acquired using RetCam3 from the Hunan Provincial Maternal and Child Health Hospital from 2015 to 2019. The collection and analysis of image data were approved by the Institutional Review Board of the Hunan Provincial Maternal and Child Health Hospital and adhered to the tenets of the Declaration of Helsinki. The resolution of images is $1600 \times 1200$ pixels. Multiple retinal images were taken from each subject at different angles. The dataset encompasses class labels determined by four professional ophthalmologists based on a series of retinal images from the subjects, categorized into Normal, Retinal Hemorrhage (RH), and Retinopathy of Prematurity (ROP). To ensure the greatest degree of accuracy and consistency, samples that were uncertain or ambiguous underwent a joint diagnostic review by all four ophthalmologists, leading to definitive annotations. We randomly divided the 8,886 subjects into the training set (train), validation set (val), and test set (test) with a ratio of 6:1:1. The distribution of annotated instances among classes is detailed in Table~\ref{tab1}. We conducted comparative evaluations of our method against others on the NFI test and reported our ablation experiments on the NFI val.

## Key Configuration Files

* LD2G-MIL with Gated-ABMIL baseline:
  * `config/_nfi_/LD2G-MIL.py`
* LD2G-MIL with DSMIL baseline:
  * `config/_nfi_/LD2G-MIL_DSMIL.py`

## Environment Configuration

```
conda create -n mmcls-nfi python=3.7 -y
conda activate mmcls-nfi

# cuda10
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -y
pip install mmcv-full==1.3.18  -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html -i https://pypi.douban.com/simple/

# cuda 11
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip install mmcv-full==1.3.18  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html -i https://pypi.douban.com/simple/

cd mmcls-nfi
chmod u+x tools/*
chmod u+x tools/*/*
pip install -r requirements.txt -i https://pypi.douban.com/simple/
pip install -v -e .  -i https://pypi.douban.com/simple/
```
