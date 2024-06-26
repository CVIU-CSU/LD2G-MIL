## Multiple Instance Learning with Parameter-Efficient Foundation Model Adaptation for Neonatal Retinal Screening

This is the code implementation of Multiple Instance Learning with Parameter-Efficient Foundation Model Adaptation for Neonatal Retinal Screening.
Our code is built on the basis of MMClassification.

MMClassification is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

## Methodological Overview

<img src="./img/framework1.jpg" alt="framework" style="zoom:75%;" />

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
