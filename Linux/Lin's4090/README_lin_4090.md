# README

## 服务器简介

服务器地址
192.168.10.201

账号
root

密码
123123

## 目录介绍

项目文件

/opt/Projects

已下载的模型文件路径

/opt/models

已下载的数据集文件路径

/opt/models/datasets

## 加载本地模型及数据集方式

### 模型

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM

MODEL_NAME = "/opt/models/gpt2"  # 对应下载时存储的本地文件夹

# MODEL_NAME = "gpt2"  # 同样可以用这个方式，模型在.cache里有符号链接

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

# Model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)

```

### 数据集

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import datasets
from datasets import load_dataset

DATASET_NAME = "/opt/models/datasets/rotten_tomatoes"  # 对应下载时存储的本地文件夹

# DATASET_NAME = "rotten_tomatoes"  # 同样可以用这个方式，数据集在.cache里有符号链接

raw_datasets = load_dataset(DATASET_NAME)

```
