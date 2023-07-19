# Notes

## Learning objects

1. Discuss model pre-training and the value of continued pre-training vs fine-tuning
2. Define the terms Generative AI, large language models, prompt, and describe the transformer architecture that powers LLMs
    - top K: for the softmax output, select an output from the top K most likely tokens
    - top p: for the softmax output, select an output from the tokens with cumulative probability p
    - temperature: for the softmax output, divide the logits by the temperature before applying softmax. Higher temperature results in more random outputs.
3. Describe the steps in a typical LLM-based, generative AI model lifecycle and discuss the constraining factors that drive decisions at each step of model lifecycle
    - scope: define the use case
    - select: choose an existing model or pretrain a new model
    - adapt and align model:
      - prompt / fine-tuning / aligh with human feedback
      - evaluate
    - application
      - optimize and deploy for inference
      - augment model and buil LLM-based application
4. Discuss computational challenges during model pre-training and determine how to efficiently reduce memory footprint
   - DDP: Distributed Data Parallel
      - Fully Sharded Data Parallel(FSDP)
        - Motivated by ZeRO paper: zero data overlap between GPUs
5. Define the term scaling law and describe the laws that have been discovered for LLMs related to training dataset size, compute budget, inference requirements, and other factors.
   - background
     - 1 PFLOPS-day = 1 quadrillion 一千兆 floating point operations per second per day = 8 V100 GPUs for 1 day = 2 A100 GPUs for 1 day
     - 3B T5模型，需要100 PFLOPS-day
     - 13B GPT-3模型，需要300 PFLOPS-day
   - chinchilla
     - very large model may be over-parameterized and under-trained
     - Compute optimal training datasize is ~20x number of parameters

LLM types:

1. Encoder-only
   - autoencoding models 自编码模型
   - objective
     - reconstruct text
   - 原理
     - 从两边往中间做完形填空
   - use cases
     - sentiment analysis
     - NER
     - word classification
   - examples
     - BERT
     - RoBERTa
2. Decoder-only
   - autoregressive models 自回归模型
   - objective
     - predict next token
   - 原理
     - 从左往右预测下一个token
   - use cases
     - text generation
   - examples
     - GPT
     - BLOOM
3. Encoder-decoder
   - seq-to-seq models
   - objective
     - reconstruct span
   - 原理
     - 从两边往中间做长空格完形填空
   - use cases
     - translation
     - text summarization
     - question answering
   - examples
     - T5

模型的存储
1B参数：
1 parameter = 4 bytes (32 bits float)
1B parameters = 4GB
Additional GPU RAM needed when training: ~20 times extra bytes per parameter
80G RAM needed @ 32-bit full precision

Quantization: 

- FP16: 16-bit half precision: 2 bytes per parameter
- BFLOAT16: 16-bit brain floating point: 2 bytes per parameter
- INT8: 8-bit integer: 1 byte per parameter

## Quiz

1. The input for working with LLMs is referred to as the prompt and the output from the LLM is referred to as the completion.
2. Invoking actions from text refers to the ability of large language models, to invoke Apis, or some actions from elsewhere, including connecting to resources that are based on the Internet.
3. Converting code comments into executable code: Translation
4. Self-attention is a mechanism that allows the model to focus on different parts of the input sequence during computation.
5. Stages of the LLM lifecycle:
   - Scope: define the problem and identify relevant datasets
   - Select
   - Adapt and align
   - Application
6. RNNs vs transformers:
   - RNNs: sequential, struggle with compute and memory
   - Transformers: parallelizable, attention mechanism helps to capture long-range dependencies



## Reading lists

Transformer Architecture
[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

- This paper introduced the Transformer architecture, with the core “self-attention” mechanism. This article was the foundation for LLMs.

[BLOOM: BigScience 176B Model (Notion)](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4)

- BLOOM is a open-source LLM with 176B parameters (similar to GPT-4) trained in an open and transparent way. In this paper, the authors present a detailed discussion of the dataset and process used to train the model.

[Vector Space Models](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3)

- Series of lessons from DeepLearning.AI's Natural Language Processing specialization discussing the basics of vector space models and their use in language modeling.

[Pre-training and scaling laws](https://arxiv.org/abs/2001.08361)
Scaling Laws for Neural Language Models

- empirical study by researchers at OpenAI exploring the scaling laws for large language models.

Model architectures and pre-training objectives
[What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/pdf/2204.05832.pdf)

- The paper examines modeling choices in large pre-trained language models and identifies the optimal approach for zero-shot generalization.

[HuggingFace Tasks](https://huggingface.co/tasks)
 and
[Model Hub](https://huggingface.co/models)

- Collection of resources to tackle varying machine learning tasks using the HuggingFace library.

[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf)

- Article from Meta AI proposing Efficient LLMs (their model with 13B parameters outperform GPT3 with 175B parameters on most benchmarks)

Scaling laws and compute-optimal models
[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)

- This paper investigates the potential of few-shot learning in Large Language Models.

[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)

- Study from DeepMind to evaluate the optimal model size and number of tokens for training LLMs. Also known as “Chinchilla Paper”.

[BloombergGPT: A Large Language Model for Finance](https://arxiv.org/pdf/2303.17564.pdf)

- LLM trained specifically for the finance domain, a good example that tried to follow chinchilla laws.
