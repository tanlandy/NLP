# Notes

## Learning objects

1. Describe how fine-tuning with instructions using prompt datasets can improve performance on one or more tasks
   - In-context learning (prompt)
     - may not work for smaller models
     - examples take up space in the context window
   - Pre-trained LLM -> train with Task-specific examples (Prompt-completion pairs)-> Fine-tuned LLM
     - the pairs includes a specific instruction to the LLM
     - known as instruction dataset
   - Fine-tuning on a single task
     - only 500-1000 examples are needed
     - can lead to reduction in ability on other tasks(catastrophic forgetting)
2. Define catastrophic forgetting and explain techniques that can be used to overcome it
   - catastrophic forgetting
     - fine-tuning on a single task can lead to reduction in ability on other tasks
   - ways to overcome
     - instruciton fine-tune on multiple tasks
       - instruction fine-tuning with FLAN
         - FLAN models refer to a specific set of instructions used to fine-tune a model
         - Fine-tuned LAnguage Net
         - FLAN-T5: Fine-tuned version of pre-trained T5 model
     - PEFT
3. Define the term Parameter-efficient Fine Tuning (PEFT)
   - types
     - Selective
       - select subset of initial LLM params to fine-tune
     - Reparameterization
       - Reparameterize model weights using a low-rank representation (LoRA)
     - Additive
       - Add trainable parameters to the pre-trained model
         - Adapters
         - Soft Prompts (prompt tuning)
4. Explain how PEFT decreases computational cost and overcomes catastrophic forgetting
   - LoRA
   1. Freeze most of the pre-trained model
   2. Inject 2 rank decomposition matrices
   3. Train the weights of the smaller matrices
   4. Update model for inference: multiply the 2 matrices, add the result to the pre-trained model
5. Explain how fine-tuning with instructions using prompt datasets can increase LLM performance on one or more tasks
   - Prompt tuning adds trainable parameters to the inputs

LLM Evaluation

- ROUGE
  - used for text summarization
  - ROUGE-1: unigram
    - Recall: matches by unigrams in reference
    - Precision: matches by model unigrams in output
  - ROUGE-2
  - ROUGE-L: LCS

- BLEU
  - used for text translation
  - average precision across range of n-gram sizes
  - range from 0 to 1, the higher the better

- Benchmarks
  - GLUE
    - General Language Understanding Evaluation
    - 9 tasks
  - MMLUE
    - Multilingual General Language Understanding Evaluation
    - 5 tasks
  - HELM

Parameters:

- Base Transformer
  - d \* k = 512 \* 64 = 32,768
- LoRA with rank r=8
  - A: r \* k = 8 \* 64 = 512
  - B: d \* r = 512 \* 8 = 4,096
  - 86% reduction in parameters

## Quiz

1. Catastrophic forgetting: fine-tuning on a single task can lead to reduction in ability on other tasks
2. Evaluation metric for text translation: BLEU
3. How LoRA works: decomposes weights into two smaller rank matrices and trains those instead of the full model weights
4. What is a soft prompt: a set of trainable tokens that are added to a prompt and whose values are updated during additional training to improve performance on  specific tasks

## Reading lists

### Multi-task, instruction fine-tuning

[Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)

- Scaling fine-tuning with a focus on task, model size and chain-of-thought data.

[Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html)

- This blog (and article) explores instruction fine-tuning, which aims to make language models better at performing NLP tasks with zero-shot inference.

### Model Evaluation Metrics

[HELM - Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/)

- HELM is a living benchmark to evaluate Language Models more transparently.

[General Language Understanding Evaluation (GLUE) benchmark](https://openreview.net/pdf?id=rJ4km2R5t7)

- This paper introduces GLUE, a benchmark for evaluating models on diverse natural language understanding (NLU) tasks and emphasizing the importance of improved general NLU systems.

[SuperGLUE](https://super.gluebenchmark.com/)

- This paper introduces SuperGLUE, a benchmark designed to evaluate the performance of various NLP models on a range of challenging language understanding tasks.

[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)

- This paper introduces and evaluates four different measures (ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S) in the ROUGE summarization evaluation package, which assess the quality of summaries by comparing them to ideal human-generated summaries.

[Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/pdf/2009.03300.pdf)

- This paper presents a new test to measure multitask accuracy in text models, highlighting the need for substantial improvements in achieving expert-level accuracy and addressing lopsided performance and low accuracy on socially important subjects.

[BigBench-Hard - Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models](https://arxiv.org/pdf/2206.04615.pdf)

- The paper introduces BIG-bench, a benchmark for evaluating language models on challenging tasks, providing insights on scale, calibration, and social bias.

### Parameter- efficient fine tuning (PEFT)

[Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)

- This paper provides a systematic overview of Parameter-Efficient Fine-tuning (PEFT) Methods in all three categories discussed in the lecture videos.

[On the Effectiveness of Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2211.15583.pdf)

- The paper analyzes sparse fine-tuning methods for pre-trained models in NLP.

### LoRA

[LoRA Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)

- This paper proposes a parameter-efficient fine-tuning method that makes use of low-rank decomposition matrices to reduce the number of trainable parameters needed for fine-tuning language models.

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)

- This paper introduces an efficient method for fine-tuning large language models on a single GPU, based on quantization, achieving impressive results on benchmark tests.

### Prompt tuning with soft prompts

[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)

- The paper explores "prompt tuning," a method for conditioning language models with learned soft prompts, achieving competitive performance compared to full fine-tuning and enabling model reuse for many tasks.
