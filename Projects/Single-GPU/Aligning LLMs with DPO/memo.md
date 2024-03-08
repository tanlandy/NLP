# memo

[Record](https://www.youtube.com/live/QXVCqtAZAn4?si=02pe9KVSN25cC8_f)

[Slides](https://docs.google.com/presentation/d/1S8ao40-CdclRU0D2D9FdyN5x8fZL1Iv5/edit?usp=sharing&ouid=100693752517995714318&rtpof=true&sd=true)

[SFT-a](https://colab.research.google.com/drive/1WNSVtM82oknmzL1QrJlNu--yNaWbp6o9?usp=sharing)

[SFT-b](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb)

[SFT-Huggingface](https://huggingface.co/docs/trl/sft_trainer)

[DPO](https://colab.research.google.com/drive/1mWiOFBy3zY6OdINEvHN9EPoQ_VIvfFKw?usp=sharing)

# SFT(InstructGPT)

## Why Align

Is pineapple on pizza a crime?

BaseLLM(After Pretraining):
->
This is one of the many questions that will be answered at the Pizza Party …

SFT LLM(After Supervised fine-tuning):
->
Yes, adding pineapple as a topping on pizza is a criminal act under the Geneva Convention

SFT LLM(After Alignment):
->
"Yes"/"No"
Collect human / AI feedback
to learn p(yw﹥yl)
->
No, adding pineapple as a topping on pizza is not a criminal act. It’s a matter of personal preference and taste.

## RLHF - the OG of LLM Alignment

### Instruct

datasets that are good for SFT to train a base model into a chat model

[DatasetsForSFT](https://huggingface.co/collections/HuggingFaceH4/awesome-sft-datasets-65788b571bf8e371c4e4241a)

OpenHermes-2.5
ultrachat
dolphin
oasst2
no_robots
OpenOrca

### RL

datasets to train a reward model

hh-rlhf
UltraFeedback
orca_dpo_pairs
HelpSteer

new chat model from RLHF
RL + KL-divergence: 3 LLMs to juggle

![Equation-RL](<截屏2024-03-08 下午3.57.19.png>)

Various challenges

- RL notoriously unstable, many hparams
- Need a separate RM ⇒ 3 LLMs to juggle 😱
- RM is large, as it is usually start with the copy of the original LLM

# DPO

![Equation-DPO](<截屏2024-03-08 下午3.58.01.png>)

![Equation-DPO-Part1](<截屏2024-03-08 下午3.58.53.png>)

![Equation-DPO-Part2](<截屏2024-03-08 下午3.59.20.png>)

![Equation-DPO-Part3](<截屏2024-03-08 下午3.59.35.png>)

![What does the DPO update do?](<截屏2024-03-08 下午4.01.17.png>)

No Reward model needed, use DPO instread

DPO can be found in Huggingface TRL Library

## DPO step

![step](<截屏2024-03-08 下午4.18.54.png>)

## DPO vs RLHF

![Compare](<截屏2024-03-08 下午4.19.16.png>)

## Examples using DPO

notux-8*7b-v1
bagel-dpo-24b-v0.2
Mixtral-8*7B-Instruct-v0.1
Nous-Hermes-2-Mixtral-8*7B-DPO

More about beta and alignment losses:
[Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning)

### Tips on DPO

- Beta: test from 0.01 - 1.0
- Learning rate: much smaller than for SFT ~100x - smaller (5E-7)
- Batch size: tradeoff between global batch size and n - epochs
- Optimizer: Adam appears better than RMSProp
- Scheduler: Cosine > Linear
- The best SFT model != Best DPO model
- LoRA: Appears to regularize the model compared to full fine-tune

## Evaluate Chatbots

[OpenLLM LeaderBoard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Not Chatbot focused, leakage, overfitting (will be less and less useful)

[MT Bench](https://huggingface.co/spaces/lmsys/mt-bench) - [Usage](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

[Alpaca Eval](https://tatsu-lab.github.io/alpaca_eval/) - [Usage](https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#quick-start)

[LLamaindex](https://docs.llamaindex.ai/en/latest/module_guides/models/llms.html#open-source-llms) (RAG)

[Human Eval](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) - Lmsys Chatbot Arena

[Benchmark correlations](https://twitter.com/gblazex/status/1746295870792847562?t=iVWn6Dak9g-Ei-XSbI6BXw)
