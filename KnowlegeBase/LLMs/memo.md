# memo

## P-tuning

该方法将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。

# 大模型技术方案对比

[Source](https://zhuanlan.zhihu.com/p/680260131)

## LLaMa2

### Tokenizer

使用bytepair encoding (BPE) 的算法，进行分词。针对连续数字分会拆为单独的数字并且未在词典中的词使用bytes替换，最终词典大小为32k。

### Architecture

位置编码：rotary positional embeddings

激活函数：SwiGLU

normal方法：RMSNorm(Root Mean Square Normalization)
只计算输入的方差（即每个元素的平方的平均值），然后通过元素级操作计算归一化后的输出。
具体步骤如下：

- 计算输入的平方的平均值（variance）。
- 使用逆平方根（torch.rsqrt()）来计算方差的倒数（相当于标准差的倒数）。
- 将输入与计算出的标准差倒数相乘，得到归一化的结果。

torch自带的LayerNorm是最完整的实现，包括了可学习的参数；而RMSNorm和自定义LayerNorm则省略了这些参数，可能会牺牲一些模型的表达能力，但在某些情况下可能更简单或更高效。RMSNorm特别适用于那些不需要额外参数的大规模模型。

content长度

grouped-query attention(GQA)，将llama1 长度2048 提升至 4096，

![LLaMa2_Attn](<截屏2024-03-19 下午3.28.10.png>)

传统的transformer为多头注意力(MHA)模型，分组查询注意力变体为GQA模型，另外还有MQA模型，该结构为特殊的GQA模型，group 为1。

LLaMa2的attention去掉了bias

FeedForward使用了3层全连接

MQA模型：

在Multi-Query Attention中，每个位置的输入可能会生成多个查询向量，这些查询向量可以独立地参与注意力计算，并与键（key）和值（value）矩阵进行交互。
多个查询向量可以捕捉到输入的不同方面或信息源，增强模型的理解和表达能力。例如，一个查询可能关注词汇级别的信息，另一个查询可能关注句法或语义级别的信息。

## Mistral 7B

### Architecture

在LLaMa2架构的基础上，提出了Sliding Window Attention和Rolling Buffer Cache

Sliding Window Attention，attention 中的操作数量与序列长度呈二次关系，通过Sliding Window Attention，可减少计算，但是会牺牲一点的效果。

![SWA](<截屏2024-03-19 下午3.41.59.png>)

Rolling Buffer Cache，传统注意力机制中的参数运算次数与序列长度呈二次关系，内存消耗与序列成线性关系。在推理时，会出现更高的延时和更小的数据吞吐量。

![RBC](<截屏2024-03-19 下午3.41.39.png>)

## Qwen

### Data

去重：标准化后进行完全匹配重复数据删除,以及使用 MinHash 和 LSH 算法进行模糊重复数据删除

过滤质量低：过滤低质量的数据,采用了规则型和基于机器学习的方法的组合。多个模型对内容进行评分,包括语言模型,文本质量评分模型以及用于识别潜在的攻击性或不适当内容的模型。人工从各种来源中对文本进行抽样并审阅,以确保其质量。

高质量指令：由于多任务指令可以增强他们的零样本和少样本性能，预训练过程中加入了高质量的指令数据。

### Tokenizer

使用基于bytepair encoding (BPE) 的tiktoken算法，其相当于BPE tokenizer分词更快。首先使用cl100k作为base token，针对连续数字分会拆为单独的数字，最终词典大小为152K。

### Architecture

FFN：和LLaMa2一样，使用了3个全连接层

Attention：仅qkv有bias，其余没有bias

位置嵌入：RoPE，选择使用FP32精度的逆频率矩阵,而不是BF16或FP16,以优先考虑模型性能并获得更高的准确性。

激活函数：SwiGLU

normal方法：RMSNorm，前馈网络(FFN)的维度从隐藏大小的4倍减少到隐藏大小的83倍

content长度：

长度外推，QKV注意力层中添加bias以增强模型的外推能力。下图可看到加上了bias，长度大于1024效果下降不是很多。

LogN-Scaling，q和v乘以一个系数，context length和training length的长度关系，来保持注意力的稳定。

window attention，将注意力限制在有限的上下文窗口内,防止模型关注距离太远的标记。基于这一发现,我们为每个层分配不同的窗口大小,对较低层使用较短的窗口,对较高层使用较长的窗口。

## BaiChuan2

### Data

来源收集数据,包括常规互联网网页、书籍、研究论文、代码库等,以构建一个广泛的世界知识体系。

去重，构建了一个大规模的重复数据删除和聚类系统,支持LSH类似特征和稠密嵌入特征。最终只保留原始数据的31.68%的数据进行训练。

### Tokenizer

字节对编码(BPE)，不对输入文本应用任何规范化,也不添加虚拟前缀。将数字拆分为单独的数字，处理额外空格的代码数据,向分词器添加仅空格标记，最大标记长度设置为32,以处理长中文词组。

### Architecture

位置嵌入：RoPE

激活函数：SwiGLU

注意力层：xFormers减少内存。

normal方法：RMSNorm，并且规范化输出嵌入lm_head。在我们规范化头部之后(蓝色),在刚开始训练的2000step左右训练变得非常稳定,这导致了更好的性能。

最大z损失，在训练过程中,发现LLM的logits可能变得非常大。添加了一个最大z损失来规范化logits。其中z是最大logit值,这有助于稳定训练,并使推理更加稳健地适应超参数。

## 对比

数据上，qwen和baichuan2去重上做了许多工作。并且qwen在数据质量上做了两方面工作首先过滤低质量语料，其次加入高质量指令提高预训练效果。

模型结构上：都在更长预测上下文长度进行提升，只是每个模型使用方式不一样。llama2使用GQA，Mistral 使用 Sliding Window Attention 和 Rolling Buffer Cache。qwen在QKV注意力层中添加bias以增强模型的外推能力、NTK-aware interpolation、LogN-Scaling和window attention。

另外，baichuan2使用规范化输出嵌入lm_head和最大z损失提升模型稳定性。qwen在核心的矩阵计算中使用FP32换取更好效果。

## ChatGLM3

### Architecture

qkv合并成一个全中
