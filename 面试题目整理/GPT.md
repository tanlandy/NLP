# GPT

## Useful links

[面试汇总](https://zhuanlan.zhihu.com/p/643560888)

## Questions

1. 为什么现在的LLM都是Decoder Only的架构？

首先简述几种主要的架构：
以GPT为代表的Decoder Only架构，
以BERT为代表的Encoder Only架构。
以T5和BART为代表的Encoder-Decoder架构，

然后说明要比较的对象：
淘汰掉BERT这种Encoder Only架构，因为Encoder Only架构的模型只能用Masked language modeling来做预训练，不擅长做生成任务，做NLU也一般需要有监督的下游数据来微调，所以不具有可比性
相比之下，Decoder Only架构的模型用next token prediction来做预训练，可以直接用于生成任务。在各种下游任务上zero shot和few shot泛化性能都很好。
而Encoder-Decoder架构的模型用seq2seq的方式来做预训练，可以直接用于生成任务，并且泛化性能也不错。

最后对比Decoder Only和Encoder-Decoder架构：

1. 架构简单，就一个Decoder，减少了模型的复杂性，有助于更有效的学习和处理大量数据
2. Decoder Only架构的模型从经验来看泛化性能更好，泛化性能更好的原因是
    - Decoder Only架构的模型的预训练任务是next token prediction，这个任务更接近生成任务，所以预训练的模型更适合做生成任务
    - 带有隐式的位置编码，打破了Transformer的局限性，可以处理任意长度的输入
3. 效率高：Decoder Only支持一直复用KV-Cache，对多轮对话更友好
4. 路径依赖
    - OPENAI做出来了，探索出了行之有效的训练方法和scaling law，有先发优势
5. 对于GLM等一些比较另类的模型结构，虽然也到了几十几百B的规模，但是还没有特别系统的实验比较说明Decoder Only架构一定最好
