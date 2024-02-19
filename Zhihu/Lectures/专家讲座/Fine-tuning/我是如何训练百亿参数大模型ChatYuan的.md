# 我是如何训练百亿参数大模型ChatYuan的

演讲嘉宾：张轩玮

Goal:
预训练
微调
强化学习

# transformer回顾

14:00左右，讲的很不错

encoder-only: BERT
decoder-only: GPT

# 模型架构相关

## 对比

![comparison](<Screenshot 2024-02-18 at 5.59.17 PM.png>)

输入信息和输出信息对等：机器翻译任务 -> 编码器和解码器

encoder-decoder构建目标：掩盖的频率和长度都有影响

## 生成式attention层面

![attention](<Screenshot 2024-02-18 at 6.07.19 PM.png>)

Fully-visible: 每个位置都可见
Causal: 只能看到前面的
Causal with prefix：前面的可以看到全部，后面的只能看到前面的

## 代表性的模型

![model](<Screenshot 2024-02-18 at 6.09.01 PM.png>)

图中深色的代表全部都可见，浅色代表只能看到前面的

最右是微软提出的Unilm，既能节省参数，又能达到输入是理解的

# 中文字典构建

![chineseDict](<Screenshot 2024-02-18 at 6.31.26 PM.png>)

对于英文来说，如果用character level，那么颗粒度不够，太粗了。如果用word level，那么颗粒度又太细了，每个embedding728维，字典太太庞大。所以使用了subword level。

构建subword有很多方法，其中一个是BPE

# 预训练

## 预训练数据的构建

C4: 750GB 谷歌提供

CLUE corpus: 300GB 基本上是中文

Yuan: 1000GB 浪潮开源的部分

Wudao: 300GB

The Stack: 6000GB 开源社区提供的代码数据

![data_prep](<Screenshot 2024-02-18 at 6.14.36 PM.png>)

完整的文本：最后不是标点符号结尾
删除黄赌毒、偏见：通过关键词剔除

相似去重：

- 布隆过滤器
- 局部哈希法

模型分析及过滤（更加精细化）

- 对文本的质量进行判别：计算PPL
- 数据类型分析模型：看数据的分布（通用预料/代码/论文）

## 预训练目标的构建

![pairs](<Screenshot 2024-02-18 at 6.35.49 PM.png>)

ChatYuan是以T5为基座

![训练参数和数据量的关系](<Screenshot 2024-02-18 at 6.48.14 PM.png>)

OPENAI：Token是训练参数的20倍，是最佳的配比。Scale Law

![并行化训练](<Screenshot 2024-02-18 at 6.51.55 PM.png>)

tensor并行：模型并行

流水线并行：不按照tensor分了，而是按照层来切分

![技巧](<Screenshot 2024-02-18 at 6.59.51 PM.png>)

学习率：直接用AdamW就可以。前期加快的目的是为了更快到最优点，衰减是为了避免跳出最优点

batchsize：前期较小，为了更快伸展出模型参数，让他不用一下子就拟合非常大的数据。后期模型能力越来越强，就给他更大的batchsize

定期保存checkpoint：一天一个。保存太多会占用空间、耽误训练时间

位置嵌入：越来越多使用相对位置编码（Alibi/RoPE）

# 任务微调数据的构建

![任务微调](<Screenshot 2024-02-18 at 7.05.52 PM.png>)

![任务类型](<Screenshot 2024-02-18 at 7.07.27 PM.png>)

![任务微调数据例子](<Screenshot 2024-02-18 at 7.07.49 PM.png>)

![训练技术的细节](<Screenshot 2024-02-18 at 7.10.22 PM.png>)

# 指令微调数据的构建

![指令数据构建](<Screenshot 2024-02-18 at 7.41.54 PM.png>)

![单轮数据的构建](<Screenshot 2024-02-18 at 7.44.56 PM.png>)

![多轮数据的构建](<Screenshot 2024-02-18 at 7.45.20 PM.png>)

单个多轮数据会构建多个训练数据

![SFT训练](<Screenshot 2024-02-18 at 7.46.44 PM.png>)

# 微调


# 强化学习

第一部份：训练RW 

![RW](<Screenshot 2024-02-18 at 7.47.53 PM.png>)

一般是用与训练模型相同的底座，训练一个分类器

![RW流程](<Screenshot 2024-02-18 at 7.50.25 PM.png>)

# 实践

![代码](<Screenshot 2024-02-18 at 7.52.29 PM.png>)

# 评测基准

![评论基准](<Screenshot 2024-02-18 at 7.52.48 PM.png>)


代码：
微调实践：https://colab.research.google.com/drive/1c3tRQZeOvdssOiAbWzKFTv31Rnw42-Bf?usp=sharing

[微调代码](./ChatYuan微调代码.ipynb)


构建字典代码：https://colab.research.google.com/drive/1shRwcfMQH4vzri_ou1Sx14OqMsgxHXFJ?usp=sharing

[构建字典代码](./sentencepiece_tokenizer构建字典实践.ipynb)



预训练代码：https://github.com/google-research/t5x

强化学习代码：https://github.com/CarperAI/trlx/tree/main/examples
