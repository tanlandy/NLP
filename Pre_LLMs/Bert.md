# Bert

BERT stands for Bidirectional Encoder Representations from Transformers.

## 数据

共16G，其中3.3B单词，2.5B来自维基百科，0.8B来自BooksCorpus

## 训练

训练过程：

1. Pretraining：基于大量的无标签数据，无监督学习表征
2. Fine-tune：使用2.5k-392k有标签数据进行微调

Tokenizer采用Byte Pair Encoding

### 超参数

Base: 110M 层数为12，隐层维度为768。
Large: 340M 层数为24，隐层维度为1024。

![BERT模型架构](<截屏2024-03-18 下午3.45.28.png>)

## 效果

效果源自：

1. MLM: Masked Language Model 通过完形填空，使模型学习到上文和下文。15%被masked了。
   直接使用 MLM 有两个问题：其一是，训练和预测并不一致，后者在下游任务的 fine-tune 中并没有 [MASK] 标记。为此，论文作如下优化，对输入数据中，随机选择 15% token 做预测，同时做如下操作：
   - 80% 标记为 [MASK]
   - 10% 随机替换成其他token
   - 10% 保持不变
2. NSP: Next Sentence Prediction：学习两个句子的关系，一个句子是否会是这一句的下个句子。数据集是50%句子对和50%随机对

## 使用

可以在很多子任务下进行微调。比如文本分类，文本相似度，问答，文本标签标注，命名实体识别等

微调阶段我们需要关注的参数有 batch_size、learning rate、epoch，这些参数和训练阶段是可以不同的，一般来说需要根据多个不同的参数组合搜索最优的组合。实验表明，基于特征和基于微调的方式都能取得很好的效果。

### 分类任务

对于分类任务来说，包括单个句子的分类（如新闻分类）和句子对分类（如文本匹配），可以直接使用[CLS]位置的输出作为类别特征，然后接上softmax输出类别概率，

直接使用[CLS]输出作为分类特征，是因为该输入可以看做上游输出的加权平均，因此是有分类依据的。此外还可以使用GlobalMaxPooling或GlobalAveragePooling，需要注意mask处理。当然还可以纵向来看，结合不同的层做以上操作，如最后两层Pooling后拼接。

### 序列标注任务

如中文分词、NER，其实使用就像LSTM一样，直接接CRF即可。

### 区间预测任务

如MRC这类任务，段落A和问题Q以 [SEP] 为间隔并拼接在一起：Q [SEP] A，然后BERT要找A中的答案的开始位置和终止位置。

# RoBERTa

可以说是充分探索、训练的Bert。RoBERTa效果相比BERT更好(提升2%-20%)

## 数据

10倍于Bert共160G数据

## 训练

相比Bert，显著增加了训练时间（*4-5）
未使用NSP，引入了动态掩码，batch size更大
Tokenizer采用Byte Pair Encoding

### 超参数

Base: 110M
Large: 340M

# DistilBERT

增加了推理速度

## 数据

共16G，其中3.3B单词

## 训练

### 超参数

Base: 66M
