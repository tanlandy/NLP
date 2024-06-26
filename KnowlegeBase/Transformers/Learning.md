# Transformer

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

[更详细的GPT模型拆解](https://jalammar.github.io/illustrated-gpt2/)

## Overview level

source: [The Narrated Transformer Language Model](https://youtu.be/-QH8fRhqFHM?si=Hg6YTvZwzz_GayVj)

GPT-2: 36 Decoder Blocks

![Training Example](<截屏2024-02-26 下午1.37.19.png>)

![Training Process](<截屏2024-02-26 下午1.38.41.png>)

If only use the FFN

![FFN Only](<截屏2024-02-26 下午1.42.18.png>)

Added Attention mechanism

![attention](<截屏2024-02-26 下午1.42.42.png>)

Can tell the difference between it

Entire structure

![Transformer Block](<截屏2024-02-26 下午1.43.48.png>)

![Token Embeddings](<截屏2024-02-26 下午1.49.01.png>)

Token Embeddings就是一个词表，这个词表中有50257个词汇

![Transformers Hidden State](<截屏2024-02-26 下午1.54.36.png>)

图片中展示了2个Transformer Block。一般来说Block越多，表示能力越大

![Hidden state to Text](<截屏2024-02-26 下午1.55.50.png>)

## GPT3

source: [How GPT3 Works - Easily Explained with Animations](https://youtu.be/MQnJZuBGmSQ?si=7uuo666v9yHQYj6w)

## KV-cache

[Source](https://r4j4n.github.io/blogs/posts/kv/)

将先前计算得到的键（K）和值（V）向量存储起来，模型只需要为新加入的序列部分计算QK值，并与缓存中的键值进行组合，从而大幅度减少计算量。

### 效果

更高效处理长序列数据，大幅减少计算量，显著减少延迟。
