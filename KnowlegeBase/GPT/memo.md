# memo

## norm比较

Batch Norm：把每个Batch中，每句话的相同位置的字向量看成一组做归一化。
Layer Norm：在每一个句子中进行归一化。

Batch Normalization（Batch Norm）：
缺点：在处理序列数据（如文本）时，Batch Norm可能不会表现得很好，因为序列数据通常长度不一，并且一次训练的Batch中的句子的长度可能会有很大的差异；此外，Batch Norm对于Batch大小也非常敏感。对于较小的Batch大小，Batch Norm可能会表现得不好，因为每个Batch的统计特性可能会有较大的波动。

Layer Normalization（Layer Norm）：
优点：Layer Norm是对每个样本进行归一化，因此它对Batch大小不敏感，这使得它在处理序列数据时表现得更好；另外，Layer Norm在处理不同长度的序列时也更为灵活。

1.在指令微调中，如何设置、选择和优化不同的超参数，以及其对模型效果的影响？

2.在指令微调中，如何选择最佳的指令策略，以及其对模型效果的影响？

3.llama, glm，bloom等现有大模型的数据处理，训练细节，以及不足之处模型架构的优化点，包括但不限于attention, norm, embedding

4.解决显存不够的方法有哪些？

5.请解释P-tuning 的工作原理，并说明它与传统的 fine-tuning方法的不同之处。

6.介绍一下Prefix-tuning的思想和应用场景，以及它如何解决一些NLP任务中的挑战

7.Lora的原理和存在的问题讲一下？

8.bf16，fp16半精度训练的优缺点

9.如何增加context length 模型训练中节约显存的技巧。

10.RLHF完整训练过程是什么？RL过程中涉及到几个模型？显存占用关系和SFT有什么区别？

11.RLHF过程中RM随着训练过程得分越来越高，效果就一定好吗？有没有极端情况？

12.encoder only，decoder only，encoder-decoder 划分的具体标注是什么？典型代表模型有哪些？
