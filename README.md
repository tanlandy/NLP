# Basic

[fast.ai.course](https://course.fast.ai/)
-> 写完每个章节后面的问题

# PyTorch

[新手如何入门pytorch](https://www.zhihu.com/question/55720139/answer/147148105)

# NLP

All about NLP
学习和实践过程中的，与NLP相关的大本营

# 李沐精讲学习方式

看三遍

- 1X第一遍，记录下不会的东西
- 1.25X第二遍，做笔记，尽量全部看懂
- 1.5X第三遍，完善笔记，全部弄懂

[深度学习论文精读GitHub](https://github.com/mli/paper-reading)

## Suggetions

[导师让我搞gpt方向，我该怎么去学？ - Uranus的回答 - 知乎](https://www.zhihu.com/question/604134581/answer/3063230236)

## NLP综述

[知识笔记](https://zhuanlan.zhihu.com/p/597586623)

## Colab

[Files](https://colab.research.google.com/notebooks/io.ipynb)

## OpenAI接口整活

[Chameleon](https://chameleon-llm.github.io/)

[Grammatical Error](https://arxiv.org/abs/2303.13648)

[ChatGPT NLG Evaluator](https://arxiv.org/abs/2303.04048)

# 面经

## LLM

1. ChatGPT Plugins体验。不足与提升
   1. chatwithPDF plugin 卡在图像识别，而且主要是应用模型进行归纳
   2. edx中文识别问题，找爬虫就找不到，找web crawl就有了，主要用来引流
   3. wolfram
   4. 不足与提升：
      1. 工具型的plugin，如何引流
      2. 如何做到用户粘性
      3. 提升chatgpt的体验：更好的适配，能够自动化调用可以使用的plugin，而不是自己知道有哪些plugin然后手动加载
      4. 提升plugin的体验：如何做到更好的适配，更好的引导用户使用。知道当想要做什么的时候，有哪些plugin可以使用，以及应该怎么用
2. LLM做控制的局限，解决办法
3. 多模态有哪些思路
4. 视觉如何融入
5. LLM的输出输入如何审核？风险控制？
6. Langchain及原理
7. autogpt原理
8. 如何突破transformer token限制
9. 大语言模型有哪几种架构，哪几种最流行，为什么
10. 语言模型自监督有哪几种方法
11. prompt设计心得
12. 推理阶段有哪些提升chatgpt性能的方法
13. ChatGPT, GPT4的应用场景
14. 新模型效果与优势

## 神经网络基础

1. 计算基本神经网络的参数量rnn, cnn, transformer
2. 微调有哪几种方法？解释原理
3. adam原理。实现需要哪些依赖，是否有优化空间
4. 系统调用是什么？训练神经网络数据传输链路，哪里可以优化？
5. attention原理
6. LoRA原理
7. attention和seq2seq区别
8. 梯度下降，随机梯度下降

## CS基础

1. 多进程和多线程的区别
2. hash表和最小堆原理以及实现
3. 排序算法有哪些，解释原理。基于比较的排序算法，理论最快时间复杂度是多少
4. 已知随机分布的均值和方差，求使得熵最大的概率分布
5. 进程间通信方法？线程间通讯方法
6. Java虚拟机简介

## 参考公司

1. Minimax
2. 王小川-五季智能
3. 周伯文-衔远科技
4. 慧文-光年之外
5. 昆仑万维
6. hoxi
7. 爱诗科技
8. 智元机器人
9. stability.ai
10. softbank
11. huggingface

# Blogs

[紫气东来专栏](https://www.zhihu.com/people/zi-qi-dong-lai-1/posts)
[一文读懂](https://www.mikecaptain.com/2023/01/22/captain-aigc-1-transformer/#21%E6%84%9F%E7%9F%A5%E5%99%A8perceptron%E8%A7%A3%E5%86%B3%E4%BA%8C%E5%85%83%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E7%9A%84%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
[整理的面试题](https://zhuanlan.zhihu.com/p/643560888)
[八股](https://zhuanlan.zhihu.com/p/643829565)


# 一些面经
    - ，问了一些简历的经历，然后问了transformer相关的比较常见的问题比如MHA，复杂度，BERT/GPT这种。
    - 技术上问了大模型训练，transformer相关的内容和过往经历，第二面好像就聊offer了如果没有记错
    - 前几轮都是考leetcode为主，问了很少很浅的大模型的内容，记得是其中一面是让手写MHA。最后一轮和主管聊完之后被分配到了框架组的offer，
    - 比如模型参数大小，训练一些细节之类的
    - 4论面试，2轮带leetcode，然后浅浅问了相关经验，第三轮主要问了大模型的内容，比较标准的问题（transformer，训练，分布式，如何处理训练一些问题比如loss spike啥的）。第四轮是cross面，推荐组的一个大佬，感觉是象征性面试，因为之前做过搜广推问了相关问题，问了probability和脑筋急转弯
    - 他家是做框架为主的，一面问了很多框架内容，各种模型切分方式，问的比较深，好像也问了flash-attention的内容，感觉答得还凑合。后来我表达了想做算法端，他们算法那边确实也在招人不过hc不多，二面是算法组的人面的，问的应用端比较多，感觉可能不做基座？然后面挂了。他家在圈子里感觉框架端名声挺大的
    - ，一面记得是leetcode+写MHA，感
    - 表达了想去模型组，然后面试官应该是和HR交流了，第二组开始就是模型组的人来面试。有没有leetcode不记得了，面试内容也比较中规中矩，transformer，大模型训练这种内容。
    - 这家感觉面试问的广度和深度是国内厂面的里边最相关也最专业的。一面是个大佬，之前Kaiming大神一些高引论文的共同作者。一开始问了包括一些基础的大模型训练的内容，transformer架构。然后开始问一些框架端的内容，分布式训练的切割和device之间交流，后面follow-up的问题问了不同device之间communication怎么做，底层communication算法有啥，我答案里有ring-reduce，然后再深入问了reduce底层如何实现，我就不会了。问了会不会用cuda写东西，我说了解很浅。最后给了比较难的leetcode，题目不记得了，没写出来，挂
    - 楼主这边背景是做基座预训练算法端为主的，对框架端和RL的内容有一定了解（面试能凑合
    - 1.多头注意力，频率太高了。coding轮，概念轮都考。复习的点包括：时间/空间复杂度，优化（kv-cache，MQA，GQA），手写多头代码。各种Norm，这个频率也不低，不过比较标准的内容，没有啥特意要说的，有的考手写，有的考概念和理解（为什么管用）。
    - 2. 框架相关内容，各种并行方式，优缺点。DeepSpeed，Megatron可以看看源代码，Flash-Attention等内容。这个点也经常考代码题。
    - 3. BERT，GPT等比较主流大模型，一些细节，比如位置编码，训练loss，激活，架构些许不同这种。自回归重点。
    - 4. 大模型训练，这个可能主要是工作经验相关，经常问比如训练loss炸掉了，如何解决，一些技巧之类的。面试时有些面试官会问一些很细节的东西，感觉是在确认确实上手跑过基座训练不是吹水。
    - 5.数据预处理，BPE，tokenization，mask相关概念和对模型/训练影响，数据配比（有paper）。
    - 6. evaluation，如何评估大模型，安全性，有效性，公开数据，个别考过手写eval框架（多选，生成）。
    - 7. 根据投的岗位，多模态和RLHF内容可以适当看看。这俩感觉paper挺重要的，也大多研究岗位。楼主也少面了一些自动驾驶，RL啥的，不过结果不咋地。


