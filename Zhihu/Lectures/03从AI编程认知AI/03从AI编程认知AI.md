# 03从AI编程认知AI

## 场景

其中在软件开发过程中，已验证能明确提效的场景：

- 市场调研
- 需求分析
- PRD 撰写
- 图形元素绘制
- 技术选型
- 从需求文本生成代码
- 从设计图生成代码
- 代码审查
- 编写测试用例
- 运维
- API文档调用
- 协议解析
- 跨语言迁移：两周的时间，2个同事将一个40个API接口的go工程迁移成一个java工程

解答技巧：

1. 内容是有规律的
2. 内容有参考格式说明

## 定义

在软件开发过程中，通过AI辅助，减少重复性工作，提高编程效率的行为，我们叫AI编程

### 使用技巧

- 所有 prompt engineering 的技巧都有效，可以把代码、错误信息直接粘贴进去
- 任何技术相关的问题都可以问，比自己搜索效率高很多

参考：ChatALL 大量依赖 GPT-4。过程实录：<https://chat.openai.com/share/ecfafa99-aaad-4432-a40c-81c261376758> （对话轮次非常多加载慢，所以几乎总是 404）。如果打不开，别到群里问题，看![这个图](gpt-4-chatall.png)

### 如何理解AI能编写程序

#### **编程能力是大模型各项能力的天花板**

- 「编程」是目前大模型能力最强的垂直领域，甚至超越了对「自然语言」本身的处理能力。因为：
  - 训练数据质量高
  - 结果可衡量
  - 编程语言无二义性

- 知道怎么用好 AI 编程，了解它的能力边界、使用场景，就能类比出在其他领域 AI 怎么落地，能力上限在哪
  - 此观点是我们原创，在分享过数次后，GitHub 官方发了一篇文章：[How to build an enterprise LLM application: Lessons from GitHub Copilot](https://github.blog/2023-09-06-how-to-build-an-enterprise-llm-application-lessons-from-github-copilot/)。当然，这只是巧合。
- 应用层： prompt engineering。Prompt 中包含：
  1. 光标前和光标后的代码片段
  1. 其它相关代码片段。当前文件和其它打开的 tab 里的代码被切成每个 60 行的片段，用 [Jaccard 相似度](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8D%A1%E5%B0%94%E6%8C%87%E6%95%B0)评分，取高分的
  1. 被取用的代码片段的路径。用注释的方式插入，例如：`# filepath: foo/bar.py`，或者 `// filepath: foo.bar.js`
- 补全格式：在函数定义、类定义、if-else 等之后，会补全整段代码，其它时候只补全当前行
![代码补全](life-of-a-completion.webp)
方法论和数据流转，自己公司如有需要构建代码补全工具，可参考

了解更多

- [Inside GitHub: Working with the LLMs behind GitHub Copilot](https://github.blog/2023-05-17-inside-github-working-with-the-llms-behind-github-copilot/)
- [How GitHub Copilot is getting better at understanding your code](https://github.blog/2023-05-17-how-github-copilot-is-getting-better-at-understanding-your-code/)
- [A developer’s guide to prompt engineering and LLMs](https://github.blog/2023-07-17-prompt-engineering-guide-generative-ai-llms/)

光标停留10ms，即开始补全提示

### 可以学习到什么

- 组织高质量的数据
- 对工作场景针对性的设计交互方式
- 数据的增加能让原来单一数据源的训练效果提升

### 产品理念

产品理念：**不需要改变用户的习惯，只是在用户需要的时候给以提醒**

### 市面上有哪些工具

1. [Bito](https://bito.ai/) - 比 Copilot 还多些创新
2. [Amazon CodeWhisperer](https://aws.amazon.com/codewhisperer/) - 代码补全，免费。AWS 相关的编程能力卓越。其它凑合
3. [Cursor](https://www.cursor.so/) - AI first 的 IDE。被逼开源
4. [Tabnine](https://www.tabnine.com/) - 代码补全，个人基础版免费
5. [Tongyi Lingma](https://tongyi.aliyun.com/lingma) -- 代码补全，免费。阿里云相关。

### 一些技巧

1. 代码有了，再写注释，更省力
2. 改写当前代码，可另起一块新写，AI 补全得更准，完成后再删旧代码
3. `Cmd/Ctrl + →` 只接受一个 token
4. 如果有旧代码希望被参考，就把代码文件在新 tab 页里打开

### 问自己两个问题

1. 我的时间都消耗在哪里？
2. 怎么让 AI 帮我省时间？

划重点：凡是「输入和输出都是文本」的场景，都值得尝试用大模型提效。

### AI 幻觉的必然性，和 AI 能力的天花板

AI 祖师爷阿兰·图灵 1947 年在[伦敦数学学会的演讲](https://www.vordenker.de/downloads/turing-vorlesung.pdf)上说过：

> ...if a machine is expected to be infallible, it cannot also be intelligent.

翻译成中文是：

> 如果一个机器绝对不会犯错，它就不可能是智能的。

拥有类人的智能，就一定和人类一样，说胡话、瞎话、错话，说有争议的话。

所以，AI 的幻觉不可完全消除，只能尽可能压制。

所以，使用者的判断力，决定了 AI 能力的上限。 就像团队领导，是团队的上限一样。

我们原创了两个定律：

AI 能力定律：

AI 能力的上限，是使用者的判断力

$\text{AI 能力} = \min(\text{AI 能力}, \text{使用者判断力})$

AI 提效定律：

AI 提升的效率，与使用者的判断力成正比，与生产力成反比

$\text{效率提升幅度} = \frac{\text{使用者判断力}}{\text{使用者生产力}}$

解读：

1. 使用者的判断力，是最重要的
2. 提升判断力，比提升实操能力更重要。所谓「眼高手低」者的福音
3. 广阔的视野是判断力的养料

如果以上的公式是公理的话，我们是否可以得出如下一条定理

$\text{编码人员的判断力} = \text{需求的理解准确度} · \text{代码的阅读量} $

$\text{效率提升幅度} = \frac{\text{需求的理解准确度} · \text{代码的阅读量}}{\text{自己手写代码的速度}}$

## 一些自动编写代码的机器人

编程 Agents
前面的工具都是帮你编程，而 Agent 是完全代替你编程。甚至能做更多……

警告：以下介绍的现在都只能玩玩，做技术探讨，还远没到实用的程度。

### MetaGPT：多智能体元编程框架

<https://github.com/geekan/MetaGPT>

它不只写代码，而且写文档、画图。详见讲座课里 MetaGPT 核心工程师的分享。

核心 prompts：<https://github.com/geekan/MetaGPT/tree/main/metagpt/prompts>

评价：

- 让 agent 模拟岗位这个思路挺有意思。未来的公司就是人和 agent 混合的，这样的系统架构更匹配公司治理
- 所以 MetaGPT 其实是个多 Agent 开发框架，而不是单纯的编程工具

## Agent 的落地难题

Agent落地条件：

1. LLM能力足够强
2. 使用场景足够窄

## 对于企业团队如何推动AI编程

### 公司内部使用

1. 先养成提问的习惯
2. 提供基本的编程认知：AI编程不是神，AI编程是一个结对编程的小伙伴。
3. 购买好账号，让团队都能使用
4. 真的使用起来

#### 可能得声音

- AI提示给我的代码，如果我完全按照他的写就死了...（参考：AI效能定律）
- AI也就写写简单的函数，没法完成复杂的架构... (参考: 金字塔架构解析)
  - 先以架构师的身份，让AI辅助你对对架构进行选型： 通过需求文档和业务文档，让AI给出架构的建议
  - 然后以开发者的身份，让AI辅助你写业务代码 （逐层拆分向下写）
- 我觉得AI怎么都不对（参考Google）

#### 践行过的方法

让每个员工每天至少给GPT提三个问题，养成提问的习惯是一切的基础

推荐一本书 [《学会提问》](https://book.douban.com/subject/35513147/)

## 课后

学习websocket
