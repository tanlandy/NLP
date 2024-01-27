# 什么是Assistants API

2023.11.7在Dev Day发布
是一个集大成者：chat、执行代码、读取外部知识、使用外部能力、持久化
是“GPTs背后的API”

官⽅Web界⾯可体验/调试

• <https://platform.openai.com/playground>

• <https://platform.openai.com/assistants>

官⽅⽂档

• <https://platform.openai.com/docs/assistants/overview>

• <https://platform.openai.com/docs/api-reference/assistants>

# 为什么Assistants API

够全、够快、够简单

# Demo展示及能力介绍

滴滴打车车费预估，使用OpenAI Assistants API结合RAG、Code interpreter、Function call三大能力实现

移动端 + PC端的demo

# Demo框架及具体实现

## 开发流

一开始就run起来

## 数据流

只要页面有变化，就从开头执行一遍代码

## Widgets组件

<https://docs.streamlit.io/library/api-reference>

![项目逻辑流程图](<Screenshot 2024-01-26 at 7.02.43 PM.png>)

# 从Demo到实际应用的Gap

没有处理安全问题

- 带偏干别的
- 违规违法问题
逻辑推理能力一般
- 思维链思维树
格式问题
- 格式问题
- 指令调优

模型相关问题

- 贵
- 国内不能toC
- 非垂类，领域专属模型
- RAG限制

项目问题

- 产品设计
- 部署交付
- 产品运营

接下来最卷的岗位：

- 后端工程师
- 前端工程师
- 传统程序员

乘风破浪的岗位：

- 产品经理

大模型工程师的职业发展：

- 继续本岗位：大大提升效率，横向卷同行，纵向卷上下游
- 成为超级个体：独立开发者
- 成为大模型训练师：做公司的技术核心，使用开源模型，训练出垂类模型，解决公司的问题，部署模型，部署向量数据库
- 独立创业：凭大模型垂直落地能力解决独有场景

AI浪潮创业：

- 快速迭代，日新月异
- 天马行空，资本乱入
- 乱世枭雄，造福神话
