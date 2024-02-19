# 生产级LLM应用

## Goal

1. 系统性维护、测试、监控一个 LLM 应用
2. 学习使用主流的工具完成上述工作

## 维护一个生产级的 LLM 应用，我们需要做什么？

1. 各种指标监控与统计：访问记录、响应时长、Token用量、计费等等
2. 调试 Prompt
3. 测试/验证系统的相关评估指标
4. 数据集管理（便于回归测试）
5. Prompt 版本管理（便于升级/回滚）

## 针对以上需求，我们介绍三个生产级 LLM App 维护平台

1. 重点讲解 **LangFuse**: 开源 + SaaS，LangSmith 平替，可集成 LangChain 也可直接对接 OpenAI API；
2. 简单讲解 **LangSmith**: LangChain 的官方平台，SaaS 服务，非开源，**目前需要排队注册**；使用的话首选
3. 简单讲解 **Prompt Flow**：微软开发，开源 + Azure AI云服务，可集成 Semantic Kernel（但貌合神离）。

## 1、LangFuse

开源，支持 LangChain 集成或原生 OpenAI API 集成

官方网站：<https://langfuse.com/>

项目地址：<https://github.com/langfuse>

1. 通过官方云服务使用：
   - 注册: cloud.langfuse.com
   - 创建 API Key

2. 通过 Docker 本地部署

```sh
# Clone repository
git clone https://github.com/langfuse/langfuse.git
cd langfuse
 
# Run server and db
docker compose up -d
```

### 1.2、通过 LangChain 的回调集成

```python

from langfuse.callback import CallbackHandler

handler = CallbackHandler(
    trace_name="SayHello",
    user_id="wzr",
)

```

```python

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

#from langchain.chat_models import ErnieBotChat
from langchain.schema import HumanMessage
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-3.5-turbo-0613")

prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("Say hello to {input}!") 
])


# 定义输出解析器
parser = StrOutputParser()

chain = (
    {"input":RunnablePassthrough()} 
    | prompt
    | model
    | parser
)

```

```python

chain.invoke(input="AGIClass", config={"callbacks":[handler]})

```

### 1.3 用Session记录一个用户的多轮对话

```python

import uuid
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage, #等价于OpenAI接口中的assistant role
    HumanMessage, #等价于OpenAI接口中的user role
    SystemMessage #等价于OpenAI接口中的system role
)


llm = ChatOpenAI()

messages = [
    SystemMessage(content="你是AGIClass的课程助理。"), 
]

handler = CallbackHandler(
    user_id="wzr",
    trace_name="test_chat",
    session_id=str(uuid.uuid4())
)  # 用于记录用户的多轮对话

while True:
    user_input=input("User: ")
    if user_input.strip() == "":
        break
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages,config={"callbacks":[handler]})
    print("AI: "+response.content)
    messages.append(response)

```

## 1.4 数据集与测试

# 总结

管理一个 LLM 应用的全生命周期，需要用到以下工具：

1. 调试 Prompt 的 Playground
2. 测试/验证系统的相关指标
3. 数据集管理
4. 各种指标监控与统计：访问量、响应时长、Token费等等

根据自己的技术栈，选择：

1. LangFuse：开源平台，支持 LangChain 和原生 OpenAI API
2. LangSmith: LangChain 的原始管理平台
3. Prompt Flow：开源平台，支持 Semantic Kernel

# 作业

选择一个工具平台，对自己之前开发的系统或模型做批量测试

看到111min，老师继续答疑
