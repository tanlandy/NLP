# 07Langchain

## Goal

1. 如何使用 LangChain：一套在大模型能力上封装的工具框架
2. 如何用几行代码实现一个复杂的 AI 应用
3. 面向大模型的流程开发的过程抽象

## 写在前面

- LangChain 也是一套面向大模型的开发框架（SDK）
- LangChain 是 AGI 时代软件工程的一个探索和原型
- LangChain 迭代速度明显快于 Semantic Kernel，几乎明天一个版本
- 学习 Langchain 要关注接口变更

现在更推荐Langchain，相比SK

## LangChain 的核心组件

1. 模型 I/O 封装
   - LLMs：大语言模型
   - Chat Models：一般基于 LLMs，但按对话结构重新封装
   - PromptTemple：提示词模板
   - OutputParser：解析输出
2. 数据连接封装
   - Document Loaders：各种格式文件的加载器
   - Document Transformers：对文档的常用操作，如：split, filter, translate, extract metadata, etc
   - Text Embedding Models：文本向量化表示，用于检索等操作（啥意思？别急，后面详细讲）
   - Verctorstores: （面向检索的）向量的存储
   - Retrievers: 向量的检索
3. 记忆封装
   - Memory：这里不是物理内存，从文本的角度，可以理解为“上文”、“历史记录”或者说“记忆力”的管理
4. 架构封装
   - Chain：实现一个功能或者一系列顺序功能组合
   - Agent：根据用户输入，自动规划执行步骤，自动选择每步需要的工具，最终完成用户指定的功能
     - Tools：调用外部功能的函数，例如：调 google 搜索、文件 I/O、Linux Shell 等等
     - Toolkits：操作某软件的一组工具集，例如：操作 DB、操作 Gmail 等等
5. Callbacks

![langchain](langchain.png)

官方文档地址：https://python.langchain.com/docs/get_started

Agent是未来的趋势

### 1.2 模型的输入与输出

![model_io](model_io.jpg)

### 1.2.1 Prompt模板封装

PromptTemplate 可以在模板中自定义变量

```python

from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("给我讲个关于{subject}的笑话")
print(template)
print(template.format(subject='小明'))

```

#### ChatPromptTemplate

```python

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是{product}的客服助手。你的名字叫{name}"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

llm = ChatOpenAI()
prompt = template.format_messages(
        product="AGI课堂",
        name="瓜瓜",
        query="你是谁"
    )

llm.invoke(prompt)

```

### 1.2.2、从文件加载Prompt模板

JSON格式

```json

{
    "_type": "prompt",
    "input_variables": ["adjective", "content"],
    "template": "Tell me a {adjective} joke about {content}."
}

```

Template单独存放

```sh
cat simple_template.txt
```

```
Tell me a {adjective} joke about {content}.
```

```json

{
    "_type": "prompt",
    "input_variables": ["adjective", "content"],
    "template_path": "simple_template.txt"
}

```

加载方式

```python

from langchain.prompts import load_prompt

prompt = load_prompt("simple_prompt.json")

print(prompt.format(adjective="funny", content="Xiao Ming"))

```

### 1.3 输出封装 OutputParser

自动把 LLM 输出的字符串按指定格式加载。

LangChain 内置的 OutputParser 包括:

- ListParser
- DatetimeParser
- EnumParser
- PydanticParser  # 解析JSON形的
- XMLParser

等等

### 1.3.1 Pydantic (JSON) Parser

自动根据Pydantic类的定义，生成输出的格式说明

```python

from langchain_core.pydantic_v1  import BaseModel, Field, validator
from typing import List, Dict

# 定义你的输出对象
class Date(BaseModel):
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")

    # ----- 可选机制 --------
    # 你可以添加自定义的校验机制
    @validator('month')
    def valid_month(cls, field):
        if field <= 0 or field > 12:
            raise ValueError("月份必须在1-12之间")
        return field
        
    @validator('day')
    def valid_day(cls, field):
        if field <= 0 or field > 31:
            raise ValueError("日期必须在1-31日之间")
        return field

    @validator('day', pre=True, always=True)
    def valid_date(cls, day, values):
        year = values.get('year')
        month = values.get('month')

        # 确保年份和月份都已经提供
        if year is None or month is None:
            return day  # 无法验证日期，因为没有年份和月份

        # 检查日期是否有效
        if month == 2:
            if cls.is_leap_year(year) and day > 29:
                raise ValueError("闰年2月最多有29天")
            elif not cls.is_leap_year(year) and day > 28:
                raise ValueError("非闰年2月最多有28天")
        elif month in [4, 6, 9, 11] and day > 30:
            raise ValueError(f"{month}月最多有30天")

        return day

    @staticmethod
    def is_leap_year(year):
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            return True
        return False

```

```python

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser


model_name = 'gpt-4'
temperature = 0
model = ChatOpenAI(model_name=model_name, temperature=temperature)

# 根据Pydantic对象的定义，构造一个OutputParser
parser = PydanticOutputParser(pydantic_object=Date)

template = """提取用户输入中的日期。
{format_instructions}
用户输入:
{query}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
    # 直接从OutputParser中获取输出描述，并对模板的变量预先赋值
    partial_variables={"format_instructions": parser.get_format_instructions()} 
)

print("====Format Instruction=====")
print(parser.get_format_instructions())


query = "2023年四月6日天气晴..."
model_input = prompt.format_prompt(query=query)

print("====Prompt=====")
print(model_input.to_string())

output = model(model_input.to_messages())
print("====模型原始输出=====")
print(output)
print("====Parse后的输出=====")
date = parser.parse(output.content)
print(date)

```

补充：Auto-Fixing Parser

利用LLM自动根据解析异常修复并重新解析

`new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(model="gpt-4"))`

### 1.4、小结

1. LangChain 统一封装了各种模型的调用接口，包括补全型和对话型两种
2. LangChain 提供了 PromptTemplate 类，可以自定义带变量的模板
3. LangChain 提供了一些列输出解析器，用于将大模型的输出解析成结构化对象；额外带有自动修复功能。
4. 上述模型属于 LangChain 中较为优秀的部分；美中不足的是 OutputParser 自身的 Prompt 维护在代码中，耦合度较高。

<div class="alert alert-danger">
LangChain 的 PDFLoader 和 TextSplitter 实现都比较粗糙，实际生产中不建议使用。
</div>

### 2.3、内置的 RAG 实现 

```python

!pip install chromadb

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# 加载文档
loader = PyPDFLoader("llama2.pdf")
pages = loader.load_and_split()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents([pages[2].page_content,pages[3].page_content])

# 灌库
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# LangChain内置的 RAG 实现
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0), 
    retriever=db.as_retriever() 
)

query = "llama 2有多少参数？"
response = qa_chain.invoke(query)
print(response["result"])


```

### 2.4、小结

1. 这部分能力 LangChain 的实现非常粗糙；
2. 实际生产中，建议自己实现，更加可控，不建议用 LangChain 的工具。

## 三、记忆封装：Memory

### 3.1、对话上下文：ConversationBufferMemory

```python

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

history = ConversationBufferMemory()
history.save_context({"input": "你好啊"}, {"output": "你也好啊"})

print(history.load_memory_variables({}))

history.save_context({"input": "你再好啊"}, {"output": "你又好啊"})

print(history.load_memory_variables({}))

```

### 3.2、只保留2个窗口的上下文：ConversationBufferWindowMemory

```python

from langchain.memory import ConversationBufferWindowMemory

window = ConversationBufferWindowMemory(k=2)
window.save_context({"input": "第一轮问"}, {"output": "第一轮答"})
window.save_context({"input": "第二轮问"}, {"output": "第二轮答"})
window.save_context({"input": "第三轮问"}, {"output": "第三轮答"})
print(window.load_memory_variables({}))

```

### 3.3、通过 Token 数控制上下文长度：ConversationTokenBufferMemory

```python

from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI

memory = ConversationTokenBufferMemory(
    llm=ChatOpenAI(),
    max_token_limit=40
)
memory.save_context(
    {"input": "你好啊"}, {"output": "你好，我是你的AI助手。"})
memory.save_context(
    {"input": "你会干什么"}, {"output": "我什么都会"})

print(memory.load_memory_variables({}))

```

### 3.4、更多类型

- ConversationSummaryMemory: 对上下文做摘要
  - https://python.langchain.com/docs/modules/memory/types/summary
- ConversationSummaryBufferMemory: 保存 Token 数限制内的上下文，对更早的做摘要
  - https://python.langchain.com/docs/modules/memory/types/summary_buffer
- VectorStoreRetrieverMemory: 将 Memory 存储在向量数据库中，根据用户输入检索回最相关的部分
  - https://python.langchain.com/docs/modules/memory/types/vectorstore_retriever_memory

### 3.5、小结

1. LangChain 的 Memory 管理机制属于可用的部分，尤其是简单情况如按轮数或按 Token 数管理；
2. 对于复杂情况，它不一定是最优的实现，例如检索向量库方式，建议根据实际情况和效果评估；
3. 但是**它对内存的各种维护方法的思路在实际生产中可以借鉴**。

## 四、Chain 和 LangChain Expression Language (LCEL)

目前Langchain最精华核心的部分

LangChain Expression Language（LCEL）是一种声明式语言，可轻松组合不同的调用顺序构成 Chain。LCEL 自创立之初就被设计为能够支持将原型投入生产环境，**无需代码更改**，从最简单的“提示+LLM”链到最复杂的链（已有用户成功在生产环境中运行包含数百个步骤的 LCEL Chain）。

LCEL的一些亮点包括：

1. **流支持**：使用 LCEL 构建 Chain 时，你可以获得最佳的首个令牌时间（即从输出开始到首批输出生成的时间）。对于某些 Chain，这意味着可以直接从LLM流式传输令牌到流输出解析器，从而以与 LLM 提供商输出原始令牌相同的速率获得解析后的、增量的输出。

2. **异步支持**：任何使用 LCEL 构建的链条都可以通过同步API（例如，在 Jupyter 笔记本中进行原型设计时）和异步 API（例如，在 LangServe 服务器中）调用。这使得相同的代码可用于原型设计和生产环境，具有出色的性能，并能够在同一服务器中处理多个并发请求。

3. **优化的并行执行**：当你的 LCEL 链条有可以并行执行的步骤时（例如，从多个检索器中获取文档），我们会自动执行，无论是在同步还是异步接口中，以实现最小的延迟。

4. **重试和回退**：为 LCEL 链的任何部分配置重试和回退。这是使链在规模上更可靠的绝佳方式。目前我们正在添加重试/回退的流媒体支持，因此你可以在不增加任何延迟成本的情况下获得增加的可靠性。

5. **访问中间结果**：对于更复杂的链条，访问在最终输出产生之前的中间步骤的结果通常非常有用。这可以用于让最终用户知道正在发生一些事情，甚至仅用于调试链条。你可以流式传输中间结果，并且在每个LangServe服务器上都可用。

6. **输入和输出模式**：输入和输出模式为每个 LCEL 链提供了从链的结构推断出的 Pydantic 和 JSONSchema 模式。这可以用于输入和输出的验证，是 LangServe 的一个组成部分。

7. **无缝LangSmith跟踪集成**：随着链条变得越来越复杂，理解每一步发生了什么变得越来越重要。通过 LCEL，所有步骤都自动记录到 LangSmith，以实现最大的可观察性和可调试性。

8. **无缝LangServe部署集成**：任何使用 LCEL 创建的链都可以轻松地使用 LangServe 进行部署。

原文：https://python.langchain.com/docs/expression_language/

举个例子

```python

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from enum import Enum

# 输出结构
class SortEnum(str, Enum):
    data = 'data'
    price = 'price'

class OrderingEnum(str, Enum):
    ascend = 'ascend'
    descend = 'descend'

class Semantics(BaseModel):
    name: Optional[str] = Field(description="流量包名称",default=None)
    price_lower: Optional[int] = Field(description="价格下限",default=None)
    price_upper: Optional[int] = Field(description="价格上限",default=None)
    data_lower: Optional[int] = Field(description="流量下限",default=None)
    data_upper: Optional[int] = Field(description="流量上限",default=None)
    sort_by: Optional[SortEnum] = Field(description="按价格或流量排序",default=None)
    ordering: Optional[OrderingEnum] = Field(description="升序或降序排列",default=None)

# OutputParser
parser = PydanticOutputParser(pydantic_object=Semantics)

# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "将用户的输入解析成JSON表示。输出格式如下：\n{format_instructions}\n不要输出未提及的字段。",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 模型
model = ChatOpenAI(temperature=0)

# LCEL 表达式
runnable = (
    {"query": RunnablePassthrough()} | prompt | model | parser
)

# 运行
print(runnable.invoke("不超过100元的流量大的套餐有哪些"))

```

复杂点的，RAG结合向量数据库

```python

from langchain_openai import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma

# 向量数据库
vectorstore = Chroma.from_texts(
    [
        "Sam Altman是OpenAI的CEO", 
        "Sam Altman被解雇了",
        "Sam Altman被复职了"
    ], embedding=OpenAIEmbeddings()
)

# 检索接口
retriever = vectorstore.as_retriever()

# Prompt模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain
retrieval_chain = (
    {"question": RunnablePassthrough(),"context": retriever}
    | prompt
    | model
    | StrOutputParser()
)

retrieval_chain.invoke("OpenAI的CEO是谁")

```

**注意**: 在当前的文档中 LCEL 产生的对象，被叫做 runnable 或 chain，经常两种叫法混用。本质就是一个自定义调用流程。

### 通过 LCEL，还可以实现

1. 配置运行时变量：https://python.langchain.com/docs/expression_language/how_to/configure
2. 故障回退：https://python.langchain.com/docs/expression_language/how_to/fallbacks
3. 并行调用：https://python.langchain.com/docs/expression_language/how_to/map
4. 逻辑分支：https://python.langchain.com/docs/expression_language/how_to/routing
5. 调用自定义流式函数：https://python.langchain.com/docs/expression_language/how_to/generators
6. 链接外部Memory：https://python.langchain.com/docs/expression_language/how_to/message_history

更多例子：https://python.langchain.com/docs/expression_language/cookbook/

## 五、智能体架构：Agent

将大语言模型作为一个推理引擎。给定一个任务，智能体自动生成完成任务所需的步骤，执行相应动作（例如选择并调用工具），直到任务完成。

![Agent](agent-overview.png)

```python

from langchain import SerpAPIWrapper
from langchain.tools import Tool, tool

search = SerpAPIWrapper()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
]

import calendar
import dateutil.parser as parser
from datetime import date

# 自定义工具
@tool("weekday")
def weekday(date_str: str) -> str:
    """Convert date to weekday name"""
    d = parser.parse(date_str)
    return calendar.day_name[d.weekday()]

tools += [weekday] 

```

### 5.3 智能体类型：ReAct

![ReAct](ReAct.png)

```python

!pip install google-search-results
!pip install langchainhub

from langchain import hub
import json

# 下载一个现有的 Prompt 模板
prompt = hub.pull("hwchase17/react")

print(prompt.template)

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# 定义一个 agent: 需要大模型、工具集、和 Prompt 模板
agent = create_react_agent(llm, tools, prompt)
# 定义一个执行器：需要 agent 对象 和 工具集
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行
agent_executor.invoke({"input": "周杰伦生日那天是星期几"})

```

<div class="alert alert-success">
<b>划重点：</b>
<ol>
<li>ReAct 是比较常用的 Planner</li>
<li>SelfAskWithSearch 更适合需要层层推理的场景（例如知识图谱）</li>
<li>OpenAI Assistants 不是万能的，LangChain 的官方文档里也不强调述接口了</li>
<li>Agent落地应用需要更多细节，后面课程中我们会专门讲 Agent 的实现</li>
</ol>
</div>

## 六、LangServe

LangServe 用于将 Chain 或者 Runnable 部署成一个 REST API 服务。

```python

# 安装 LangServe
!pip install "langserve[all]"

# 也可以只安装一端
# !pip install "langserve[client]"
# !pip install "langserve[server]"

```

### 6.1、Server端

```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9999)
```

### 6.2、Client端

```python
import requests

response = requests.post(
    "http://localhost:9999/joke/invoke",
    json={'input': {'topic': '小明'}}
)
print(response.json())
```

## LangChain 与 Semantic Kernel 对比

| 功能/工具           | LangChain                       | Semantic Kernel                  |
|-------------------|:---------------------------------:|:----------------------------------:|
| 版本号        |  0.1.0  | python-0.4.4.dev  |
| 适配的 LLM        | 多   | 少 + 外部生态   |
| Prompt 工具        | 支持    | 支持     |
| Prompt 函数嵌套    | 需要通过 LCEL | 支持        |
| Prompt 模板嵌套    | 不支持  | 不支持       |
| 输出解析工具       | 支持  | 不支持  |
| 上下文管理工具           | 支持 | C#版支持，Python版尚未支持  |
| 内置工具           | 多，但良莠不齐  | 少 + 外部生态  |
| 三方向量数据库适配           | 多 | 少 + 外部生态  |
| 服务部署 | LangServe | 与 Azure 衔接更丝滑
| 管理工具 | LangSmith/LangFuse | Prompt Flow


## 总结

1. LangChain 随着版本迭代可用性有明显提升
2. 使用 LangChain 要避开存在大量代码内 Prompt 的模块
3. 它的内置基础工具，建议充分测试效果后再决定是否使用

## 作业

用 LangChain 重构 ChatPDF 的作业

看到94min，老师开始答疑
