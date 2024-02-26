import random

from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import os, json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from enum import Enum

PROMPT_TEMPLATE = """
给定一段user与assistant的对话，其中search表示搜索条件，return表示返回的结果:
```
{dialog}
```

请补充生成一轮对话，模拟user，针对assistant提供的酒店，随机选则酒店的任意一个属性进行比较性提问，包括：价格、评分、地址、电话、是否包含某设置等。
例如: 
{examples}
etc.

请用很口语的方式提问。Try your best to paraphrase the question!尽可能改写问题！
注意不要重复提问之前对话中assistant已经提供的信息。

然后模拟assistant，根据酒店的实际属性值，回答该问题。答案必须与给出的酒店的真实信息一致。
对话必须先由user提问，然后assistant回答。每人只一轮。
只输出一轮对话，不需要输出多轮对话，不要重复以上的对话内容。
确保你输出与原句语言保持一致。
按以下形式输出结果：
{format_instruction}
"""

examples = [
    "“哪个便宜？”",
    "“哪个高档一些？”",
    "“哪个评分高？”",
    "”哪个有wifi”",
    "“有带商务中心的吗”",
    "“有带棋牌室的吗”",
]

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    role: Role = Field(description="对话角色")
    content: str = Field(description="对话内容")


class QA(BaseModel):
    dialog: List[Turn] = Field(description="对话内容")


class QAGen2:
    '''
    为对话生成一轮对比型问答：例如哪个酒店便宜、高档、评分高、有wifi等
    '''
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4", temperature=0.8)
        self.output_parser = PydanticOutputParser(pydantic_object=QA)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
        )

    def gen(self, dialog: List) -> List|None:
        random.shuffle(examples)
        response = self.chain.run(
            dialog=json.dumps(dialog, indent=4, ensure_ascii=False),
            examples="\n".join(examples),
        )
        try:
            qa = json.loads(response)
            if isinstance(qa, list):
                if len(qa) != 2:
                    raise Exception("Invalid format")
                for turn in qa:
                    if "role" not in turn or "content" not in turn:
                        raise Exception("Invalid format")
                if qa[0]["role"] != "user" or qa[1]["role"] != "assistant":
                    raise Exception("Invalid format")
                ans = qa
            else:
                raise Exception("Invalid format")
        except:
            qa = self.robust_parser.parse(response)

            ans = []
            for turn in qa.dialog:
                t = {
                    "role": turn.role,
                    "content": turn.content,
                }
                ans.append(t)
            if len(ans) != 2:
                return None
        return ans
