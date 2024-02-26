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

请补充生成一轮对话，模拟user表示愿意入住这个酒店。你可以使用不同的说法。用很口语的方式表达。亲切、礼貌一些。
并模拟assistant，回答“好的，祝您入住愉快”。
对话必须先由user开始，然后assistant回复。每人只一轮。
只输出一轮对话，不需要输出多轮对话，不要重复以上的对话内容。
确保你输出与原句语言保持一致。
按以下形式输出结果：
{format_instruction}
"""


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Turn(BaseModel):
    role: Role = Field(description="对话角色")
    content: str = Field(description="对话内容")


class Bye(BaseModel):
    dialog: List[Turn] = Field(description="对话内容")


class ByeGen:
    '''
    为对话生成一轮结束语：例如“好的，祝您入住愉快”
    '''
    def __init__(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
        self.output_parser = PydanticOutputParser(pydantic_object=Bye)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
        )

    def gen(self, dialog: List) -> List|None:
        response = self.chain.run(
            dialog=json.dumps(dialog, indent=4, ensure_ascii=False),
        )
        try:
            qa = json.loads(response)
            if isinstance(qa,list):
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
