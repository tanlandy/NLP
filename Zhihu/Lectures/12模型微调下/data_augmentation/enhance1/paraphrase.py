from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import os, json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Phrase(BaseModel):
    text: str = Field(description="改写后的文本")


PROMPT_TEMPLATE = """
给定短语:
{phrase}

请用口语的方式paraphrase这个短语。例如:
公共区域和部分房间提供wifi: 有wifi | 无线上网 | 无线网 
中式餐厅: 中餐厅 | 中餐
国际长途电话: 国际电话 | 国际长途
免费市内电话: 免费电话 | 免费市话
酒店各处提供wifi: wifi全覆盖 | 无线上网 | 无线网

确保你输出的短语与原短语不同。
确保你输出的是中文（wifi这个词可以保留）。
按以下形式输出结果：
{format_instruction}
"""


class Paraphraser:
    '''
    为给定短语生成一个口语化的改写
    '''
    def __init__(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
        self.output_parser = PydanticOutputParser(pydantic_object=Phrase)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
        )

    def gen(self,phrase: str) -> str:
        response = self.chain.run(
            phrase=phrase,
        )
        phrase = self.robust_parser.parse(response)
        if ":" in phrase.text:
            phrase.text = phrase.text.split(":")[1]
        return phrase.text
