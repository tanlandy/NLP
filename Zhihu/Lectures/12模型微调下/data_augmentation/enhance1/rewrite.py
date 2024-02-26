from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import os, json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Utterance(BaseModel):
    text: str = Field(description="重写后的句子文本")


PROMPT_TEMPLATE = """
给定句子:
{utterance}
请将句中的"{phrase}"替换为"{replacement}"。
并重新组织该句子，保证其意思不变的前提下，语言通顺。
确保你输出与原句语言保持一致。
按以下形式输出结果：
{format_instruction}
"""


class UtteranceRewriter:
    '''
    替换句子中的短语，并重写句子使其通顺
    '''
    def __init__(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=Utterance)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=llm)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            format_instruction=self.output_parser.get_format_instructions(),
        )
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
        )

    def rewrite(self, utterance: str, phrase: str, replacement: str) -> str:
        response = self.chain.run(
            utterance=utterance,
            phrase=phrase,
            replacement=replacement,
        )
        utterance = self.robust_parser.parse(response)
        return utterance.text
