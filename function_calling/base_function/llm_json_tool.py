from .base_tool import BaseTool
from .rag_generator import RAGGenerator
from .prompt_generator import PrompGenerator, PrompGeneratorType
from .tool_render import ToolRender
from .json_tool_parser import JsonToolParser

from openai import OpenAI

from typing import Any, Dict, Optional, TypedDict, List


class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]


class LLMJsonTool(BaseTool):
    def __init__(self, args: dict):
        super().__init__(args)
        self.debug_mode = self.args_base["debug_mode"]

        self.prompt_generator = PrompGenerator(args=self.args, prompt_type=PrompGeneratorType.JSONTOOL)
        self.llm_base = OpenAI(api_key=self.args_base["api_key"], base_url=self.args_base["base_url"])
        self.level_print("Success: Base LLM Init.", 1)

        self.rag_generator = RAGGenerator(args=self.args)
        self.tools = []
        self.tool_render = ToolRender()

        self.json_parser = JsonToolParser(args=self.args)

    def chat(self, user: str, system: str = ""):
        completion = self.llm_base.chat.completions.create(
            model=self.args_base["model_name"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
                ]
            ,
            max_tokens=1000
        )

        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def exec(self, tool_call_request: ToolCallRequest):
        if not self.debug_mode:
            if tool_call_request["name"] == "none" or tool_call_request["name"] is None:
                return None
            func = getattr(self, tool_call_request["name"])

            return func(**tool_call_request["arguments"])
        return ""

    def __call__(self, query: str):
        rag = self.rag_generator(query)
        rendered_tools = self.tool_render.rendered_tools

        prompt_input_dict = {"input": query, "rag": rag, "rendered_tools": rendered_tools}
        prompt_output_dict = self.prompt_generator(prompt_input_dict)
        self.level_print(prompt_output_dict, 3)

        return self.chat(system=prompt_output_dict["system"], user=prompt_output_dict["user"])

