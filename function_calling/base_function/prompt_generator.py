from function_calling.base_function.base_tool import BaseTool
from enum import Enum


class PrompGeneratorType(Enum):
    JSONTOOL = "jsontool"


class PrompGenerator(BaseTool):
    def __init__(self, args: dict, prompt_type: PrompGeneratorType):
        super().__init__(args)
        self.prompt_type = prompt_type
        self.system_prompt = ""
        self.user_prompt = ""

    def get_prompt_jsontool(self, query: dict) -> dict:
        system_prompt = f"""{self.args_prompt["system"]}
            {{rag}}
            You are an assistant that has access to the following set of tools.
            Here are the names and descriptions for each tool:

            {{rendered_tools}}

            Given the user input, return the name and input of the tool to use.
            Return your response as a JSON blob with 'name' and 'arguments' keys.

            The `arguments` should be a dictionary, with keys corresponding
            to the argument names and the values corresponding to the requested values.
            """.format(**query)
        user_prompt = self.args_prompt["user"].format(**query)
        output_dict = {"system": system_prompt, "user": user_prompt}
        return output_dict

    def __call__(self, query: dict) -> dict:
        if self.prompt_type == PrompGeneratorType.JSONTOOL:
            return self.get_prompt_jsontool(query)





