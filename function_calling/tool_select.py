from function_calling.base_function.base_tool import BaseTool
from langchain_core.tools import tool
from langchain_core.tools import render_text_description
from langchain_core.prompts import ChatPromptTemplate
from cnc_baseline.function_calling.exception_handle.jsonparser_jsoncall import my_chain_invoke, tool_custom_exception
from cnc_baseline.function_calling.cnc_blank import CNCBlank


# global_args = {}
# global_command = ""

class ToolSelect(BaseTool):
    def __init__(self, args):
        super().__init__(args)
        global global_args

        global_args = self.args

        self.tools = [
            cnc_blanking,
            cnc_tool_change]
        self.rendered_tools = render_text_description(self.tools)

        self.few_shot_prompt = None
        self.init_prompt()
        self.chain = self.few_shot_prompt | self.llm_base | tool_custom_exception

    def init_prompt(self):
        # rag =
        system = f"""{self.args_prompt["system"]}
            {{rag}}
            You are an assistant that has access to the following set of tools.
            Here are the names and descriptions for each tool:

            {self.rendered_tools}

            Given the user input, return the name and input of the tool to use.
            Return your response as a JSON blob with 'name' and 'arguments' keys.

            The `arguments` should be a dictionary, with keys corresponding
            to the argument names and the values corresponding to the requested values.
            """
        self.few_shot_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                self.args_prompt["example"],
                ("user", self.args_prompt["user"])
            ]
        )
        self.level_print("Prompt: " + str(self.few_shot_prompt.invoke({"rag": "11", "input": "如何进行苹果的CNC下料？"})), 4)

    # 就算config.json中设置了use_rag，这里如果是False也不启动rag
    def __call__(self, input_message, use_rag=True):
        global global_command
        global_command = input_message
        # 使用rag
        rag = ""
        if use_rag:
            rag = self.get_rag_content(input_message)
        # 获取json形式的输出列表
        json_outputs = my_chain_invoke(self.chain, self.tools, input_message,
                                       rag=rag, retry_times=self.args_base["retry_times"])
        self.level_print("LLM Output: " + str(json_outputs), 3)
        # 执行函数
        index = 0
        while 1:
            if (json_outputs is None or index >= len(json_outputs)):
                break
            self.level_print(json_outputs[index], 2)
            output = self.exec(json_outputs[index])
            if output != "":
                json_outputs = my_chain_invoke(self.chain, self.tools, input_message + output,
                                               rag=rag, retry_times=self.args_base["retry_times"])
                self.level_print("LLM Output: " + str(json_outputs), 3)
            index += 1


        return json_outputs

######################## 设置大模型工具 ########################
@tool
def cnc_blanking(object_name: str) -> str:
    """执行CNC下料任务"""
    global global_args
    global global_command
    cnc_blank = CNCBlank(global_args)
    cnc_blank(global_command)
    return ""

@tool
def cnc_tool_change(tool_type: str) -> str:
    """执行CNC切换工具任务"""
    print("执行CNC切换工具任务")
    return ""



