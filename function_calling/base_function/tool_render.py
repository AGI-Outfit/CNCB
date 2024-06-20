from typing import List
from typing import Any, Dict, Optional, TypedDict, Union
import inspect

class ToolRender:
    def __init__(self, tools: Union[List[classmethod], None] = None):
        self.tools = tools
        self.rendered_tools = ""
        self.tools_define_strs = []
        self.tools_docs_strs = []
        if self.tools is not None:
            self.set_tools(tools)

    def set_tools(self, tools):
        self.tools = tools
        for tool in self.tools:
            source_lines, _ = inspect.getsourcelines(tool)
            self.tools_define_strs.append(source_lines[0].strip())
            self.tools_docs_strs.append(tool.__doc__)

    def get_tool_render(self):
        """
        获取工具渲染字符串信息
        :return:
        """
        output_str = ""
        for i in range(len(self.tools_docs_strs)):
            define_str = self.tools_define_strs[i][4:-1]
            doc = self.tools_docs_strs[i]
            output_str += "\n" + define_str + " - " + doc
        output_str += "\n"
        self.rendered_tools = output_str
        return output_str
