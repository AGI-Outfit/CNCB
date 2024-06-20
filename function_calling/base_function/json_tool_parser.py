from .base_tool import BaseTool
from langchain_core.output_parsers import JsonOutputParser
from typing import Union

from inspect import signature
from typing import Any, Callable, Optional, TypeVar, Generic, get_type_hints

T = TypeVar('T')


# def function_to_pydantic_model(func: Callable[..., T]) -> type:
#     # 获取函数签名
#     sig = signature(func)
#     annotations = get_type_hints(func)
#
#     # 构建Pydantic模型的字段字典
#     fields = {}
#     for param_name, param in sig.parameters.items():
#         # 忽略自变量（如*args, **kwargs）
#         if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
#             continue
#
#         # 获取参数的类型提示，如果没有则设为Any
#         field_type = annotations.get(param_name, Any)
#
#         # 处理默认值
#         default = param.default if param.default != param.empty else ...
#
#         # 添加字段到字典
#         fields[param_name] = (field_type,)
#
#     # 动态创建Pydantic模型类
#     model_name = func.__name__
#     model_class = type(model_name, (BaseModel,), fields)
#
#     return model_class

class JsonToolParser(BaseTool):
    def __init__(self, args: dict):
        super().__init__(args)
        self.json_parser = JsonOutputParser()

    def remove_annotation(self, input_str, annotations=None):
        if annotations is None:
            annotations = ["#", "/"]
        a = input_str
        for annotation in annotations:
            while 1:
                if annotation in a:
                    b = a.index(annotation)
                    c = a.find('\n', b)
                    a = a.replace(a[b:c], "")
                else:
                    break
        return a

    def evaluate_jsontool(self, json_dict: dict, tools: list[classmethod]) -> bool:
        flag_func_call = True
        output_str = ""
        # tools_name = [tool.__code__.co_name for tool in tools]
        tools_func = {tool.__code__.co_name: tool for tool in tools}
        # 检查每个函数是否包含必要的属性。
        if "name" not in json_dict:
            output_str += "Fail: Some json function has no property \'name\'\n"
            self.level_print(output_str, 1)
            return False
        if "arguments" not in json_dict:
            json_dict["arguments"] = {}

        # 验证函数是否在提供的tools中。
        if json_dict["name"] not in tools_func:
            output_str += "Fail: Json function \'{0}\' is not in tools\n".format(json_dict["name"])
            self.level_print(output_str, 1)
            return False

        # 验证参数名参数类型是否对应。
        sig = signature(tools_func[json_dict["name"]])
        # print(sig.parameters.keys())
        for arg in json_dict["arguments"]:
            if arg not in sig.parameters.keys():
                output_str += "Fail: Json function \'{0}\''s \'{1}\' is not in \'{2}\'\n".format(json_dict["name"],
                                                                                                 arg,
                                                                                                 json_dict["name"])
                flag_func_call = False
                self.level_print(output_str, 1)
                return False
            if type(json_dict["arguments"][arg]) is not sig.parameters[arg].annotation:
                output_str += "Fail: Json function \'{0}\''s \'{1}\' is not in \'{2}\'\n".format(json_dict["name"],
                                                                                                 arg,
                                                                                                 json_dict["name"])
                flag_func_call = False
                self.level_print(output_str, 1)
                return False

        return True

    def __call__(self, input_str: str) -> Union[dict, None]:
        input_str = self.remove_annotation(input_str)
        try:
            self.level_print("Success: Json parse.\n", 3)
            return self.json_parser.invoke(input_str)
        except Exception as e:
            self.level_print("Error: Invalid json.", 1)
            self.level_print(input_str + "\n", 1)
            return None