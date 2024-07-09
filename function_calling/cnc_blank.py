from function_calling.base_function.llm_json_tool import LLMJsonTool
from function_calling.base_function.speech_recog import SpeechRecognizer
class CNCBlank(LLMJsonTool):
    def __init__(self, args):
        super().__init__(args)

        self.tools = [
                 self.detect_object_location,
                 self.open_suck,
                 self.move_end,
                 self.move_to_origin,
                 self.open_the_door,
                 self.close_suck]
        self.tool_render.set_tools(self.tools)
        self.tool_render.get_tool_render()
        self.speech_recognizer = SpeechRecognizer(self.args)

    # 就算config.json中设置了use_rag，这里如果是False也不启动rag
    def __call__(self, query, use_rag=True):
        """
        使对象实例可被调用，处理查询并生成相应的输出。

        :param query: 用户的查询字符串
        :param use_rag: 是否使用RAG（Region of Interest Attention Graph）模型，默认为True
        """
        # 语音识别获得Prompt
        record_path = self.speech_recognizer.record()
        # record_path = "output/speechrecog/record-2024-06-12_15-21-52.wav"
        recog_result = ""
        if record_path is not None:
            recog_result = self.speech_recognizer.speech_recog(record_path)
        query = query + recog_result
        # 根据use_rag参数决定是否使用RAG模型
        if use_rag:
            rag = self.rag_generator(query)
        else:
            rag = ""

        # 获取工具渲染后的状态
        rendered_tools = self.tool_render.rendered_tools

        # 构建prompt输入字典
        prompt_input_dict = {"input": query, "rag": rag, "rendered_tools": rendered_tools}

        # 通过prompt_generator生成prompt输出字典
        prompt_output_dict = self.prompt_generator(prompt_input_dict)

        # 以指定级别打印输出
        self.level_print(prompt_output_dict, 3)

        # 进行聊天交互并获取输出
        chat_output = self.chat(system=prompt_output_dict["system"], user=prompt_output_dict["user"])

        # 解析聊天输出为JSON格式
        json_outputs = self.json_parser(chat_output)

        # 以指定级别打印LLM（Large Language Model）的输出
        self.level_print("LLM Output: " + str(json_outputs), 3)

        # 初始化索引和重试次数
        # 执行函数
        index = 0
        retry_times = 0
        chat_fail_flag = False

        # 主循环，处理聊天直到失败标记为True
        while 1:
            # 内循环，处理重试逻辑
            while 1:
                # 当重试次数超过限制时，跳出循环
                if retry_times >= self.args_base["retry_times"]:
                    self.level_print("Retry times exceed 5, stop retrying", 1)
                    chat_fail_flag = True
                    break

                # 如果json_outputs为空，重新发起聊天交互
                if json_outputs is None:
                    prompt_input_dict = {"input": query, "rag": rag, "rendered_tools": rendered_tools}
                    prompt_output_dict = self.prompt_generator(prompt_input_dict)
                    chat_output = self.chat(system=prompt_output_dict["system"], user=prompt_output_dict["user"])
                    json_outputs = self.json_parser(chat_output)
                    self.level_print("\nRetry.", 1)
                    retry_times += 1
                    continue

                # 如果当前索引超出json_outputs范围，标记失败并跳出循环
                if index >= len(json_outputs):
                    chat_fail_flag = True
                    break

                # 打印当前输出，并检查是否满足工具的评价条件
                self.level_print(json_outputs[index], 2)
                if self.json_parser.evaluate_jsontool(json_outputs[index], self.tools):
                    break
                else:
                    self.level_print("\nRetry.", 1)
                    self.level_print("LLM Output: " + str(json_outputs), 3)
                    prompt_input_dict = {"input": query, "rag": rag, "rendered_tools": rendered_tools}
                    prompt_output_dict = self.prompt_generator(prompt_input_dict)
                    chat_output = self.chat(system=prompt_output_dict["system"], user=prompt_output_dict["user"])
                    json_outputs = self.json_parser(chat_output)
                    retry_times += 1

            # 如果聊天失败标记为True，跳出主循环
            if chat_fail_flag:
                break

            # 执行json_outputs中的指令，并根据执行结果重新发起聊天
            output = self.exec(json_outputs[index])
            if output != "":
                query = query + output
                prompt_input_dict = {"input": query + "为以下内容进行补充：" + str(json_outputs), "rag": rag, "rendered_tools": rendered_tools}
                prompt_output_dict = self.prompt_generator(prompt_input_dict)
                chat_output = self.chat(system=prompt_output_dict["system"], user=prompt_output_dict["user"])
                json_outputs = self.json_parser(chat_output)

            # 更新索引，准备处理下一个指令
            index += 1


######################## 设置大模型工具 ########################
    def detect_object_location(self, object_name: str) -> str:
        """检测指定物体的位置。"""
        if "金属块" in object_name:
            detect_location = [3.23, 5.78, 9.0, 90.0, 0.0, -90.0]
        else:
            detect_location = [-1.23, 2.23, 9.0, 90.0, 0.0, -90.0]
        # output_str = "你已经完成了{0}的位置检测，{1}的位置为{2}。".format(object_name, object_name, detect_location)
        output_str = " {0} 的位置为 {1} 。".format(object_name, detect_location)
        print(output_str)
        return output_str

    def move_end(self, position: list) -> str:
        """移动机械臂末端夹爪到指定位姿。"""
        # output_str = "你已经完成了移动机械臂末端到位置 {0}。".format(str(position))
        # robot_tcp = RobotTcp('192.168.71.22', 8001)

        output_str = "你已经完成了移动机械臂末端到位置 {0}。".format(str(position))
        print(output_str)
        return ""

    def open_suck(self, object_name: str) -> str:
        """机械臂末端的吸盘开始真空吸取，必须得先将机械臂末端移动到指定目标大致位置才能吸到目标。"""
        output_str = "你已经完成了 {0} 的吸取。".format(object_name)
        print(output_str)
        return ""

    def move_to_origin(self) -> str:
        """机械臂末端移动到原点。"""
        output_str = "你已经完成了移动机械臂末端到原点。"
        print(output_str)
        return ""

    def open_the_door(self) -> str:
        """打开门。"""
        output_str = "你已经完成了打开门。"
        print(output_str)
        return ""

    def close_suck(self) -> str:
        """机械臂末端吸盘破真空释放吸取的物体。"""
        output_str = "你已经完成了破真空。"
        print(output_str)
        return ""

