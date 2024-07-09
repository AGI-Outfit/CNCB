from function_calling.base_function.llm_json_tool import LLMJsonTool
from function_calling.base_function.speech_recog import SpeechRecognizer
from robot_tcp import RobotTcp
from visual.visual_handle import VisualHandle
from robotAds import RobotAds
from act.inference import Inference
from visual.hkCamera import hkCamera
import torch
import json
import numpy as np
from math import *
import time
import cv2
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

        self.visual_handle = VisualHandle(args=args)
        self.hk_camera = hkCamera()
        self.robot_tcp = RobotTcp(args["jkrobot"]["ip"], args["jkrobot"]["port"])
        self.robot_ads = RobotAds(args["jkrobot"]["ip"], args["jkrobot"]["adsport"])
        self.inference = Inference(ckpt_dir='act/weight/hk/random_crop_sim_od_spatial_distance6.0_q30')
        self.metal_num = 0

    # 就算config.json中设置了use_rag，这里如果是False也不启动rag
    def __call__(self, query, use_rag=True):
        """
        使对象实例可被调用，处理查询并生成相应的输出。

        :param query: 用户的查询字符串
        :param use_rag: 是否使用RAG（Region of Interest Attention Graph）模型，默认为True
        """
        # 语音识别获得Prompt
        # record_path = self.speech_recognizer.record()
        # self.level_print("请输入语音...", 1)
        # time.sleep(3)

        self.level_print("正在加载音频...", 1)
        record_path = "output/speechrecog/20240620_171432.m4a"
        recog_result = ""
        self.level_print("正在识别语音...", 1)
        if record_path is not None:
            recog_result = self.speech_recognizer.speech_recog(record_path)
            recog_result = recog_result.replace("一", "1")
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

        import pickle
        with open('prompt_output_dict.pkl', 'wb') as f:
            pickle.dump(prompt_output_dict, f)

        # 以指定级别打印输出
        self.level_print(prompt_output_dict, 3)

        json_outputs = None
        while json_outputs is None:
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
        command_len = len(json_outputs)
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
                    self.level_print('chat fail! 前索引超出json_outputs范围', 1)
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
                self.level_print('chat fail!', 1)
                break

            # 执行json_outputs中的指令，并根据执行结果重新发起聊天
            run_command = json_outputs[index]
            self.level_print(f'run command: {run_command}', 1)
            output = self.exec(run_command)
            if output != "":
                if "位置" in output:
                    pos = output.split('[')[1].split(']')[0].split(',')
                    pos = [float(i) for i in pos]
                    json_outputs[index+1]["arguments"]["position"] = pos

                # query = query + output
                # prompt_input_dict = {"input": query + "为以下内容进行补充：" + str(json_outputs), "rag": rag,
                #                      "rendered_tools": rendered_tools}
                # prompt_output_dict = self.prompt_generator(prompt_input_dict)
                # chat_output = self.chat(system=prompt_output_dict["system"], user=prompt_output_dict["user"])
                # json_outputs = self.json_parser(chat_output)

            # 更新索引，准备处理下一个指令
            index += 1


    def detect_object_location(self, object_name: str) -> str:
        """检测指定物体的位置。"""
        color_src, depth_src = self.visual_handle.get_picture(if_show=False)
        fragments_points, angles = self.visual_handle.instance_fragment(color_src, object_name)
        list_pos2pixels = []
        list_depths = []
        for fragment_points in fragments_points:
            # CNC下料专用版本 滤波+平均深度
            depths, _ = self.visual_handle.get_filtered_aligned_depth(np.array(fragment_points), depth_src,
                                                                      if_filtered=True)
            depths = np.tile(np.mean(depths), (len(fragment_points)))
            list_pos2pixels.append(fragment_points)
            list_depths.append(depths / 1000)

        objects_num = len(list_depths)
        # 坐标变换获得抓取点
        end2base = np.array(self.robot_ads.getEE())  # 获取JK机械臂末端位姿
        print(f"end2base: {end2base}")
        for i in range(objects_num):
            # pos2pixel  size(2, points_num_filtered) (v, u)
            # depth      size(1, points_num_filtered)
            pos2pixels = np.array(list_pos2pixels[i])
            pos2pixels = pos2pixels.transpose(1, 0)
            depths = np.array(list_depths[i]).reshape(1, -1)

            # end2base_after_angle = end2base + np.array([0, 0, 0, 0, 0, angles[i]])
            pos2base = self.visual_handle.pixel2base.cal_pos2pixel(end2base=end2base, pos2pixel=pos2pixels,
                                                                   depth=depths)

            pos2base = np.mean(pos2base, axis=1)
            # 机械臂移动
            move_end_pos = np.concatenate((pos2base, end2base[3:6]), axis=0)
            move_end_pos[5] += angles[i]
            # 末端和吸盘机械安装位置中心点Z偏移0.08
            move_end_pos[2] = move_end_pos[2] + 0.087
            # 末端和吸盘机械安装位置中心点间隔为0.73，Y方向偏移0.003，X正对
            move_end_pos[1] += -0.003 + 0.073 * sin(angles[i] / 180 * pi)  #
            move_end_pos[0] += 0.073 * cos(angles[i] / 180 * pi)
            # if i == self.metal_num:
            #     self.metal_num += 1
            # print(f"move_end_pos: {move_end_pos}")
            output_str = " {0} 的位置为 {1} 。".format(object_name, move_end_pos.tolist())
            return output_str
        return None

        # # output_str = "你已经完成了{0}的位置检测，{1}的位置为{2}。".format(object_name, object_name, detect_location)
        # output_str = " {0} 的位置为 {1} 。".format(object_name, detect_location)
        # print(output_str)
        # return output_str

    def move_end(self, position: list) -> str:
        """移动机械臂末端夹爪到指定位姿。"""
        position = [float(p) for p in position]
        p_end = position
        p_start = np.array(self.robot_tcp.get_end_pos_jkrobot())
        p_end = np.array(p_end)
        delta_r = p_start[0:2] - p_end[0:2]
        delta_z = p_start[2] - p_end[2]

        if delta_z >= 0:
            p_true_end = p_end.copy()
            p_end += np.array([0, 0, 0.05, 0, 0, 0])
            p_mid = np.concatenate((p_end[0:2], p_start[2:3], p_end[3:6]), axis=0)
        else:
            p_true_end = p_end.copy()
            p_mid = np.concatenate((p_start[0:2], p_end[2:3], p_end[3:6]), axis=0)
        radius = 0.75 * min(np.abs(delta_z), np.sqrt(delta_r.dot(delta_r)))

        p_transition = np.concatenate((p_start, p_mid), axis=0)
        p_transition = np.round(p_transition, 4)
        p_len = p_transition.shape[0] / 6
        self.robot_tcp.end_absmovecircle_jkrobot(p_len, radius, p_transition.tolist(), p_end.tolist(), 0.2, 1.0,
                                                 1.0)
        self.robot_tcp.end_absmove_jkrobot(p_true_end.tolist())
        output_str = "你已经完成了移动机械臂末端到位置 {0}。".format(str(position))
        print(output_str)
        return ""

    def open_suck(self, object_name: str) -> str:
        """机械臂末端的吸盘开始真空吸取，必须得先将机械臂末端移动到指定目标大致位置才能吸到目标。"""
        self.robot_tcp.suck_start([2])
        output_str = "你已经完成了 {0} 的吸取。".format(object_name)
        print(output_str)
        return ""

    def move_to_origin(self) -> str:
        """机械臂末端移动到原点。"""
        p_end = [-0.442576, 0.00507468, 0.38258, 9.0e+01, 0.0, -9.0e+01]
        p_start = np.array(self.robot_tcp.get_end_pos_jkrobot())
        p_end = np.array(p_end)
        delta_r = p_start[0:2] - p_end[0:2]
        delta_z = p_start[2] - p_end[2]

        if delta_z >= 0:
            p_true_end = p_end.copy()
            p_end = p_start + np.array([0, 0, 0.10, 0, 0, 0])
            self.robot_tcp.end_absmove_jkrobot(p_end.tolist())
        else:
            self.robot_tcp.end_absmove_jkrobot(p_end.tolist())
            return ""

        self.robot_tcp.end_absmove_jkrobot(p_true_end.tolist())
        # output_str = "你已经完成了移动机械臂末端到原点。"
        # print(output_str)
        return ""

    def open_the_door(self) -> str:
        """打开门。"""
        """打开门。"""
        frame_width = 1440
        frame_height = 1080
        fps = 25.0  # 帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器

        # 创建VideoWriter对象
        hk_image_video = cv2.VideoWriter('hk_image_act.mp4', fourcc, fps, (frame_width, frame_height))

        # 假设你的图像文件是以数字顺序命名的，例如：frame1.jpg, frame2.jpg, ...
        print("开始录制")
        last_image = np.random.randint(0, 255, (frame_height, frame_width, 3))

        zero_time = 0
        while True:
            # start_time = time.time()
            hk_image = self.hk_camera.getColorImage()
            now_image_full = self.hk_camera.getColorImageFullSize()
            if np.array_equal(last_image, now_image_full):
                pass
            else:
                hk_image_video.write(cv2.cvtColor(now_image_full, cv2.COLOR_RGB2BGR))
                last_image = now_image_full

            # ob_image, _ = self.visual_handle.get_picture(if_show=False, if_save=False)
            qpos = self.robot_ads.getEE()
            # vel = self.robot_tcp.get_end_vel_jkrobot()
            # print(vel)
            # print(f"收集信息时长： {time.time() - start_time}")

            evaluate_time = time.time()
            with torch.inference_mode():
                action = self.inference(qpos, hk_image, time_prefcounter=time.perf_counter())
                # action = self.inference(qpos, hk_image, ob_image)
            # print(f"推理时长： {time.time() - evaluate_time}")
            # print(f"qpos: {qpos[0:3]}")
            # print(f"action: {action[0:3]}")
            # print(f"bias: {(qpos - action)[0:3]}")

            if np.linalg.norm((qpos - action)[0:3]) < 0.01:
                zero_time += 1
                if zero_time >= 25:
                    break
            else:
                zero_time = 0

            # tcp_time = time.time()
            self.robot_tcp.end_absmoveimmediate_jkrobot(action.tolist())
            # print(f"TCP通讯时长： {time.time() - evaluate_time}")
            # print(f"总时长： {time.time() - start_time}")
            # time.sleep(0.005)
            # output_str = "你已经完成了打开门。"
            # print(output_str)
        hk_image_video.release()
        return ""

    def close_suck(self) -> str:
        """机械臂末端吸盘破真空释放吸取的物体。"""
        self.robot_tcp.suck_stop(suck_io_id=[2], unsuck_io_id=[3])
        output_str = "你已经完成了破真空。"
        print(output_str)
        return ""
