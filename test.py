# from http import HTTPStatus
# import dashscope
#
#
# def simple_multimodal_conversation_call():
#     """Simple single round multimodal conversation call.
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 # {"image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"},
#                 {"text": "你是谁"}
#             ]
#         }
#     ]
#     response = dashscope.MultiModalConversation.call(model='qwen-vl-plus',
#                                                      messages=messages,
#                                                      api_key="sk-d374d3c1b7d946c2ba5b03d91c520aa4")
#     # The response status_code is HTTPStatus.OK indicate success,
#     # otherwise indicate request is failed, you can get error code
#     # and message from code and message.
#     if response.status_code == HTTPStatus.OK:
#         print(response)
#     else:
#         print(response.code)  # The error code.
#         print(response.message)  # The error message.
#
#
# if __name__ == '__main__':
#     simple_multimodal_conversation_call()
import time

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage
#
# model = ChatOpenAI(api_key="sk-d374d3c1b7d946c2ba5b03d91c520aa4",
#                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#                  model="qwen-vl-max")
#
# message = HumanMessage(
#     content=[
#         {"type": "text", "text": "are these two images the same?"},
#         {"type": "image_url", "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
#     ],
# )
# response = model.invoke([message])
# print(response.content)


# from langchain_core.tools import tool
# from langchain_core.tools import render_text_description
# from typing import List
# from typing import Any, Dict, Optional, TypedDict
# import inspect
#
#
# class ToolCallRequest(TypedDict):
#     """A typed dict that shows the inputs into the invoke_tool function."""
#
#     name: str
#     arguments: Dict[str, Any]
#
# class ToolRender:
#     def __init__(self, tools: List[classmethod]):
#         self.tools = tools
#         self.tools_define_strs = []
#         self.tools_docs_strs = []
#         for tool in self.tools:
#             source_lines, _ = inspect.getsourcelines(tool)
#             self.tools_define_strs.append(source_lines[0].strip())
#             self.tools_docs_strs.append(tool.__doc__)
#
#     def get_tool_render(self):
#         """
#         获取工具渲染字符串信息
#         :return:
#         """
#         output_str = ""
#         for i in range(len(self.tools_docs_strs)):
#             define_str = self.tools_define_strs[i][4:-1]
#             doc = self.tools_docs_strs[i]
#             output_str += "\n" + define_str + " - " + doc
#         output_str += "\n"
#         return output_str


#
# class CNCBlank():
#     def __init__(self):
#         self.tools = [
#             self.detect_object_location,
#             self.move_end_to_object,
#             self.grasp_object,
#             self.move_to_cnc,
#             self.open_grasp]
#
#         funcs = [getattr(self, tool.__code__.co_name) for tool in self.tools]
#         self.tool_render = ToolRender(funcs)
#         print(self.tool_render.get_tool_render())

        # funcs = []
        # for tool in self.tools:
        #     funcs.append(getattr(self, tool.__code__.co_name))
        # func = funcs[0]
        # print(type(func))
        # # 获取注释信息
        # print(func.__doc__)
        # source_lines, _ = inspect.getsourcelines(func)
        #
        # # 移除空格和制表符，获取实际的第一行代码
        # # 注意：这里的第一行通常是函数定义行，可能包含缩进
        # first_line = source_lines[0].strip()
        #
        # print(first_line)
        # # 获取参数信息
        # print(func.__annotations__)
        # print(func.__annotations__["object_name"].__name__)


# from transformers import AutoModel, AutoTokenizer
# import torch
# import faiss
# import numpy as np
#
# # 1. 加载预训练模型和分词器
# model_name = "shibing624/text2vec-base-chinese"
# tokenizer = AutoTokenizer.from_pretrained("/home/jk/PycharmProjects/xinference/cnc_baseline/config/text2vec-base-chinese")
# # tokenizer.save_pretrained("/home/jk/")
# model = AutoModel.from_pretrained("/home/jk/PycharmProjects/xinference/cnc_baseline/config/text2vec-base-chinese")
# # model.save_pretrained("/home/jk/")
# # 2. 假设我们有一组文本需要转换为向量
# texts = ["CNC工具切换，将当前工具切换到指定工具，一般一个完整的CNC的工具切换过程：先放下当前工具，然后切换成指定工具。", "CNC下料，将指定物体精准放置到指定位置，一般一个完整的CNC下料过程：先检测目标位置，然后移动机械臂末端到目标处，然后抓取目标，然后移动机械臂末端到CNC下料处，最后张开夹爪以释放目标。"]
# tokenized_texts = [tokenizer.encode(text, return_tensors='pt') for text in texts]
#
# # 3. 使用模型获取文本的embedding
# with torch.no_grad():
#     embeddings = [model(text)[0].mean(dim=1).numpy() for text in tokenized_texts]
#
# # 4. 将嵌入向量转换为FAISS所需的格式，并构建索引
# embeddings = np.squeeze(embeddings)
# d = embeddings[0].shape[0]  # 维度
# index = faiss.IndexFlatL2(d)  # 使用L2距离作为度量标准，创建索引
# # IndexIVFFlat
# # 转换为FAISS接受的float32数组格式
# emb_array = np.array(embeddings, dtype='float32')
#
# # 添加向量到索引中
# index.add(emb_array)
#
# # 现在你可以使用这个索引来搜索最相似的向量了
# query_text = "如何切换CNC工具？可以自己假设函数"
# query_tokenized = tokenizer.encode(query_text, return_tensors='pt')
# with torch.no_grad():
#     query_emb = model(query_tokenized)[0].mean(dim=1).numpy()
# query_emb = np.squeeze(query_emb)
# # 转换查询向量格式
# query_emb = np.array([query_emb], dtype='float32')
#
# # 搜索最相似的k个向量
# k = 2
# distances, indices = index.search(query_emb, k)
#
# print(f"Query: {query_text}")
# for i in range(k):
#     print(f"Similar sentence #{i+1} with distance {distances[0][i]}:")
#     print(texts[indices[0][i]])


import json
# from function_calling.cnc_blank import CNCBlank
# from function_calling.base_function.speech_recog import SpeechRecognizer
from function_calling.cnc_blank_real import CNCBlank
if __name__ == '__main__':
# ######################## 读取 json 文件 ########################
    filename = './config/config.json'
    # 使用 with 语句打开文件，确保最后文件会被正确关闭
    with open(filename, 'r', encoding='utf-8') as file:
        # 加载JSON数据
        args = json.load(file)
# ######################## 初始化 ########################
    # s = SpeechRecognizer(args)
    llm = CNCBlank(args=args)
    # llm("")
    llm.open_the_door()
    llm.hk_camera.__del__()


# from robot_tcp import RobotTcp
# from visual.visual_handle import VisualHandle
# from robotAds import RobotAds
# from act.inference import Inference
# from visual.hkCamera import hkCamera
# import torch
# import json
# import numpy as np
# from math import *
#
# class test:
#     def __init__(self):
#         ######################## 读取 json 文件 ########################
#         filename = './config/config.json'
#         # 使用 with 语句打开文件，确保最后文件会被正确关闭
#         with open(filename, 'r', encoding='utf-8') as file:
#             # 加载JSON数据
#             args = json.load(file)
#         ######################## 初始化 ########################
#         self.visual_handle = VisualHandle(args=args)
#         self.hk_camera = hkCamera()
#         self.robot_tcp = RobotTcp(args["jkrobot"]["ip"], args["jkrobot"]["port"])
#         self.robot_ads = RobotAds(args["jkrobot"]["ip"], args["jkrobot"]["adsport"])
#         self.inference = Inference(ckpt_dir='act/weight/hk/random_crop_sim_od_spatial_distance6.0_q30')
#         self.metal_num = 0
#     def detect_object_location(self, object_name: str) -> list:
#         """检测指定物体的位置。"""
#         color_src, depth_src = self.visual_handle.get_picture(if_show=True)
#         fragments_points, angles = self.visual_handle.instance_fragment(color_src, object_name)
#         list_pos2pixels = []
#         list_depths = []
#         for fragment_points in fragments_points:
#             # CNC下料专用版本 滤波+平均深度
#             depths, _ = self.visual_handle.get_filtered_aligned_depth(np.array(fragment_points), depth_src,
#                                                                       if_filtered=True)
#             depths = np.tile(np.mean(depths), (len(fragment_points)))
#             list_pos2pixels.append(fragment_points)
#             list_depths.append(depths / 1000)
#
#         objects_num = len(list_depths)
#         # 坐标变换获得抓取点
#         end2base = np.array(self.robot_ads.getEE())  # 获取JK机械臂末端位姿
#         print(f"end2base: {end2base}")
#         for i in range(objects_num):
#             # pos2pixel  size(2, points_num_filtered) (v, u)
#             # depth      size(1, points_num_filtered)
#             pos2pixels = np.array(list_pos2pixels[i])
#             pos2pixels = pos2pixels.transpose(1, 0)
#             depths = np.array(list_depths[i]).reshape(1, -1)
#
#             # end2base_after_angle = end2base + np.array([0, 0, 0, 0, 0, angles[i]])
#             pos2base = self.visual_handle.pixel2base.cal_pos2pixel(end2base=end2base, pos2pixel=pos2pixels,
#                                                                    depth=depths)
#
#             pos2base = np.mean(pos2base, axis=1)
#             # 机械臂移动
#             move_end_pos = np.concatenate((pos2base, end2base[3:6]), axis=0)
#             move_end_pos[5] += angles[i]
#             # 末端和吸盘机械安装位置中心点Z偏移0.08
#             move_end_pos[2] = move_end_pos[2] + 0.087
#             # 末端和吸盘机械安装位置中心点间隔为0.73，Y方向偏移0.003，X正对
#             move_end_pos[1] += -0.003 + 0.073 * sin(angles[i] / 180 * pi) #
#             move_end_pos[0] += 0.073 * cos(angles[i] / 180 * pi)
#             # if i == self.metal_num:
#             #     self.metal_num += 1
#             print(f"move_end_pos: {move_end_pos}")
#             return move_end_pos.tolist()
#         return None
#
#         # # output_str = "你已经完成了{0}的位置检测，{1}的位置为{2}。".format(object_name, object_name, detect_location)
#         # output_str = " {0} 的位置为 {1} 。".format(object_name, detect_location)
#         # print(output_str)
#         # return output_str
#
#     def move_end(self, position: list) -> str:
#         """移动机械臂末端夹爪到指定位姿。"""
#         p_end = position
#         p_start = np.array(self.robot_tcp.get_end_pos_jkrobot())
#         p_end = np.array(p_end)
#         delta_r = p_start[0:2] - p_end[0:2]
#         delta_z = p_start[2] - p_end[2]
#
#         if delta_z >= 0:
#             p_true_end = p_end.copy()
#             p_end += np.array([0, 0, 0.05, 0, 0, 0])
#             p_mid = np.concatenate((p_end[0:2], p_start[2:3], p_end[3:6]), axis=0)
#         else:
#             p_true_end = p_end.copy()
#             p_mid = np.concatenate((p_start[0:2], p_end[2:3], p_end[3:6]), axis=0)
#         radius = 0.75 * min(np.abs(delta_z), np.sqrt(delta_r.dot(delta_r)))
#
#         p_transition = np.concatenate((p_start, p_mid), axis=0)
#         p_transition = np.round(p_transition, 4)
#         p_len = p_transition.shape[0] / 6
#         self.robot_tcp.end_absmovecircle_jkrobot(p_len, radius, p_transition.tolist(), p_end.tolist(), 0.5, 5.0, 5.0)
#         self.robot_tcp.end_absmove_jkrobot(p_true_end.tolist())
#         output_str = "你已经完成了移动机械臂末端到位置 {0}。".format(str(position))
#         print(output_str)
#         return ""
#
#     def open_suck(self, object_name: str) -> str:
#         """机械臂末端的吸盘开始真空吸取，必须得先将机械臂末端移动到指定目标大致位置才能吸到目标。"""
#         self.robot_tcp.suck_start([2])
#         output_str = "你已经完成了 {0} 的吸取。".format(object_name)
#         print(output_str)
#         return ""
#
#     def move_to_origin(self) -> str:
#         """机械臂末端移动到原点。"""
#         p_end = [-0.442576, 0.00507468, 0.38258, 9.0e+01, 0.0, -9.0e+01]
#         p_start = np.array(self.robot_tcp.get_end_pos_jkrobot())
#         p_end = np.array(p_end)
#         delta_r = p_start[0:2] - p_end[0:2]
#         delta_z = p_start[2] - p_end[2]
#
#         if delta_z >= 0:
#             p_true_end = p_end.copy()
#             p_end = p_start + np.array([0, 0, 0.10, 0, 0, 0])
#             self.robot_tcp.end_absmove_jkrobot(p_end.tolist())
#         else:
#             self.robot_tcp.end_absmove_jkrobot(p_end.tolist())
#             return ""
#
#         self.robot_tcp.end_absmove_jkrobot(p_true_end.tolist())
#         # output_str = "你已经完成了移动机械臂末端到原点。"
#         # print(output_str)
#         return ""
#
#     def open_the_door(self) -> str:
#         """打开门。"""
#         zero_time = 0
#         while True:
#             # start_time = time.time()
#             hk_image = self.hk_camera.getColorImage()
#             # ob_image, _ = self.visual_handle.get_picture(if_show=False, if_save=False)
#             qpos = self.robot_ads.getEE()
#             # vel = self.robot_tcp.get_end_vel_jkrobot()
#             # print(vel)
#             # print(f"收集信息时长： {time.time() - start_time}")
#
#             evaluate_time = time.time()
#             with torch.inference_mode():
#                 action = self.inference(qpos, hk_image, time_prefcounter=time.perf_counter())
#                 # action = self.inference(qpos, hk_image, ob_image)
#             # print(f"推理时长： {time.time() - evaluate_time}")
#             # print(f"qpos: {qpos[0:3]}")
#             # print(f"action: {action[0:3]}")
#             # print(f"bias: {(qpos - action)[0:3]}")
#
#             if np.linalg.norm((qpos - action)[0:3]) < 0.01:
#                 zero_time += 1
#                 if zero_time >= 25:
#                     break
#             else:
#                 zero_time = 0
#
#             # tcp_time = time.time()
#             self.robot_tcp.end_absmoveimmediate_jkrobot(action.tolist())
#             # print(f"TCP通讯时长： {time.time() - evaluate_time}")
#             # print(f"总时长： {time.time() - start_time}")
#             # time.sleep(0.005)
#             # output_str = "你已经完成了打开门。"
#             # print(output_str)
#         return ""
#
#     def close_suck(self) -> str:
#         """机械臂末端吸盘破真空释放吸取的物体。"""
#         self.robot_tcp.suck_stop(suck_io_id=[2], unsuck_io_id=[3])
#         output_str = "你已经完成了破真空。"
#         print(output_str)
#         return ""
#
# import threading
# import cv2
# test = test()
# def my_function(test):
#     """示例函数，将在新线程中运行"""
#     test.open_the_door()
#     # print(f"Hello, from thread {threading.current_thread().name}")
#
#
# # 创建一个新线程
# new_thread = threading.Thread(target=my_function, args=(test,))
#
# # 启动新线程
# new_thread.start()
#
#
#
# location = test.detect_object_location("金属块2号")
# test.move_end(location)
# time.sleep(0.5)
# test.open_suck("金属块2号")
# test.move_to_origin()
# test.open_the_door()
# location = test.detect_object_location("底座2号")
# # print(location)
# test.move_end(location)
#
# test.close_suck()
# test.move_to_origin()
#
#
#
#
#
#
# # 海康相机录屏
# frame_width = 1280
# frame_height = 720
# hk_frame_width = 1440
# hk_frame_height = 1080
#
# fps = 30.0  # 帧率
# hk_fps = 25.0  # 帧率
# hk_fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
# # 创建VideoWriter对象
# image_video = cv2.VideoWriter('act_image.mp4', fourcc, fps, (frame_width, frame_height))
# hk_image_video = cv2.VideoWriter('act_hk_image.mp4', hk_fourcc, hk_fps, (hk_frame_width, hk_frame_height))
#
# # 假设你的图像文件是以数字顺序命名的，例如：frame1.jpg, frame2.jpg, ...
# print("开始录制")
# last_hk_image = np.random.randint(0, 255, (hk_frame_height, hk_frame_width, 3))
# last_image = np.random.randint(0, 255, (frame_height, frame_width, 3))
# for _ in range(300):  # 假设有100帧图像
#     time.sleep(0.035)
#     hk_image = test.hk_camera.getColorImageFullSize()
#     image, _ = test.visual_handle.get_picture()
#
#     now_hk_iamge = hk_image
#     now_iamge = image
#     if np.array_equal(last_hk_image, now_hk_iamge):
#         pass
#     else:
#         hk_image_video.write(cv2.cvtColor(now_hk_iamge, cv2.COLOR_RGB2BGR))
#         last_hk_image = now_hk_iamge
#
#     if np.array_equal(last_image, now_iamge):
#         pass
#     else:
#         hk_image_video.write(cv2.cvtColor(now_iamge, cv2.COLOR_RGB2BGR))
#         last_image = now_iamge
#
# test.hk_camera.__del__()
# new_thread.join()
# for i in range(3):
#     location = test.detect_object_location("金属块1号")
#     test.move_end(location)
#
#     test.robot_tcp.end_absmove_jkrobot([-0.442576, 0.00507468, 0.38258, 9.0e+01, 0.0, -9.0e+01])


