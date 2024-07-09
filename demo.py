from robot_tcp import RobotTcp
from visual.visual_handle import VisualHandle
from robotAds import RobotAds
from act.inference import Inference
from visual.hkCamera import hkCamera
import torch
import json
import numpy as np
from math import *

class test:
    def __init__(self):
        ######################## 读取 json 文件 ########################
        filename = './config/config.json'
        # 使用 with 语句打开文件，确保最后文件会被正确关闭
        with open(filename, 'r', encoding='utf-8') as file:
            # 加载JSON数据
            args = json.load(file)
        ######################## 初始化 ########################
        self.visual_handle = VisualHandle(args=args)
        self.hk_camera = hkCamera()
        self.robot_tcp = RobotTcp(args["jkrobot"]["ip"], args["jkrobot"]["port"])
        self.robot_ads = RobotAds(args["jkrobot"]["ip"], args["jkrobot"]["adsport"])
        self.inference = Inference(ckpt_dir='act/weight/hk/random_crop_sim_od_spatial_distance6.0_q30')
        self.metal_num = 0
    def detect_object_location(self, object_name: str) -> list:
        """检测指定物体的位置。"""
        color_src, depth_src = self.visual_handle.get_picture(if_show=True)
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
            move_end_pos[1] += -0.003 + 0.073 * sin(angles[i] / 180 * pi) #
            move_end_pos[0] += 0.073 * cos(angles[i] / 180 * pi)
            # if i == self.metal_num:
            #     self.metal_num += 1
            print(f"move_end_pos: {move_end_pos}")
            return move_end_pos.tolist()
        return None

        # # output_str = "你已经完成了{0}的位置检测，{1}的位置为{2}。".format(object_name, object_name, detect_location)
        # output_str = " {0} 的位置为 {1} 。".format(object_name, detect_location)
        # print(output_str)
        # return output_str

    def move_end(self, position: list) -> str:
        """移动机械臂末端夹爪到指定位姿。"""
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
        self.robot_tcp.end_absmovecircle_jkrobot(p_len, radius, p_transition.tolist(), p_end.tolist(), 0.5, 5.0, 5.0)
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
                # action = self.inference(qpos, hk_image, time_prefcounter=None)
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

import cv2
import time
import threading

test = test()

# 开新线程奥比中光录屏
import threading

def my_function(test):
    """示例函数，将在新线程中运行"""
    # location = test.detect_object_location("金属块2号")
    print("start")
    time.sleep(2)
    res_pos = np.array(test.robot_tcp.get_end_pos_jkrobot()) + np.array([0, 0, -0.05, 0, 0, 0])
    test.robot_tcp.end_absmove_jkrobot(res_pos.tolist())
    # location = test.detect_object_location("金属块2号")
    # test.move_end(location)


# 创建一个新线程
new_thread = threading.Thread(target=my_function, args=(test,))

# 启动新线程
# new_thread.start()



# test.open_the_door()


# """示例函数，将在新线程中运行"""
# # 海康相机录屏
# frame_width = 1280
# frame_height = 720
# fps = 25.0  # 帧率
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
#
# # 创建VideoWriter对象
# ob_image_video = cv2.VideoWriter('ob_image.mp4', fourcc, fps, (frame_width, frame_height))
#
# # 假设你的图像文件是以数字顺序命名的，例如：frame1.jpg, frame2.jpg, ...
# print("开始录制")
# last_image = np.random.randint(0, 255, (frame_height, frame_width, 3))
#
# for _ in range(300):  # 假设有100帧图像
#     last_time = time.perf_counter()
#     while True:
#         if time.perf_counter() - last_time >= 0.025:
#             break
#     last_time = time.perf_counter()
#     ob_image, _ = test.visual_handle.get_picture()
#     if np.array_equal(last_image, ob_image):
#         continue
#     else:
#         ob_image_video.write(cv2.cvtColor(ob_image, cv2.COLOR_RGB2BGR))
#         last_image = ob_image
#
#
#
# # 释放VideoWriter对象
# ob_image_video.release()



# location = test.detect_object_location("金属块2号")
# test.move_end(location)
# time.sleep(0.5)
# test.open_suck("金属块2号")
# test.move_to_origin()
test.open_the_door()
# location = test.detect_object_location("底座2号")
# test.move_end(location)
#
# test.close_suck()
# test.move_to_origin()

# location = test.detect_object_location("底座2号")
# test.move_end(location)
#
# test.close_suck()
# test.move_to_origin()
# new_thread.join()
test.hk_camera.__del__()





# for i in range(3):
#     location = test.detect_object_location("金属块1号")
#     test.move_end(location)
#
#     test.robot_tcp.end_absmove_jkrobot([-0.442576, 0.00507468, 0.38258, 9.0e+01, 0.0, -9.0e+01])
