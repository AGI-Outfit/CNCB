import json
import numpy as np
import math
from math import *

class Pixel2Base:
    def __init__(self, inner_path: str, outer_path: str, log_level: int = 3):
        # 日志等级
        self._log_level = log_level
        # 获取内参和畸变系数
        with (open(inner_path, 'r', encoding='utf-8') as json_file):
            data = json.load(json_file)
            self._mtx = np.array(data["mtx"])
            self._dist = np.array(data["dist"])
        self._level_print("mtx: \n{0}".format(self._mtx), 2)
        self._level_print("dist: \n{0}".format(self._dist), 2)
        self._level_print("Get inner Calibration Param.", 1)
        # 获取外参
        with (open(outer_path, 'r', encoding='utf-8') as json_file):
            data = json.load(json_file)
            R_cam2end, T_cam2end = np.array(data["R"]), np.array(data["T"])
            self._RT_cam_to_end = np.column_stack((R_cam2end, T_cam2end))
            self._RT_cam_to_end = np.row_stack((self._RT_cam_to_end, np.array([0, 0, 0, 1])))
        self._level_print("cam_to_end: \n{0}".format(self._RT_cam_to_end), 1)
        self._level_print("Get outer Calibration Param.", 1)

    def _level_print(self, input_str, level):
        if level <= self._log_level:
            print(input_str)

    def myRPY2R_robot(self, x: float, y: float, z: float) -> np.ndarray[np.float64]:
        """
        将欧拉角（roll-pitch-yaw）转换为旋转矩阵。

        :param x: roll角，绕x轴的旋转角度，单位为弧度。
        :param y: pitch角，绕y轴的旋转角度，单位为弧度。
        :param z: yaw角，绕z轴的旋转角度，单位为弧度。

        :return: 旋转矩阵，表示从世界坐标系到机器人坐标系的转换
        """
        # 计算绕x轴的旋转矩阵
        Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        # 计算绕y轴的旋转矩阵
        Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        # 计算绕z轴的旋转矩阵
        Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
        # 通过矩阵乘法得到最终的旋转矩阵，顺序为z轴、y轴、x轴
        R = Rz @ Ry @ Rx
        return R

    # 用于根据位姿计算变换矩阵
    def pose_robot(self, Tx: float, Ty: float, Tz: float, Rx: float, Ry: float, Rz: float) -> np.ndarray[np.float64]:
        """
        计算并返回机器人的位姿矩阵。

        :param Rx: 控制机器人的旋转角度，单位为度。
        :param Ry: 控制机器人的旋转角度，单位为度。
        :param Rz: 控制机器人的旋转角度，单位为度。
        :param Tx: 控制机器人的平移量，单位为m。
        :param Ty: 控制机器人的平移量，单位为m。
        :param Tz: 控制机器人的平移量，单位为m。

        :return: RT1: 4x4 的齐次变换矩阵，用于描述机器人在世界坐标系中的位置和姿态。
        """
        # 将角度从度转换为弧度
        thetaX = Rx / 180 * pi
        thetaY = Ry / 180 * pi
        thetaZ = Rz / 180 * pi

        # 根据RPY角计算旋转矩阵
        R = self.myRPY2R_robot(thetaX, thetaY, thetaZ)

        # 定义平移向量
        t = np.array([[Tx * 1000], [Ty * 1000], [Tz * 1000]])

        # 将旋转矩阵和平移向量合并为变换矩阵
        RT1 = np.column_stack([R, t])  # 列合并

        # 添加齐次坐标项，完成4x4变换矩阵的构建
        RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))

        return RT1

    def pixel2cam(self, pos2pixel: np.ndarray, depth: np.ndarray) -> np.ndarray:
        # pos2cam = self._mtx
        # u, v  size(1, points_num)
        u, v = pos2pixel[1, :].reshape(1, -1), pos2pixel[0, :].reshape(1, -1)
        fx, fy, cx, cy = self._mtx[0, 0], self._mtx[1, 1], self._mtx[0, 2], self._mtx[1, 2]

        # 步骤1：从像素坐标转换到归一化坐标
        x_n = (u - cx) / fx
        y_n = (v - cy) / fy

        # depth = depth.reshape(3, 1)
        # 步骤2：从归一化坐标转换到相机坐标系（假设Z=depth）
        Z_c = depth * 1000
        X_c = x_n * Z_c
        Y_c = y_n * Z_c

        return np.row_stack((X_c, Y_c, Z_c))

    def cal_pos2pixel(self, end2base: np.ndarray, pos2pixel: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
            将像素位置转换为基座坐标系中的位置。

            参数:
            :param end2base: np.ndarray，表示机器人末端点到基座的位姿[x, y, z, rx, ry, rz] (m, deg)。
            :param pos2pixel: np.ndarray，表示物体在像素坐标系中的位置[v, u] size(2, n)。
            :param depth: np.ndarray，表示物体的距离，用于从像素坐标到相机坐标的转换(m), size(1, n)。

            返回值:
            :return: np.ndarray，表示物体在基座坐标系中的位置[x, y, z] (m)。
            """
        RT_end_to_base = self.pose_robot(*end2base)
        pos2cam = self.pixel2cam(pos2pixel, depth)
        # pos2cam = np.expand_dims(pos2cam, axis=-1)
        self._level_print("pos2cam: \n{0}".format(pos2cam), 5)
        points_num = pos2pixel.shape[1]
        pos2end = self._RT_cam_to_end @ np.row_stack((pos2cam, np.ones((1, points_num))))
        self._level_print("pos2end: \n{0}".format(pos2end[:3]), 5)
        pos2base = (RT_end_to_base @ pos2end) / 1000

        return pos2base[:3, :]



