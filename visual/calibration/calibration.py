import cv2
import numpy as np
import glob
from math import *
import os
import time
import random
import math
from enum import Enum
import string
import os
import datetime
import json
from typing import Union, Optional


class CalibType(Enum):
    CHESSBOARD = 1
    CIRCLEGRID = 2


class Calibration:
    def __init__(self, board_size: tuple, board_length: tuple,  calib_type: CalibType,
                 image_for_innercalib: str = "",
                 log_level: int = 3):
        """
        初始化相机标定对象。注意，该类不适用于不规则圆网络。

        :param board_size: 棋盘格模板的规格尺寸，格式为(w, h)。
        :param board_length: 棋盘格模板的X、Y每个格子或者圆点的间距，单位mm，格式为(x, y)。
        :param image_for_innercalib: 用于校准的图像文件路径, 格式为"path+format", exp: "/home/jk/+.bmp"。
        :param calib_type: 校准类型，来自CalibType枚举。
        """
        # 获取棋盘格模板的规格
        self._board_size = board_size
        w = board_size[0]
        h = board_size[1]
        self._board_length = board_length
        # 读取用于内参标定的图片
        if image_for_innercalib != "":
            self._image_path, self._image_format = image_for_innercalib.split("+")
            self._images = glob.glob(self._image_path + "*" + self._image_format)
        else:
            self._images = []
            self._image_path = ""
            self._image_format = ""
        # 校准类型 棋盘格或圆网格
        self._calib_type = calib_type
        # 找角点必备参数
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        self._objp = np.zeros((w * h, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        self._objp[:, 0] = self._objp[:, 0] * self._board_length[0]
        self._objp[:, 1] = self._objp[:, 1] * self._board_length[1]
        # 储存棋盘格角点的世界坐标和图像坐标对
        self._objpoints = []  # 在世界坐标系中的三维点
        self._imgpoints = []  # 在图像平面的二维点
        self._rvecs = None
        self._tvecs = None
        # 待标定的内参
        self._mtx = None
        self._dist = None
        # 日志等级
        self._log_level = log_level
        # 输出路径
        self._output_path = "./output/calib_output/"
        self._measure_path = "measure/"
        self._image_for_outercalib_path = "image_for_outercalib/"
        self._undistort_path = "undistort/"
        self._end2base = "end2base/"

    def _level_print(self, input_str, level):
        if level <= self._log_level:
            print(input_str)

    def find_corners(self, img, if_draw: bool = True, if_check: bool = True) -> Optional[np.ndarray]:
        """
            寻找图像中的角点。

            :param img: 输入图像，要求为BGR格式。
            :param if_draw: 是否显示角点检测过程。
            :param if_check: 是否显示过程等待按键。

            :return: 返回找到的角点的坐标，如果未找到则返回None。
        """
        # 初始化是否显示停顿
        key = int(1)
        if if_check:
            key = int(0)
        # 将输入图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self._calib_type == CalibType.CHESSBOARD:
            ret, corners = cv2.findChessboardCorners(gray, self._board_size, None)
        elif self._calib_type == CalibType.CIRCLEGRID:
            ret, corners = cv2.findCirclesGrid(gray, self._board_size)
        else:
            self._level_print("Error: Wrong calib type.", 1)
            return None
        # 如果找到了圆点，对结果进行进一步精确定位
        if ret:
            cv2.cornerSubPix(gray, corners, self._board_size, (-1, -1), self._criteria)
            # self._objpoints.append(self._objp)
            # self._imgpoints.append(corners)

            if if_draw:
                cv2.drawChessboardCorners(img, self._board_size, corners, ret)
                cv2.imshow('findCorners', img)
                cv2.waitKey(key)
            return corners  # 返回找到的角点坐标
        else:
            self._level_print("Fail: Can't find enough corners. Require {0}; Find {1}.".format(
                self._board_size[0] * self._board_size[1], corners.shape[0]), 1)
            cv2.imshow('NotFindCorners', img)
            cv2.waitKey(key)
            cv2.destroyWindow('NotFindCorners')
            return None  # 未找到角点时返回None

    def inner_calib(self, if_draw: bool = True, if_check: bool = True, if_write_json: bool = False) -> (
    np.ndarray[np.float64], np.ndarray[np.float64]):
        """
            内部调用的标定函数，用于标定摄像头的内参和进行图像校正。

            :param if_draw: 是否在找到的角点上绘制标记，默认为True。
            :param if_check: 是否显示过程等待按键，默认为True。
            :param if_write_json: 是否将结果写入JSON文件，默认为False。
            :return: (mtx, dist)返回校正后的相机矩阵mtx和畸变系数dist。
        """
        # 清理内存
        self._objpoints.clear()
        self._imgpoints.clear()
        for image in self._images:
            src = cv2.imread(image)
            corners = self.find_corners(src, if_draw, if_check)
            if corners is not None:
                self._objpoints.append(self._objp)
                self._imgpoints.append(corners)
        # 标定摄像头内参
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self._objpoints, self._imgpoints, src.shape[1::-1], None, None)
        self._mtx = mtx
        self._dist = dist
        self._rvecs = rvecs
        self._tvecs = tvecs
        self._level_print("mtx: ", 2)
        self._level_print(self._mtx, 2)
        self._level_print("dist: ", 2)
        self._level_print(self._dist, 2)
        self._level_print("Inner Calibration finished.", 1)

        if if_write_json:
            # 假设你有一个字典变量
            data_dict = {
                "mtx": mtx.tolist(),
                "dist": dist.tolist(),
                "rvecs": [rvec.tolist() for rvec in rvecs],
                "tvecs": [tvec.tolist() for tvec in tvecs],
                "objpoints": [objpoint.tolist() for objpoint in self._objpoints],
                "imgpoints": [imgpoint.tolist() for imgpoint in self._imgpoints]
            }
            # JSON文件的输出路径
            directory = self._output_path + self._measure_path
            if not os.path.exists(directory):
                os.makedirs(directory)
            current_time = datetime.datetime.now()
            save_path = directory + "mtx-dist" + str(current_time.strftime("%Y-%m-%d %H:%M:%S")) + ".json"
            # 使用json.dump()方法将字典写入JSON文件
            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(data_dict, json_file, indent=4, ensure_ascii=False)

            self._level_print("mtx-dist.json saved.", 1)
        return self._mtx, self._dist

    def undistort(self, image: Optional[Union[np.ndarray, None]], img_path: str = "", if_check: bool = True,
                  if_write_image: bool = False, json_path: str = "") -> np.ndarray[np.float64]:
        """
           对图像进行去畸变处理。

           :param img_path: str - 需要进行去畸变处理的图像路径。
           :param if_check: bool = True - 是否进行畸变校验，默认为True。
           :param if_write_image: bool = False - 是否将去畸变后的图像写入文件，默认为False。
           :param json_path: str = "" = 选择读取json文件的内参和畸变参数

           :return: 去畸变后的图像
           """
        # 检查相机矩阵是否已设置，若未设置则先进行内部校准
        if json_path != "":
            self._read_mtx_dist_from_json(json_path)
        if self._mtx is None:
            self._level_print("Warning: Please run inner_calib() first. Auto run.", 1)
            self.inner_calib()

        # 读取图像，并计算新的相机矩阵以进行去畸变
        if image is not None:
            src = cv2.imread(img_path)
        else:
            src = image
        newcameramtx, roi = (
            cv2.getOptimalNewCameraMatrix(self._mtx,
                                          self._dist,
                                          src.shape[1::-1],
                                          0, src.shape[1::-1]))  # 计算新的相机矩阵

        self._level_print("Undistort the image: " + img_path, 3)
        dst = cv2.undistort(src, self._mtx, self._dist, None, newcameramtx)  # 去畸变处理
        self._level_print("Undistort finished.", 3)

        # 显示去畸变前后的图像，并等待按键
        cv2.imshow("dist", dst)
        key = int(1)
        if if_check:
            key = int(0)
        cv2.waitKey(key)
        cv2.destroyWindow("dist")

        # 如果指定，则将去畸变后的图像以及原始图像保存到文件
        if if_write_image:
            directory = self._output_path + self._undistort_path
            if not os.path.exists(directory):
                os.makedirs(directory)
            current_time = datetime.datetime.now()
            save_path = directory + "undistort-" + str(current_time.strftime("%Y-%m-%d %H:%M:%S")) + ".jpg"
            cv2.imwrite(save_path, dst)  # 保存去畸变后的图像
            self._level_print("Undistort image saved to: " + save_path, 1)
            save_path = directory + "distort-" + str(current_time.strftime("%Y-%m-%d %H:%M:%S")) + ".jpg"
            cv2.imwrite(save_path, src)  # 保存原始图像

        return dst

    def evaluate_mtx(self, json_path: str = ""):
        """
           评估相机矩阵的校准质量。
           若提供了json_path，则从指定的json文件中读取相机参数；若未提供，则检查是否已经进行了内部校准。
           接着，计算反投影误差来评估校准的质量。0.1左右是很不错的，此时每个参数差异都在5以内。

           :param json_path: str, 相机参数json文件的路径。如果为空字符串，则表示不从文件读取参数。

           :return: 无
       """
        # 检查相机矩阵是否已设置，若未设置则先进行内部校准
        if json_path != "":
            self._read_mtx_dist_from_json(json_path)

        if self._mtx is None:
            self._level_print("Warning: Please run inner_calib() first. Auto run.", 1)
            self.inner_calib()

        # 反投影误差
        total_error = 0
        for i in range(len(self._objpoints)):
            imgpoints2, _ = cv2.projectPoints(self._objpoints[i], self._rvecs[i], self._tvecs[i], self._mtx, self._dist)
            error = cv2.norm(self._imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print("total error: ", total_error / len(self._objpoints))

    def _myRPY2R_robot(self, x: float, y: float, z: float) -> np.ndarray[np.float64]:
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
    def _pose_robot(self, Rx: float, Ry: float, Rz: float, Tx: np.ndarray[np.float64], Ty: np.ndarray[np.float64],
                    Tz: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        计算并返回机器人的位姿矩阵。

        :param Rx: 控制机器人的旋转角度，单位为度。
        :param Ry: 控制机器人的旋转角度，单位为度。
        :param Rz: 控制机器人的旋转角度，单位为度。
        :param Tx: 控制机器人的平移量，单位为mm。
        :param Ty: 控制机器人的平移量，单位为mm。
        :param Tz: 控制机器人的平移量，单位为mm。

        :return: RT1: 4x4 的齐次变换矩阵，用于描述机器人在世界坐标系中的位置和姿态。
        """
        # 将角度从度转换为弧度
        thetaX = Rx / 180 * pi
        thetaY = Ry / 180 * pi
        thetaZ = Rz / 180 * pi

        # 根据RPY角计算旋转矩阵
        R = self._myRPY2R_robot(thetaX, thetaY, thetaZ)

        # 定义平移向量
        t = np.array([[Tx], [Ty], [Tz]])

        # 将旋转矩阵和平移向量合并为变换矩阵
        RT1 = np.column_stack([R, t])  # 列合并

        # 添加齐次坐标项，完成4x4变换矩阵的构建
        RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))

        return RT1

    def _read_mtx_dist_from_json(self, json_path: str, if_detial: bool = False):
        self._level_print("Get mtx and dist from " + json_path, 3)
        with (open(json_path, 'r', encoding='utf-8') as json_file):
            data = json.load(json_file)
            self._mtx = np.array(data["mtx"])
            self._dist = np.array(data["dist"])
            if if_detial:
                self._tvecs = [np.array(tvec) for tvec in data["tvecs"]]
                self._rvecs = [np.array(rvec) for rvec in data["rvecs"]]
                self._objpoints = [np.array(objpoint) for objpoint in data["objpoints"]]
                self._imgpoints = [np.array(imgpoint) for imgpoint in data["imgpoints"]]

    def _get_RT_from_chessboard(self, image: Optional[Union[None, np.ndarray]] = None, img_path: str = "",
                               json_path: str = "",
                               if_draw: bool = True, if_check: bool = True) -> Optional[np.ndarray]:
        '''
        从棋盘格图像中获取相机外参。

        :param image: 可选参数，图像的numpy数组。如果提供，则直接使用该图像进行角点检测。
        :param img_path: 图片路径，如果提供，将从该路径读取图像。
        :param json_path: 相机参数json文件的路径。如果为空字符串，则表示不从文件读取参数。
        :param if_draw: 是否在图像上绘制检测到的角点。
        :param if_check: 是否对角点检测结果进行校验。
        :return: 返回相机外参矩阵，如果过程中出现错误则返回None。
        '''

        # 根据json路径读取相机内参矩阵和畸变参数
        if json_path != "":
            self._read_mtx_dist_from_json(json_path)

        # 如果相机内参矩阵未设置，则进行内部校准
        if self._mtx is None:
            self._level_print("Warning: Please run inner_calib() first. Auto run.", 1)
            self.inner_calib()

        # 根据提供的图像路径或图像数据读取图像
        if image is not None:
            img = image
        elif img_path != "":
            img = cv2.imread(img_path)
        else:
            self._level_print("Error: Please provide image path or image.", 1)
            return None

        # 检测棋盘格角点
        corners = self.find_corners(img, if_draw, if_check)
        if corners is None:
            return None

        # 使用cv2.solvePnP计算相机外参
        retval, rvec, tvec = cv2.solvePnP(self._objp, corners, self._mtx, self._dist)
        self._level_print("PNP结算结果评估:\n {0}".format(retval), 4)
        self._level_print("PNP结算结果rvec:\n {0}".format(rvec), 4)
        self._level_print("PNP结算结果tvec:\n {0}".format(tvec), 4)

        # 将旋转向量转换为旋转矩阵，并构建相机外参矩阵RT
        RT = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
        RT = np.row_stack((RT, np.array([0, 0, 0, 1])))

        self._level_print("RT:\n {0}".format(RT), 4)
        return RT

    def make_end_pose_json(self, use_robot_tcp: bool = False) -> str:
        from visual.visual_handle import VisualHandle

        filename = './config/config.json'
        # 使用 with 语句打开文件，确保最后文件会被正确关闭
        with open(filename, 'r', encoding='utf-8') as file:
            # 加载JSON数据
            args = json.load(file)
        camera = VisualHandle(args)

        if use_robot_tcp:
            from robot_tcp import RobotTcp
            robot_tcp = RobotTcp(args["jkrobot"]["ip"], args["jkrobot"]["port"])

        # 图像保存路径
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory_image = self._output_path + self._measure_path + self._image_for_outercalib_path
        if not os.path.exists(directory_image):
            os.makedirs(directory_image)
        # json文件保存路径
        directory_json = self._output_path + self._measure_path + self._end2base
        if not os.path.exists(directory_json):
            os.makedirs(directory_json)
        json_save_path = directory_json + "end2base-" + str(current_time) + ".json"

        datas = []
        get_info_times = 0
        while 1:
            src, _ = camera.get_picture()
            cv2.imshow("src", cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(30)
            if key == ord('w') or key == ord('W'):
                # 记录机械臂末端位姿
                if use_robot_tcp:
                    end_pos = robot_tcp.get_end_pos_jkrobot()
                    strTx, strTy, strTz, strRx, strRy, strRz = end_pos
                else:
                    end_pos_output = input("Input the end pos，unit: m、deg. ep: Tx,Ty,Tz,Rx,Ry,Rz")
                    strTx, strTy, strTz, strRx, strRy, strRz = end_pos_output.split(",")
                strTx *= 1000
                strTy *= 1000
                strTz *= 1000
                # 保存图像
                get_info_times += 1
                image_name = str(current_time) + f"-color_image{get_info_times}" + ".jpg"
                image_save_path = directory_image + image_name
                cv2.imwrite(image_save_path, src)  # 保存去畸变后的图像
                # 保存图像路径以及末端位姿数据
                data = {
                        "endpos": [float(strRx), float(strRy), float(strRz), float(strTx), float(strTy), float(strTz)],
                        "image_name": image_name
                    }
                datas.append(data)
                with open(json_save_path, 'w', encoding='utf-8') as save_path:
                    json.dump(datas, save_path, indent=4, ensure_ascii=False)
                self._level_print("Success: Save.", 1)
            elif key == ord("q") or key == ord("Q"):
                if get_info_times < 0:
                    self._level_print(f"You have saved {get_info_times} infos < 4 infos. go on.", 1)
                else:
                    break
        return json_save_path




    def outer_calib(self, end_pose_path: str = "", inner_param_path: str = "", if_write: bool = False, if_evaluate: bool = False, if_draw: bool = False) -> Optional[
        np.ndarray]:
        """
        执行外部校准，通过给定的机器人末端位姿和对应的棋盘格图像，进行手眼标定。

        参数:
        :param end_pose_path: 末端位姿json文件路径，包含每个位姿的旋转和平移参数。
        :param inner_param_path: 相机内参矩阵和畸变参数的json文件路径。
        :param if_write: 是否将标定结果写入json文件。
        :param if_evaluate: 是否对标定外参进行评估。
        :param if_draw: 是否绘制圆点检测图

        返回值:
        :return: 标定后的相机到机器人末端的变换矩阵(R_cam2end, T_cam2end)。
        """

        # 根据路径读取相机内参和畸变参数
        if inner_param_path != "":
            self._read_mtx_dist_from_json(inner_param_path)

        # 若未设置相机内参矩阵，则先进行内部校准
        if self._mtx is None:
            self._level_print("Warning: Please run inner_calib() first. Auto run.", 1)
            self.inner_calib()

        # 读取末端位姿和对应的棋盘格图像
        if end_pose_path != "":
            with (open(end_pose_path, 'r', encoding='utf-8') as json_file):
                datas = json.load(json_file)
        else:
            datas = self.make_end_pose_json()

        # 初始化存储变换矩阵的列表
        R_all_end_to_base = []
        T_all_end_to_base = []
        R_all_chess_to_cam = []
        T_all_chess_to_cam = []

        with (open("config/camera-calib/cam2end-2024-06-06 20-14-51.json", 'r', encoding='utf-8') as json_file):
            datas0 = json.load(json_file)
            R_came_to_gripper = np.array(datas0["R"])
            T_came_to_gripper = np.array(datas0["T"])

        for data in datas:
            # 计算机器人末端到基座的变换矩阵
            RT1 = self._pose_robot(data["endpos"][0], data["endpos"][1], data["endpos"][2],
                                   data["endpos"][3], data["endpos"][4], data["endpos"][5])
            R_all_end_to_base.append(RT1[:3, :3])
            T_all_end_to_base.append(RT1[:3, 3].reshape((3, 1)))
            # 读取图像并计算棋盘格到相机的变换矩阵
            src = cv2.imread(
                self._output_path + self._measure_path + self._image_for_outercalib_path + data["image_name"])
            RT2 = self._get_RT_from_chessboard(src, if_draw=if_draw)
            R_all_chess_to_cam.append(RT2[:3, :3])
            T_all_chess_to_cam.append(RT2[:3, 3].reshape((3, 1)))

        # 执行手眼标定
        R_cam2end, T_cam2end = cv2.calibrateHandEye(R_all_end_to_base, T_all_end_to_base,
                                                    R_all_chess_to_cam, T_all_chess_to_cam,
                                                    R_came_to_gripper, T_came_to_gripper)
        self._level_print("R_cam2end: \n {0}".format(R_cam2end), 2)
        self._level_print("T_cam2end: \n {0}".format(T_cam2end), 2)
        self._level_print("Outer Calibration Done.", 1)

        # 若需要，则将标定结果写入json文件
        if if_write:
            directory = self._output_path + self._measure_path
            if not os.path.exists(directory):
                os.makedirs(directory)
            data_dict = {}
            data_dict["R"] = R_cam2end.tolist()
            data_dict["T"] = T_cam2end.tolist()
            data_dict["end2base_path"] = end_pose_path
            current_time = datetime.datetime.now()
            save_path = directory + "cam2end-" + str(current_time.strftime("%Y-%m-%d %H-%M-%S")) + ".json"
            with open(save_path, 'w', encoding='utf-8') as save_path:
                json.dump(data_dict, save_path, indent=4, ensure_ascii=False)
            self._level_print("Save outer calibration result to {0}".format(save_path), 1)

        if if_evaluate:
            RT_chess_to_bases = []
            for i in range(len(datas)):
                # 得到机械手末端到基座的变换矩阵，通过机械手末端到基座的旋转矩阵与平移向量先按列合并，然后按行合并形成变换矩阵格式
                RT_end_to_base = np.column_stack((R_all_end_to_base[i], T_all_end_to_base[i]))
                RT_end_to_base = np.row_stack((RT_end_to_base, np.array([0, 0, 0, 1])))
                # print(RT_end_to_base)

                # 标定版相对于相机的齐次矩阵
                RT_chess_to_cam = np.column_stack((R_all_chess_to_cam[i], T_all_chess_to_cam[i]))
                RT_chess_to_cam = np.row_stack((RT_chess_to_cam, np.array([0, 0, 0, 1])))
                # print(RT_chess_to_cam)

                # 手眼标定变换矩阵
                RT_cam_to_end = np.column_stack((R_cam2end, T_cam2end))
                RT_cam_to_end = np.row_stack((RT_cam_to_end, np.array([0, 0, 0, 1])))
                # print(RT_cam_to_end)

                # 即为固定的棋盘格相对于机器人基坐标系位姿

                RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam
                # RT_chess_to_base = np.linalg.inv(RT_chess_to_base)
                print('第', i, '次')
                print(RT_chess_to_base[:3, :])
                print('')
                RT_chess_to_bases.append(RT_chess_to_base[:3, :])
        a = np.array(RT_chess_to_bases)
        sigma = np.std(a, axis=0)
        mean = np.mean(a, axis=0)
        print("sigma: \n")
        print(sigma)
        print("mean: \n")
        print(mean)
        return R_cam2end, T_cam2end



