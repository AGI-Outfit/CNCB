import numpy as np
import datetime
import cv2
import re
import json
from visual.camera import Camera
from .pixel2base import Pixel2Base

from visual.visual_detect_gdinosam.GroundingDino import GroundingDino
from visual.visual_detect_gdinosam.SegmentAnything import SegmentAnything
# from visual.visual_detect_gdinosam.ImageProcessV2 import *
from visual.visual_detect_gdinosam.DinoV2 import DinoV2
from visual.visual_detect_gdinosam.ImageProcessV3 import *

class VisualHandle:
    def __init__(self, args):
        self.args = args
        self.args_camera = self.args["visual"]["camera"]
        self.args_fragment = self.args["visual"]["fragment"]
        self.ip = self.args_camera["ip"]
        self.port = self.args_camera["port"]
        self.log_level = self.args["base"]["log_level"]
        self.pixel2base = Pixel2Base(self.args_camera["inner_calib_path"],
                                     self.args_camera["outer_calib_path"],
                                     self.log_level)
        self.use_camera = self.args_camera["use_camera"]
        self.fragment_read_from_json = self.args_fragment["fragment_read_from_json"]
        self.points_save_path = "output/fragment/points.json"
        if not self.fragment_read_from_json:
            self.gddino = GroundingDino()
            self.segmenter = SegmentAnything()
            self.dinov2 = DinoV2()
        if self.use_camera:
            # 创建socket对象
            self.camera = Camera(self.ip, self.port)
            # 连接服务器
            self.level_print("Success: Camera init.", 1)

    def get_aligned_depth(self, points: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        得到输入对应坐标的深度图的值矩阵
        :param points size(points_num, 2) (v, u)
        :param depth_image size(height, width) 深度值(mm)
        :return: result size(points_num, 1) 深度值(mm)
        """
        result = depth_image[points[:, 0], points[:, 1]]
        return result

    def get_filtered_aligned_depth(self, points: np.ndarray, depth_image: np.ndarray, if_filtered: bool = True) -> (np.ndarray, np.ndarray):
        """
        得到输入对应坐标的深度图的值矩阵，并且滤波掉1sigma以内的坐标
        :param points size(points_num, 2) (v, u)
        :param depth_image size(height, width) 深度值(mm)
        :param if_filtered 是否滤波
        :return:result_depths size(points_num_filtered, 1) 深度值(mm)
                result_points size(points_num_filtered, 2) (v, u)
         """
        result = depth_image[points[:, 0], points[:, 1]]
        filtered_points = points
        if if_filtered:

            # 使用nonzero函数获取所有非零元素的索引
            non_zero_indices = np.nonzero(result)

            # 使用这些索引从原数组中提取所有非零元素
            non_zero_elements = result[non_zero_indices]

            # 使用flatten函数将结果展平成一维数组
            result = non_zero_elements.flatten()

            # 求数组的平均值和方差
            mean = np.mean(result)
            std = np.std(result)
            filtered_indices = (result > mean - std) & (result < mean + std)
            within_one_sigma = result[filtered_indices]
            # # 计算出最为可信的深度信息
            # output = np.mean(within_one_sigma)
            result = within_one_sigma
            filtered_points = points[non_zero_indices][filtered_indices]
        return result, filtered_points

    # def pixeldist_to_worlddist(self, point1: np.ndarray, point2: np.ndarray, depth_src: np.ndarray) -> float:
    #     """
    #     将像素距离转换为世界距离。
    #     :param point1: size(2, 1)
    #     :param point2: size(2, 1)
    #     :param depth_src (mm)
    #
    #     :return: dist (mm)
    #     """
    def instance_fragment(self, color_src: np.ndarray, object_name: str):
        if not self.fragment_read_from_json:
            # match = re.search(r'\d+', object_name)
            # if match:
            #     # 如果找到数字，打印出来
            #     cls = int(match.group(0))  # 将匹配的数字字符串转换为整数
            #     self.level_print("Find object num: {0}".format(cls), 3)
            # else:
            #     self.level_print("No number found in string: {0}".format(object_name), 1)
            #     return None


            image_processor = ImageProcessor(color_src, show=True)

            boxes_filt, pred_phrases = self.gddino.get_grounding_output(color_src)
            # gddino_time = time.time()

            box_masks = image_processor.get_boxes_by_cls(boxes_filt, object_name, self.dinov2, self.segmenter, show_cls=True)
            # sam_from_box = time.time()
            fragments_points, angles = image_processor.get_rect(box_masks)
            objects_num = len(fragments_points)
            datas = []
            for i in range(objects_num):
                datas.append({"fragment_points": fragments_points[i], "angle": angles[i]})
            with open(self.points_save_path, 'w', encoding='utf-8') as file:
                # 加载JSON数据
                json.dump(datas, file)
        else:
            fragments_points = []
            angles = []
            with open(self.points_save_path, 'r', encoding='utf-8') as file:
                # 加载JSON数据
                datas = json.load(file)
                for data in datas:
                    angles.append(data["angle"])
                    fragments_points.append(data["fragment_points"])
        return fragments_points, angles

    def fragment(self, color_src):
        if not self.fragment_read_from_json:
            import time

            # start_time = time.time()
            boxes_filt, pred_phrases = self.gddino.get_grounding_output(color_src)
            # print(time.time() - start_time)
            # start_time = time.time()
            masks = self.segmenter.get_sam_box_results(color_src, boxes_filt)  # show_each显示masks
            # print(time.time() - start_time)
            # start_time = time.time()
            self.segmenter.show_masks(masks, choose_show=True)  # choose_show显示masks
            # print(time.time() - start_time)
            # start_time = time.time()
            fragments_points, angles = get_rect(masks, color_src, show_points=True)  # show_points显示points




            # print(time.time() - start_time)
            objects_num = len(fragments_points)
            datas = []
            for i in range(objects_num):
                datas.append({"fragment_points": fragments_points[i], "angle": angles[i]})
            with open("output/fragment/points.json", 'w', encoding='utf-8') as file:
                # 加载JSON数据
                json.dump(datas, file)
        else:
            fragments_points = []
            angles = []
            with open("output/fragment/points.json", 'r', encoding='utf-8') as file:
                # 加载JSON数据
                datas = json.load(file)
                for data in datas:
                    angles.append(data["angle"])
                    fragments_points.append(data["fragment_points"])
        return fragments_points, angles


    def get_picture(self, if_show=False, if_save=False):
        if self.use_camera:
            # color data
            color_array = self.camera.getColorImage()
            self.level_print(color_array.shape, 5)

            # depth data
            depth_origin_array = self.camera.getDepthImage()
            self.level_print(depth_origin_array.shape, 5)

            # depth data only for show
            depth_show_array = depth_origin_array / (depth_origin_array.max()+1)

            #color data
            self.level_print(color_array.shape, 5)

            # 显示
            if if_show:
                cv2.imshow('Color Image', cv2.cvtColor(color_array, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth Image", depth_show_array)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # 保存
            if if_save:
                self.save_image(color_array=color_array, depth_origin_array=depth_origin_array, depth_show_array=depth_show_array)

            return color_array.copy(), depth_origin_array.copy()
        else:
            return None, None

    def level_print(self, input_str, level):
        if level <= self.log_level:
            print(input_str)

    def save_image(self, color_array: np.ndarray = None, depth_origin_array: np.ndarray = None, depth_show_array: np.ndarray = None):
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = "output/camera/"
            if color_array is not None:
                cv2.imwrite(save_path + "color-" + current_time + ".jpg", color_array)
            if depth_origin_array is not None:
                cv2.imwrite(save_path + "depth-origin-" + current_time + ".png", depth_origin_array)
            if depth_show_array is not None:
                cv2.imwrite(save_path + "depth-show-" + current_time + ".jpg", (depth_show_array*255).astype(np.uint8))


if __name__ == '__main__':
    filename = '../config/config.json'
    # 使用 with 语句打开文件，确保最后文件会被正确关闭
    with open(filename, 'r', encoding='utf-8') as file:
        # 加载JSON数据
        args = json.load(file)
    camera = Camera(args)
