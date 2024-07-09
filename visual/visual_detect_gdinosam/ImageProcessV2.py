import numpy as np
import cv2
from PIL import Image
import sys
import torch.nn.functional as F
sys.path.append("visual/visual_detect_gdinosam/")
import re
import GroundingDINO.groundingdino.datasets.transforms as T
from torchvision import transforms
import torch

# class ImageProcessor:
#     def __init__(self, image_array : np.ndarray, masks):
#         self.image = image_array
#         self.masks = masks
#         self.new_contours = []
#         self.dst_points = np.array([
#                 [0, 0],[800, 0],
#                 [0, 1600],[800, 1600]
#                 ], dtype=np.float32)
        
# ================ masks to points ================
def filter_regions_by_area(regions : np.array)-> np.array:
    filtered_areas = np.zeros(len(regions))
    for i, region in enumerate(regions):
        # 计算区域的左上角和右下角坐标
        x1, y1 = region[0], region[1]
        x2, y2 = region[2], region[3]
        # 计算区域的宽度和高度
        width = x2 - x1
        height = y2 - y1
        # 计算区域的面积
        area = width * height
        filtered_areas[i] = np.abs(area)
        # 检查面积是否在阈值范围内
    mean_area = np.mean(filtered_areas)
    index = np.where((0.9 * filtered_areas <= mean_area) & (filtered_areas >= 0.2 * mean_area))
    filtered_regions = regions[index]
    return filtered_regions

def get_from_string(string : str) -> str:
    pattern = r'(金属块|底座)\s*(\d+)'
    # 使用 re.search 查找匹配项
    match = re.search(pattern, string)
    # 如果找到匹配项
    if match:
        # 获取匹配的整个文本
        full_match = match.group(0)
        # 获取第一个捕获组匹配的文本，即 "金属块" 或 "底座"
        block_type = match.group(1)
        # 获取第二个捕获组匹配的文本，即数字
        cls = int(match.group(2))
        print(f"Full match: {full_match}, Block type: {block_type}, cls: {cls}")
    else:
        print("No match found.")
    return block_type, cls

def get_half_img(image):
    h, w = image.shape[:2]
    half_width = w // 2
    image = image[:, half_width:]
    return image


def get_boxes_by_cls(image_array, boxes, string, dinov2, segmenter):
    block_type, cls = get_from_string(string)
    boxes = np.round(boxes).astype(int)
    boxes = filter_regions_by_area(boxes)
    box_mask = []
    prompt1_image = cv2.imread(f"visual/visual_detect_gdinosam/assets/prompt/{block_type}1.jpg")
    prompt2_image = cv2.imread(f"visual/visual_detect_gdinosam/assets/prompt/{block_type}2.jpg")
    if block_type == "底座":
        prompt1_image = get_half_img(prompt1_image)
        prompt2_image = get_half_img(prompt2_image)
    ref1_feats =dinov2.get_feats(prompt1_image)
    ref2_feats =dinov2.get_feats(prompt2_image)
    for box in boxes:
        cropped_image = image_array[box[1]:box[3], box[0]:box[2]] if block_type == "金属块" else \
                image_array[box[1]:box[3], int((box[2]+box[0])/2) : box[2]]
        with torch.no_grad():
            fea = dinov2.get_feats(cropped_image)
        score1 = F.cosine_similarity(ref1_feats, fea, dim=1, eps=1e-8)
        score2 = F.cosine_similarity(ref2_feats, fea, dim=1, eps=1e-8)
        index = 1 if score1 > score2 else 2
        text_position = (box[0] + 20, box[1] + 20)
        # 在图像上绘制编号
        if index == cls:
            mask = segmenter.get_sam_box_results(image_array, torch.tensor(box), show_each = False) # show_each显示masks
            cv2.putText(image_array, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 255), 5, cv2.LINE_AA)
            box_mask.append(mask.squeeze(1))
    # cv2.namedWindow("image")
    # cv2.imshow("image", image_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return box_mask

def filter_masks(masks: torch.tensor) -> torch.tensor:
    ## chose the region of right area
    areas = []
    for mask in masks.cpu().numpy():
        areas.append(mask.sum())
    areas = np.array(areas)
    # areas = np.array(mask.sum() for mask in masks.cpu().numpy())
    similarity_threshold = areas.sum() / areas.shape[0]
    masks = masks[0.8 * areas < similarity_threshold]
    return masks

def combine_masks(image: np.ndarray, masks: torch.tensor) -> np.ndarray:
    masks = filter_masks(masks)
    binary_mask = (masks.cpu().numpy() * 255).astype(np.uint8)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(len(binary_mask)):
        # 注意：这里假设掩码值为1表示对应的区域有效，如果是其他情况请适当调整
        combined_mask = np.logical_or(combined_mask, binary_mask[i, 0]).astype(np.uint8)
    return combined_mask

def warp_perspective(image, src_points, dst_points):
    # print(src_points, "////",self.dst_points)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(image, M, (800, 1600))

    transformed_image = cv2.warpPerspective(image, M, (800, 1600))

    cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    radius = 200
    transformed_center = np.array([[400, 800]])
    points_in_circle = circle_points(image, (400, 800), radius)
    # transformed_width, transformed_height = 800, 1600
    # # 计算矩形的边界
    # xmin = transformed_center[0][0] - transformed_width // 4  # 假设 points[0] 是所有 x 坐标的数组
    # ymin = transformed_center[0][1] - transformed_height // 4  # 假设 points[1] 是所有 y 坐标的数组
    # xmax = transformed_center[0][0] + transformed_width // 4  # 假设 points[0] 是所有 x 坐标的数组
    # ymax = transformed_center[0][1] + transformed_height // 4  # 假设 points[1] 是所有 y 坐标的数组
    # # # 生成水平边的点
    # # horizontal_edges = np.array([[xmin, y] for y in np.arange(ymin, ymax + 1)] + 
    # #                             [[xmax, y] for y in np.arange(ymin, ymax + 1)])
    # # # 生成垂直边的点
    # # vertical_edges = np.array([[x, ymin] for x in np.arange(xmin, xmax + 1)] + 
    # #                             [[x, ymax] for x in np.arange(xmin, xmax + 1)])
    # # # 合并所有边的点
    # # all_points = np.concatenate((transformed_center, horizontal_edges, vertical_edges)).reshape(-1, 1, 2).astype(np.float32)
    # # grasp_points = np.int32(cv2.perspectiveTransform(all_points, np.linalg.inv(M).astype(np.float32)))
    # content_points = np.array([(x,y) for x in np.arange(xmin, xmax + 1) \
    #                                  for y in np.arange(ymin, ymax + 1)])
    all_points = np.concatenate((transformed_center,points_in_circle)).reshape(-1, 1, 2).astype(np.float32)
    # # 变换回去
    grasp_points = np.int32(cv2.perspectiveTransform(all_points, np.linalg.inv(M).astype(np.float32)))
    # print(type(grasp_points), len(grasp_points),  grasp_points.shape)
    return transformed_image, grasp_points

def circle_points(image, center, radius) -> np.array:
    # 绘制轮廓和最小外接矩形
    cv2.circle(image, center, int(radius), (0, 0, 0), 5)
    height, width = image.shape[:2]
    # 创建一个表示所有图像坐标的矩阵
    yx = np.indices((height, width), dtype=np.int32)
    yx = yx.reshape(2, -1).T  # 转换为 (height * width, 2) 形状
    # 计算所有点到圆心的距离的平方
    distances_squared = (yx[:, 0] - center[1])**2 + (yx[:, 1] - center[0])**2
    # 找到所有在圆内（包括圆上）的点的索引
    points_in_circle_indices = np.where(distances_squared <= radius**2)
    # 如果你需要所有在圆内的点的坐标
    points_in_circle = yx[points_in_circle_indices]
    for point in points_in_circle:
            # 将点转换为列表或形状为(2,)的数组，因为cv2.circle需要这样的参数
            y, x = point[0], point[1]
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return points_in_circle

def get_rect(mask, image_array, show_points=False):
    dst_points = np.array([
                [0, 0],[800, 0],
                [0, 1600],[800, 1600]
                ], dtype=np.float32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("mask", mask*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 选择最大的轮廓
    max_areas = []
    grasp_stack = []
    corner_stack = []
    angles = []
    j = 0
    for i, contour in enumerate(contours):
        # 计算轮廓的中心点（质心）
        area = cv2.contourArea(contour)
        if area < 10000:
            j += 1# 在这里处理或绘制满足条件的轮廓
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        drawing = np.zeros_like(mask, dtype=np.uint8)
        # 绘制近似多边形
        if len(approx) > 2:
            # 绘制多边形的边界
            cv2.polylines(drawing, [approx], isClosed=True, color=255, thickness=2, lineType=cv2.LINE_AA)
        elif len(approx) < 4:
            print("近似多边形的顶点数少于4个，无法形成四边形。")
        # 寻找最小面积四边形
        max_area = 0
        max_polygon = None
        for i in range(len(approx)):
            for j in range(i + 1, len(approx)):
                # 检查两个顶点是否相邻
                if np.linalg.norm(approx[i] - approx[j]) < epsilon:
                    continue
                # 寻找四个顶点构成的四边形
                polygon = [approx[i], approx[(i + 1) % len(approx)], approx[j], approx[(j + 1) % len(approx)]]
                area = cv2.contourArea(np.array([polygon], dtype=np.float32).reshape(4, 2))
                if area > max_area:
                    max_area = area
                    max_polygon = polygon
        max_areas.append(max_area)
        # 绘制最小四边形
        if max_polygon is not None:
            for point in max_polygon:
                cv2.circle(image_array, (int(point[0][0]), int(point[0][1])), 5, (0, 255, 0), -1)
        corner_stack.append(max_polygon)
        points = np.array(max_polygon).reshape(4,2)
        center = np.mean(points, axis=0).astype(np.int32)
        radius = min(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[0] - points[2]),np.linalg.norm(points[0] - points[3])) // 4
        points_in_circle = circle_points(image_array, (center[0], center[1]), radius)
        
        angle = get_angle(contour)
        angles.append(angle)
        text_position = (center[0] - 50, center[1] + 20)  # 根据需要调整位置
        cv2.putText(image_array, f"Angle{i-j+1}: {angle}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # src_points = np.array(points_in_circle).reshape(4, 2).astype(np.float32)
        # transformed_image, grasp_points = warp_perspective(image_array, src_points, dst_points) 
        grasp_stack.append(points_in_circle.tolist())
    if show_points:
        # cv2.namedWindow("Approximated Polygon", cv2.WINDOW_NORMAL)
        # cv2.imshow('Approximated Polygon', drawing)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.namedWindow("Minimum Enclosing Quadrilateral", cv2.WINDOW_NORMAL)
        cv2.imshow('Minimum Enclosing Quadrilateral', image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return grasp_stack, angles

def get_angle(contour):
    # 计算最小外接矩形
    box = cv2.minAreaRect(contour)
    (x, y), (width, height), angle = box
    if height > width:
        # 如果 width 更长，交换它们，并调整角度
        width, height = height, width
        angle = angle-90  # 如果需要，调整角度的符号
    angle = round(angle, 2)
    return -angle

# ================ crop/transform image to dinov2 ================
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_image_crop_resize(image, box, resize_shape):
    """Crop image according to the box, and resize the cropped image to resize_shape
    @param image: the image waiting to be cropped
    @param box: [x0, y0, x1, y1]
    @param resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    scale = np.array([box[2] - box[0], box[3] - box[1]])

    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    image_crop = cv2.warpAffine(
        image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR
    )

    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)
    return image_crop, trans_crop_homo


# if __name__ == "__main__":
#     processor = ImageProcessor("outputs/material/5/raw_image.jpg", "outputs/material/5/mask.jpg")
#     src_points = processor.process_image()
#     # print(src_points)
