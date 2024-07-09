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

class ImageProcessor:
    def __init__(self, image_array: np.ndarray,  show = False):
        self.image = image_array
        self.show = show
        # self.dst_points = np.array([
        #         [0, 0],[800, 0],
        #         [0, 1600],[800, 1600]
        #         ], dtype=np.float32)
    # ================ get_boxes ================
    def filter_regions_by_area(self, regions: torch.Tensor) -> torch.Tensor:
        filtered_areas = torch.zeros(len(regions))
        for i, region in enumerate(regions):
            x1, y1 = region[0], region[1]
            x2, y2 = region[2], region[3]
            width = x2 - x1
            height = y2 - y1
            area = width * height
            filtered_areas[i] = torch.abs(area)
        mean_area = torch.mean(filtered_areas)
        index = np.where((0.9 * filtered_areas <= mean_area) & (filtered_areas >= 0.2 * mean_area))
        filtered_regions = regions[index]
        return filtered_regions

    def get_from_string(self, string: str) -> str:
        pattern = r'(金属块|底座)\s*(\d+)'
        match = re.search(pattern, string)
        if match :
            full_match = match.group(0)
            block_type = match.group(1)
            cls = int(match.group(2))
            if block_type:
                print(f"Full match: {full_match}, Block type: {block_type}, cls: {cls}")
                return block_type, cls
            else:
                print("输入物体错误.")
        else:
            print("匹配失败.")       

    def get_half_img(self, image):
        h, w = image.shape[:2]
        half_width = w // 2
        image = image[:, half_width:]
        return image
    def show_boxes(self, boxes):
        show_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        self.boxes = boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(show_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.namedWindow("boxes")
        cv2.imshow("boxes", show_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_masks(self, masks):
        show_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        color_mask = np.zeros_like(self.image)
        color_mask[masks > 0] = (255, 255, 0)
        maks_image = cv2.addWeighted(show_image, 0.4, color_mask, 0.6, 0)
        cv2.namedWindow("masks")
        cv2.imshow("masks", maks_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def get_boxes_by_cls(self, boxes_filt, string, dinov2, segmenter, show_cls = False):
        block_type, cls = self.get_from_string(string)
        boxes_filt = torch.round(boxes_filt).to(torch.int32)
        boxes_filt = self.filter_regions_by_area(boxes_filt) 
        self.show_boxes(boxes_filt.numpy()) if self.show else None
        target_boxes = []
        prompt1_image = cv2.imread(f"visual/visual_detect_gdinosam/assets/prompt/{block_type}1.jpg")
        prompt2_image = cv2.imread(f"visual/visual_detect_gdinosam/assets/prompt/{block_type}2.jpg")
        if block_type == "底座":
            prompt1_image = self.get_half_img(prompt1_image)
            prompt2_image = self.get_half_img(prompt2_image)
        ref1_feats =dinov2.get_feats(prompt1_image)
        ref2_feats =dinov2.get_feats(prompt2_image)
        for tenor_box in boxes_filt:
            box = tenor_box.numpy()
            cropped_image = self.image[box[1]:box[3], box[0]:box[2]] if block_type == "金属块" else \
                    self.image[box[1]:box[3], int((box[2]+box[0])/2) : box[2]]
            with torch.no_grad():
                fea = dinov2.get_feats(cropped_image)
            score1 = F.cosine_similarity(ref1_feats, fea, dim=1, eps=1e-8)
            score2 = F.cosine_similarity(ref2_feats, fea, dim=1, eps=1e-8)
            index = 1 if score1 > score2 else 2
            text_position = (box[0] + 50, box[1] + 20)
            if index == cls:
                target_boxes.append(tenor_box)
                cv2.putText(self.image, str(index), text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 5, cv2.LINE_AA)
        box_masks = segmenter.get_sam_box_results(self.image, torch.stack(target_boxes)) # (n ,1, h, w)
        masks = self.combine_masks(box_masks.squeeze(1)) # (n, h, w) -> # (h, w)
        self.show_masks(masks) if self.show else None
        return masks

    # ================ get_rect_corner ================
    def filter_masks(self, masks: torch.tensor) -> torch.tensor:
        ## chose the region of right area
        areas = []
        for mask in masks.cpu().numpy():
            areas.append(mask.sum())
        areas = np.array(areas)
        # areas = np.array(mask.sum() for mask in masks.cpu().numpy())
        similarity_threshold = areas.sum() / areas.shape[0]
        masks = masks[0.8 * areas < similarity_threshold]
        return masks

    def combine_masks(self, masks: torch.tensor) -> np.ndarray:
        masks = self.filter_masks(masks)
        binary_mask = (masks.cpu().numpy() * 255).astype(np.uint8)
        combined_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for i in range(len(binary_mask)):
            combined_mask = np.logical_or(combined_mask, binary_mask[i]).astype(np.uint8)
        return combined_mask

    def rect_points(self, transformed_center):
        transformed_width, transformed_height = 800, 1600
        xmin = transformed_center[0][0] - transformed_width // 4  # 假设 points[0] 是所有 x 坐标的数组
        ymin = transformed_center[0][1] - transformed_height // 4  # 假设 points[1] 是所有 y 坐标的数组
        xmax = transformed_center[0][0] + transformed_width // 4  # 假设 points[0] 是所有 x 坐标的数组
        ymax = transformed_center[0][1] + transformed_height // 4  # 假设 points[1] 是所有 y 坐标的数组
        x_range = np.arange(xmin, xmax + 1)
        y_range = np.arange(ymin, ymax + 1)
        points_in_rect = np.stack(np.meshgrid(x_range, y_range), axis=-1).reshape(-1, 2)
        for point in points_in_rect:
            y, x = point[0], point[1]
            cv2.circle(self.image, (x, y), 5, (0, 255, 255), -1)
        return points_in_rect

    def show_points(self, contours_info):
        show_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB)
        for approx, points_in_circle, corners, center, angle in contours_info:
            # cv2.polylines(show_image, [approx], isClosed=True, color=(255, 0,255), thickness=2, lineType=cv2.LINE_AA)
            [cv2.circle(show_image, (int(corners[i][0]), int(corners[i][1])), 5, (0, 255, 255), -1) for i in range(4)]
            for point in points_in_circle:
                y, x = point[0], point[1]
                cv2.circle(show_image, (x, y), 5, (0, 255, 255), -1)
            text_position = (center[0] - 50, center[1] + 20)
            cv2.putText(show_image, f"Angle: {angle}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        print("show points")
        cv2.namedWindow("Points")
        cv2.imshow('Points', show_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def circle_points(self, center, corners) -> np.array:
        radius = min(np.linalg.norm(corners[0] - corners[i]) for i in range(1, 4)) // 4
        cv2.circle(self.image, center, int(radius), (0, 0, 0), 5)
        height, width = self.image.shape[:2]
        yx = np.indices((height, width), dtype=np.int32)
        yx = yx.reshape(2, -1).T  
        distances_squared = (yx[:, 0] - center[1])**2 + (yx[:, 1] - center[0])**2
        points_in_circle_indices = np.where(distances_squared <= radius**2)
        points_in_circle = yx[points_in_circle_indices]
        return points_in_circle

    def warp_perspective(self, src_points, dst_points):
        # print(src_points, "////",self.dst_points)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_image = cv2.warpPerspective(self.image, M, (800, 1600))

        cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Transformed Image', transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        radius = 200
        transformed_center = np.array([[400, 800]])
        points_in_circle = self.circle_points((400, 800), radius)

        all_points = np.concatenate((transformed_center,points_in_circle)).reshape(-1, 1, 2).astype(np.float32)
        # # 变换回去
        grasp_points = np.int32(cv2.perspectiveTransform(all_points, np.linalg.inv(M).astype(np.float32)))
        # print(type(grasp_points), len(grasp_points),  grasp_points.shape)
        return transformed_image, grasp_points
    
    def get_angle(self, contour):
        box = cv2.minAreaRect(contour)
        (x, y), (width, height), angle = box
        if height > width:
            # 如果 width 更长，交换它们，并调整角度
            width, height = height, width
            angle = angle-90  # 如果需要，调整角度的符号
        angle = round(angle, 2)
        return -angle

    def get_rect(self, box_masks):
        contours, _ = cv2.findContours(box_masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(box_masks, contours, -1, (0, 255, 0), 2)
        # cv2.imshow("masks", box_masks*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 选择最大的轮廓
        contours_info = []
        max_areas = []
        angles = []
        grasp_points = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 10000:
                continue
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
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
                corners = np.array(max_polygon).reshape(4,2)
                center = np.mean(corners, axis=0).astype(np.int32)
                points_in_circle = self.circle_points((center[0], center[1]), corners)
                angle = self.get_angle(contour)
                contours_info.append((approx, points_in_circle, corners, center, angle)) if self.show else None

                angles.append(angle)
                grasp_points.append(points_in_circle.tolist())
            else:
                print("为检测出四边形")
            
        self.show_points(contours_info) if self.show else None
            
            # src_points = np.array(points_in_circle).reshape(4, 2).astype(np.float32)
            # transformed_image, grasp_points = warp_perspective(self.image, src_points, dst_points) 
        return grasp_points, angles



# if __name__ == "__main__":
#     processor = ImageProcessor("outputs/material/5/raw_image.jpg", "outputs/material/5/mask.jpg")
#     src_points = processor.process_image()
#     # print(src_points)