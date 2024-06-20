import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# from FastSam.fastsam import FastSAM, FastSAMPrompt
from GroundingDino import GroundingDino
import GroundingDINO.groundingdino.datasets.transforms as T
import time


class SegmentAnything:
    def __init__(self,  base_path = "visual/visual_detect_gdinosam/"):
        self.sam_version = "vit_b"
        self.sam_checkpoint = base_path + "weights/sam_vit_b_01ec64.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model = self.initialize_sam()
        self.predictor = SamPredictor(self.sam_model)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)

# ================== Init SAM model ==================
    def initialize_sam(self):
        sam_model = sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device)
        return sam_model
    
# ================== Get SAM result ==================
    def get_sam_box_results(self, image: np.ndarray, boxes_filt: torch.Tensor) -> np.ndarray:
        image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_cv)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(self.device)
        # Segment with SAM
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        # count = np.count_nonzero(masks > 0)
        return masks
    
    def get_sam_entire_results(self, image: np.ndarray) -> np.ndarray: 
        masks = self.mask_generator.generate(image)
        return masks
    
    def show_masks(self, masks, choose_show = False):
        if choose_show:
            cv2.namedWindow("masks", cv2.WINDOW_NORMAL)
            cv2.imshow("masks", masks*255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# class FastSam():
#     def __init__(self,  base_path = "visual/visual_detect_gdinosam/"):
#         self.fastsam_checkpoint = base_path + "weights/FastSAM-x.pt"
#         self.fastsam_model = FastSAM(self.fastsam_checkpoint)
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     def get_masks(self, image, mode:str = "everything", bboxes = [],text:str = "", points = [], pointlabel = []) -> np.array:
#         bboxes = load_image_to_sam(image, bboxes).numpy()
#         for i, bbox in enumerate(bboxes):
#             bbox = bbox.tolist()  # 转换成列表以便操作
#             bbox = [int(i) for i in bbox]  # 确保为整数
#             # 注意：这里的顺序假设为 [x_min, y_min, x_max, y_max]，根据你的实际格式调整
#             x1, y1, x2, y2 = bbox
#             cropped_image = image[y1-15:y2+15, x1-15:x2+1, :]
#             cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
#             cv2.imshow("crop", cropped_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#             everything_results = self.fastsam_model(cropped_image, device=self.device, retina_masks=True, imgsz=max(image.shape[:2]), conf=0.4, iou=0.9,)
#             prompt_process = FastSAMPrompt(cropped_image, everything_results, device=self.device)
#             ann = prompt_process.everything_prompt()
#         # if mode == "everything":
#         #     ann = prompt_process.everything_prompt()
#         # elif mode == "box":
#         #     if isinstance(bboxes, torch.Tensor):
#         #         bboxes = load_image_to_sam(image, bboxes).numpy()
#         #     ann = prompt_process.box_prompt(bboxes=bboxes)   # bboxes=[[200, 200, 300, 300]]
#         # elif mode == "text":
#         #     ann = prompt_process.text_prompt(text)    # text='a photo of metal'
#         # elif mode == "point":
#         #     ann = prompt_process.point_prompt(points) # points=[[620, 360]], pointlabel=[1]
#             prompt_process.plot(annotations=ann,output_path=f'./output/fastsam/metal3/mask{i}.jpg',) # show and save
#         return ann

# if __name__ == "__main__":

# #================= SAM =================
#     image = Image.open("/home/embodied/Pictures/origin/CNC/test3.jpg")
#     image_array = np.array(image)
#     image_pil = Image.fromarray(image_array).convert("RGB")

#     gddino = GroundingDino()
    # segmenter = SegmentAnything()   
    # boxes_filt, pred_phrases = gddino.get_grounding_output(image_array)
    # masks = segmenter.get_sam_box_results(image_array, boxes_filt, show_each=True)

#================= fast SAM =================
    # # image = Image.open("/home/embodied/Pictures/origin/CNC/test1.jpg")
    # # image_array = np.array(image)
    # fastsam = FastSam()
    # ann =  fastsam.get_masks(image_array, mode= "box", bboxes=boxes_filt)

    # model = FastSAM('visual/visual_detect_gdinosam/weights/FastSAM-x.pt')
    # IMAGE_PATH = '/home/embodied/Pictures/origin/CNC/test3.jpg'
    # DEVICE = 'cuda'
    # everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    # prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
    # everything prompt
    # ann = prompt_process.everything_prompt()
    # # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    # ann = prompt_process.box_prompt(bboxes=[[200, 200, 300, 300]])
    # # text prompt
    # ann = prompt_process.text_prompt(text='a photo of metal')
    # # point prompt
    # # points default [[0,0]] [[x1,y1],[x2,y2]]
    # # point_label default [0] [1,0] 0:background, 1:foreground
    # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])
    # prompt_process.plot(annotations=ann,output_path='./output/fastsam/metal3.jpg',)