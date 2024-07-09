from PIL import Image
import numpy as np
import torch
import visual.visual_detect_gdinosam.GroundingDINO.groundingdino.datasets.transforms as T
import sys

sys.path.append("visual/visual_detect_gdinosam/")

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

class GroundingDino:
    def __init__(self, base_path = "visual/visual_detect_gdinosam/"):
        self.groundingdino_config_file = base_path + "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.groundingdino_checkpoint = base_path + "weights/groundingdino_swint_ogc.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.groundingdino_model = self.load_grounding_model()

# ================== Init groundingdino model ==================
    def load_grounding_model(self):
        args = SLConfig.fromfile(self.groundingdino_config_file)
        args.device = self.device
        groundingdino_model = build_model(args)
        checkpoint = torch.load(self.groundingdino_checkpoint, map_location="cpu")
        load_res = groundingdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        groundingdino_model.eval()
        return groundingdino_model.to(self.device)
    
    def load_image_to_gddino(self, image_array: np.ndarray):
        image_pil = Image.fromarray(image_array).convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return  image

# ================== Get groundingdino result ==================
    def filter_outputs(self, logits: torch.Tensor, boxes: torch.Tensor, box_threshold) -> torch.Tensor:
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]
        return logits_filt, boxes_filt

    def get_pred_phrases(self, logits_filt, boxes_filt, caption, text_threshold, with_logits=True):
        tokenlizer = self.groundingdino_model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        return pred_phrases

    def resize_box(self, image: np.ndarray, boxes_filt: torch.Tensor) -> torch.Tensor:
        H, W = image.shape[:2]
        boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]
        boxes_filt = boxes_filt.cpu()
        return boxes_filt

    def get_grounding_output(self, image : np.ndarray, text_prompt = "many silver metal_blocks",box_threshold=0.3, text_threshold=0.25):
        tranformed_image = self.load_image_to_gddino(image)
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption += "."
        with torch.no_grad():
            outputs = self.groundingdino_model(tranformed_image[None].to(self.device), captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        logits_filt, boxes_filt = self.filter_outputs(logits, boxes, box_threshold)
        pred_phrases = self.get_pred_phrases(logits_filt, boxes_filt, caption, text_threshold)
        boxes_filt = self.resize_box(image, boxes_filt)
        return boxes_filt, pred_phrases
    
if __name__ == "__main__":
    image = Image.open("/home/embodied/Pictures/origin/CNC/test1.jpg")
    image_array = np.array(image)
    # image_pil = Image.fromarray(image_array).convert("RGB")
    # transform = T.Compose([
    #     T.RandomResize([800], max_size=1333),
    #     T.ToTensor(),
    #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    # image, _ = transform(image_pil, None)
    gddino = GroundingDino(image_array)
    boxes_filt, pred_phrases = gddino.get_grounding_output()
    print(pred_phrases)