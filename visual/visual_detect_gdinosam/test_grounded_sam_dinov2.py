from GroundingDino import GroundingDino
from SegmentAnything import SegmentAnything
from ImageProcessV2 import *
from DinoV2 import DinoV2


# image = Image.open("/home/embodied/Pictures/origin/CNC/test23.jpg")
image = Image.open("/home/embodied/Pictures/origin/CNC/table2.jpg")
image_array = np.array(image)
# string = "金属块2"
string = "底座2"

gddino = GroundingDino()
segmenter = SegmentAnything() 
dinov2 = DinoV2()
boxes_filt, pred_phrases = gddino.get_grounding_output(image_array)
box_mask = get_boxes_by_cls(image_array, boxes_filt, string, dinov2, segmenter)
masks = combine_masks(image_array, torch.stack(box_mask)) # filter the big one
grasp_stack, angles= get_rect(masks, image_array, show_points=True)
