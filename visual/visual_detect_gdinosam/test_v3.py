from GroundingDino import GroundingDino
from SegmentAnything import SegmentAnything
from ImageProcessV3 import *
from DinoV2 import DinoV2

# image = Image.open("/home/embodied/Pictures/origin/CNC/test23.jpg")
image = Image.open("/home/embodied/Pictures/origin/CNC/table2.jpg")
image_array = np.array(image)
# string = "金属块1"
string = "底座1"

# import time
# start_time = time.time()

gddino = GroundingDino()
# load_gddino_time = time.time()

segmenter = SegmentAnything() 
# load_sam_time = time.time()

dinov2 = DinoV2()
# load_dinov2_time = time.time()  

image_processor = ImageProcessor(image_array, show=True)

boxes_filt, pred_phrases = gddino.get_grounding_output(image_array)
# gddino_time = time.time()

box_masks = image_processor.get_boxes_by_cls(boxes_filt, string, dinov2, segmenter)
# sam_from_box = time.time()

grasp_stack, angles= image_processor.get_rect(box_masks)
# process_from_dinov2 = time.time()

# print(f"load GroundingDino:{load_gddino_time - start_time}s")
# print(f"load SAM:{load_sam_time - load_gddino_time}s")
# print(f"load DinoV2:{load_dinov2_time - load_sam_time}s")

# print(f"process GroundingDino:{gddino_time - load_dinov2_time}s")
# print(f"process SAM+DinoV2:{sam_from_box - gddino_time}s")
# print(f"process masks:{process_from_dinov2 - sam_from_box}s")


