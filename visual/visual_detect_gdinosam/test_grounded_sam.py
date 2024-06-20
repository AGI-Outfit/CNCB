from GroundingDino import GroundingDino
from SegmentAnything import SegmentAnything
from ImageProcessV2 import *
import matplotlib.pyplot as plt
import time 

image = Image.open("/home/embodied/Pictures/origin/CNC/table.jpg")
image_array = np.array(image)

import time 
time_start = time.time()
gddino = GroundingDino()
load_gd_time = time.time()
segmenter = SegmentAnything() 
load_seg_time = time.time()

start = time.time()
boxes_filt, pred_phrases = gddino.get_grounding_output(image_array)
gddino_time = time.time()

boxes_filt = load_image_to_sam(image_array, boxes_filt)
masks = segmenter.get_sam_box_results(image_array, boxes_filt) # show_each显示masks
masks = combine_masks(image_array, masks) # filter the big one
sam_time = time.time()
# segmenter.show_masks(masks) # choose_show显示masks
# grasp_stack = process_image(image_array, masks, show_points=True)  # show_points显示points
# grasp_stack, angles = rect_points(image_array, masks, show_contour=True)  # show_contour显示contour
get_rect_time = time.time()
grasp_stack, angles= get_rect(masks, image_array, show_points=True)
# print(angles)
end_time = time.time()

print(f"load_gd_time:{load_gd_time - time_start}s")
print(f"load_seg_time:{load_seg_time - load_gd_time}s")

print(f"gddino2_time:{gddino_time - start}s")
print(f"sam_time:{sam_time - gddino_time}s")

print(f"get_rect:{end_time - sam_time}s")
print(f"total_time:{end_time - start}s")