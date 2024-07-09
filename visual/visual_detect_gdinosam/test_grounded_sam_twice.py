from GroundingDino import GroundingDino
from SegmentAnything import SegmentAnything
from ImageProcessV2 import *
import matplotlib.pyplot as plt
import re
import time 

image = Image.open("/home/embodied/Pictures/origin/CNC/test1.jpg")
image_array = np.array(image)

gddino = GroundingDino()
segmenter = SegmentAnything()   

start = time.time()
boxes_filt, pred_phrases = gddino.get_grounding_output(image_array, text_prompt="black_box")
gddino1_time = time.time()

highest_score = 0
for index, item in enumerate(pred_phrases):
    # 使用正则表达式查找括号内的数字
    match = re.search(r'\((\d+(\.\d+)?)\)', item)
    if match:
        # 将匹配的数字转换为浮点数
        number = float(match.group(1))
        # 更新最大值和索引
        if number > highest_score:
            highest_score = number
            max_index = index

# print(f"最大数字是 {highest_score}，对应的索引是 {max_index}")
print(boxes_filt[max_index])
box = boxes_filt[max_index]
H, W, _ =image_array.shape
box = box * torch.Tensor([W, H, W, H])

# from xywh to xyxy
box[:2] -= box[2:] / 2
box[2:] += box[:2]
# draw
x1, y1, x2, y2 = box
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
image_cropped = image_array[y1:y2, x1:x2]
# masks = segmenter.get_sam_box_results(boxes_filt[max_index][None,:])
# for mask in masks.cpu().numpy():
#     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     plt.gca().imshow(mask_image)
#     plt.show()
# cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
# cv2.imshow("crop", image_cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

boxes_filt, pred_phrases = gddino.get_grounding_output(image_cropped, text_prompt = "metal_block")
gddino2_time = time.time()
# boxes_filt, pred_phrases = gddino.get_grounding_output(image_array)

masks = segmenter.get_sam_box_results(image_cropped, boxes_filt)
sam_time = time.time()
print(masks.shape)
# for mask in masks.cpu().numpy():
#     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     plt.gca().imshow(mask_image)
#     plt.show()
masks = combine_masks(image_cropped, masks)
# count = np.count_nonzero(masks > 0)
# print(count)
# cv2.namedWindow("masks", cv2.WINDOW_NORMAL)
# cv2.imshow("masks", masks*255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# grasp_points = process_image(image_cropped, masks)
end_time = time.time()
print(f"gddino1_time:{gddino1_time - start}s")
print(f"gddino2_time:{gddino2_time - gddino1_time}s")
print(f"sam_time:{sam_time - gddino2_time}")
print(f"total_time:{end_time - start}s")
