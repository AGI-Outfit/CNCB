import json
import time

import math
import os

import cv2
import torch
import pickle
import numpy as np
from act.utils import random_crop_and_resize
from act.policy import ACTPolicy


class Inference(object):
    """
    :param ckpt_dir:保存训练结果的文件夹地址
    """

    def __init__(self, ckpt_dir):
        # ckpt_dir = './data'
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = f'policy_best.ckpt'
        self.ckpt_path = os.path.join(ckpt_dir, self.ckpt_name)
        self.now_image = None
        self.last1_image = None
        self.last2_image = None
        self.last3_image = None
        self.action_history = []

        self.policy_config = self.__get_inference_data()

        self.use_random_crop = self.policy_config['use_random_crop']
        self.use_history = False
        if 'hk_image_1' in self.policy_config['camera_names']:
            self.use_history = True


        #load policy
        self.policy = ACTPolicy(self.policy_config)

        loading_status = self.policy.load_state_dict(torch.load(self.ckpt_path))
        print(loading_status)

        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {self.ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    def __get_image_from_history(self, image):
        if self.now_image is None:
            self.now_image = np.copy(image)
            self.last1_image = np.copy(image)
            self.last2_image = np.copy(image)
            self.last3_image = np.copy(image)
            return np.stack([self.now_image, self.last1_image, self.last2_image, self.last3_image])

        if not np.array_equal(self.now_image, image):
            self.last3_image = np.copy(self.last2_image)
            self.last2_image = np.copy(self.last1_image)
            self.last1_image = np.copy(self.now_image)
            self.now_image = np.copy(image)

        return np.stack([self.now_image, self.last1_image, self.last2_image, self.last3_image])

    def __get_inference_data(self):
        inference_path = os.path.join(self.ckpt_dir, f'inference.json')
        inference_data = json.load(open(inference_path, 'r'))
        inference_data['policy_config']['is_inference_mode'] = True
        return inference_data['policy_config']

    def __call__(self, qpos, hk_image, ob_image=None, time_prefcounter=None):
        """
        :param qpos: qpos numpy array, shape [6,], type: np.float32
        :param hk_image: image numpy array, shape:[480, 640, 3], type: np.uint8
        :param ob_image: image numpy array, shape:[720, 1280, 3], type: np.uint8
        :return: action numpy array, shape[6,], type.float32
        """
        qpos = self.pre_process(qpos.astype(np.float32))
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = qpos_data.unsqueeze(0)

        hk_image = hk_image.astype(np.uint8)
        if self.use_history:
            all_hk_image = self.__get_image_from_history(hk_image)
            if self.use_random_crop:
                all_hk_image = random_crop_and_resize(all_hk_image)
            hk_image_torch = torch.from_numpy(all_hk_image)
        else:
            if self.use_random_crop:
                hk_image = random_crop_and_resize(hk_image)
            hk_image_torch = torch.from_numpy(hk_image)
            hk_image_torch = hk_image_torch.unsqueeze(0)
        image_data = hk_image_torch

        # ob_image 传入的图像需要裁剪成480*640*3大小
        if ob_image is not None:
            ob_image = ob_image.astype(np.uint8)
            cropped_image = ob_image[:, 160:1120, :]  # 裁剪为720*960*3的大小
            ob_image = cv2.resize(cropped_image, (640, 480))  # 缩放
            if self.use_random_crop:
                ob_image = random_crop_and_resize(ob_image) #随机裁剪
            ob_image_torch = torch.from_numpy(ob_image)
            ob_image_torch = ob_image_torch.unsqueeze(0)
            image_data = torch.cat((image_data, ob_image_torch), axis=0)

        # channel last
        image_data = torch.einsum('k h w c ->k c h w', image_data)

        image_data = image_data.unsqueeze(0)  # 1, 1, 3, 480, 640

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        qpos_data = qpos_data.cuda()
        image_data = image_data.cuda()

        action = self.policy(qpos_data, image_data)  # no action, sample from prior
        action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        action = self.post_process(action)

        if time_prefcounter is None:
            return action[0]
        else:
            self.action_history.append((action, time_prefcounter))
            if len(self.action_history) >= 30:
                self.action_history.pop(0)
            now_time = time_prefcounter
            # 先远后近
            action_list = []
            for ac, record_time in self.action_history:
                delta_ms = (now_time-record_time)*1000
                if delta_ms <= 20:
                    action_list.append(ac[0])
                elif delta_ms <= 1000:
                    index = delta_ms/40.0
                    left = int(math.floor(index))
                    right = int(math.ceil(index))
                    a = ac[left]*(index-left) + ac[right]*(right-index)
                    action_list.append(a)
                else:
                    continue

            if len(action_list) == 1:
                return action_list[0]
            else:
                rtn = 0
                for i in action_list:
                    param = 0.90
                    rtn = param*i + (1-param)*rtn

                return rtn


def main():
    inference = Inference(ckpt_dir='/home/HDD1/Share/act-checkpoint/hk_ob/sim_step8_q30')
    inference = Inference(ckpt_dir='/home/HDD1/Share/act-checkpoint/hk/sim_step8_q30')
    qpos = np.random.randn(6)
    hk_image = np.random.randint(0, 255, (480, 640, 3))
    ob_image = np.random.randint(0, 255, (720, 1280, 3))
    ob_image = None
    with torch.inference_mode():
        action = inference(qpos, hk_image, ob_image, time_prefcounter=time.perf_counter())
    print(action)


if __name__ == '__main__':
    main()
