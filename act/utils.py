import numpy as np
import torch
import os
import h5py
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, min_episode_len, use_random_crop):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.min_episode_len = min_episode_len
        self.use_random_crop = use_random_crop
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        #为了对其数据，这个地方加了一个数据长度的最小值
        padded_action = padded_action[:self.min_episode_len]
        is_pad = is_pad[:self.min_episode_len]

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # 做一个随机裁剪，并且恢复回来原来的大小
        if self.use_random_crop:
            all_cam_images = random_crop_and_resize(all_cam_images)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # old
    # all_qpos_data = torch.stack(all_qpos_data)
    # all_action_data = torch.stack(all_action_data)
    # all_action_data = all_action_data
    #
    # # normalize action data
    # action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    # action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
    #
    # # normalize qpos data
    # qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    # qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    # qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    # new
    all_qpos_data = torch.concatenate(all_qpos_data, dim=0)
    all_action_data = torch.concatenate(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, min_episode_len, use_random_crop):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, min_episode_len, use_random_crop)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, min_episode_len, use_random_crop)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def count_files_starting_with(folder_path, prefix):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            count += 1
    return count


def random_crop_and_resize(images_array, crop_fraction=0.8):
    """
        Randomly crop an RGB image to a specified fraction of its original size,
        then resize it back to the original dimensions.

        :param images_array: NumPy array representing the RGB image with shape (height, width, 3) or (batch_size, height, width, 3)
        :param crop_fraction: Fraction of the image to crop (default is 0.8)
        :return: Resized RGB image with the same dimensions as the original
        """

    def random_crop_and_resize_one_image(image_array, crop_fraction):
        height, width, _ = image_array.shape

        # 计算裁剪尺寸
        crop_height = int(height * crop_fraction)
        crop_width = int(width * crop_fraction)

        # 随机选择裁剪的起始点并裁剪
        start_height = np.random.randint(0, height - crop_height)
        start_width = np.random.randint(0, width - crop_width)
        cropped_array = image_array[start_height:start_height + crop_height, start_width:start_width + crop_width]

        # 使用Pillow缩放图像到原始大小
        cropped_image = Image.fromarray(cropped_array)
        resized_image = cropped_image.resize((width, height), Image.NEAREST)

        # 返回numpy
        return np.array(resized_image)

    img_shape_len = len(images_array.shape)
    if img_shape_len == 3:
        return random_crop_and_resize_one_image(images_array, crop_fraction)
    elif img_shape_len == 4:
        rtn = []
        for image in images_array:
            rtn.append(random_crop_and_resize_one_image(image, crop_fraction))
        return np.array(rtn)
    else:
        raise Exception(f"Image array shape error: now shape is {images_array.shape}")
