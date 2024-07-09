import math
import itertools
from functools import partial
import torch
from torchvision import transforms
import mmcv
from mmcv.runner import load_checkpoint
import urllib
import urllib.request
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from Dinov2.dinov2.utils.config import get_cfg
from Dinov2.dinov2.utils.utils import load_pretrained_weights
from Dinov2.dinov2.models import build_model_from_cfg
from Dinov2.dinov2.eval.depth.models import build_depther


class DinoV2:
    def __init__(self, 
                 model_name = "vits14",
                 base_path = "visual/visual_detect_gdinosam/"):
        self.dinov2_config_file =  base_path + f"Dinov2/dinov2/configs/eval/{model_name}_pretrain.yaml"
        self.dinov2_model_name = model_name
        self.dinov2_checkpoints = base_path + f'weights/dinov2_{model_name}_pretrain.pth'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dinov2_cfg = get_cfg(self.dinov2_config_file)
        self.dinov2_model = self.get_dinov2_model().to(self.device)
    
    def get_dinov2_model(self):
        model, _, embed_dim = build_model_from_cfg(self.dinov2_cfg, only_teacher=False)
        load_pretrained_weights(model, self.dinov2_checkpoints, checkpoint_key="student")
        model.eval()
        return model

    def load_image_to_dinov2(self, image: np.ndarray, center_crop = False) -> torch.tensor:
        if center_crop:
            prep = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.CenterCrop((196,196)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            prep = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        input_tensor = prep(image)[None, ...]
        input_tensor = input_tensor.cuda()
        return input_tensor

    def get_feats(self, image: np.ndarray):
        image_torch = self.load_image_to_dinov2(image, center_crop=True)
        out = self.dinov2_model(image_torch, is_training=True )
        cls_token = out['x_norm_clstoken']
        # norm_cls_token = torch.nn.functional.normalize(cls_token)
        return cls_token

############# Depth_Estimator ##############

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

class DepthEstimator(DinoV2):
    def __init__(self, head_dataset = "nyu",head_type="dpt"):
        super().__init__()
        self.backbone_model = self.dinov2_model
        self.head_type = head_type
        self.head_dataset = head_dataset
        self.cfg = self.get_cfg_str() 
        self.dpt_ckpt = f"visual/visual_detect_gdinosam/weights/{self.dinov2_model_name}_{self.head_dataset}_{self.head_type}_head.pth"
        self.model = self._create_depther()

    def get_cfg_str(self):
        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        head_config_url = f"{DINOV2_BASE_URL}/{self.dinov2_model_name}/{self.dinov2_model_name}_{self.head_dataset}_{self.head_type}_config.py"
        with urllib.request.urlopen(head_config_url) as f:
            cfg_str =  f.read().decode()
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        return cfg
    def _create_depther(self):
        train_cfg = self.cfg.get("train_cfg")
        test_cfg = self.cfg.get("test_cfg")
        depther = build_depther(self.cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

        depther.backbone.forward = partial(
            self.backbone_model.get_intermediate_layers,
            n=self.cfg.model.backbone.out_indices,
            reshape=True,
            return_class_token=self.cfg.model.backbone.output_cls_token,
            norm=self.cfg.model.backbone.final_norm,
        )

        if hasattr(self.backbone_model, "patch_size"):
            depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(self.backbone_model.patch_size)(x[0]))
        load_checkpoint(depther, self.dpt_ckpt , map_location="cuda")
        depther.cuda()
        depther.eval()
        return depther

    def load_url_image(self, image_url):
        with urllib.request.urlopen(image_url) as f:
            image = Image.open(f).convert("RGB")
        return image
    
    def load_local_image(self, image_path):
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert("RGB")
        return image

    def make_depth_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

    def estimate_depth(self, image_path, scale_factor=1):
        image = self.load_local_image(image_path)
        transform = self.make_depth_transform()
        rescaled_image = image.resize((scale_factor * (image.width // 14) *14 , scale_factor * (image.height// 14) *14 ))
        transformed_image = transform(rescaled_image)
        batch = transformed_image.unsqueeze(0).cuda()  # Make a batch of one image

        with torch.inference_mode():
            result = self.model.whole_inference(batch, img_meta=None, rescale=True)

        depth_image = self.render_depth(result.squeeze().cpu())
        return depth_image

    def render_depth(self, values, colormap_name="magma_r"):
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)

        import matplotlib.pyplot as plt
        colormap = plt.get_cmap(colormap_name)
        colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
        colors = colors[:, :, :3]  # Discard alpha component
        return Image.fromarray(colors)

# Example usage:
# depth_estimator = DepthEstimator()
# depth_image = depth_estimator.estimate_depth("visual/visual_detect_gdinosam/material/1.jpg")
# depth_image.show()


        



