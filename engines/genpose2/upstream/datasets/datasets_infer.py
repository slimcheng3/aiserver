import copy

import cv2
import numpy as np
import torch
from cutoop.data_types import CameraIntrinsicsBase
from cutoop.image_meta import ImageMetaData, ViewInfo
from cutoop.transform import pixel2xyz

from datasets.pcl_ops import depth_to_pcl, get_bbox, rgb_transform, sample_points
from utils.datasets_utils import aug_bbox_eval, crop_resize_by_warp_affine, get_2d_coord_np


class InferDataset:
    def __init__(self, data: dict, img_size: int = 224, device="cuda", n_pts=1024):
        self._depth = data["depth"]
        self._color = data["color"]
        self._mask = data["mask"]
        if isinstance(data["meta"], dict):
            camera_intrinsics = data["meta"]["camera"]["intrinsics"]
            camera_intrinsics = CameraIntrinsicsBase(
                fx=camera_intrinsics["fx"],
                fy=camera_intrinsics["fy"],
                cx=camera_intrinsics["cx"],
                cy=camera_intrinsics["cy"],
                width=camera_intrinsics["width"],
                height=camera_intrinsics["height"],
            )
            camera = ViewInfo(
                quaternion=(1.0, 0.0, 0.0, 0.0),
                translation=(0.0, 0.0, 0.0),
                intrinsics=camera_intrinsics,
                scene_obj_path="",
                background_image_path="",
                background_depth_path="",
                distances=[],
                kind="",
            )
            self._meta = ImageMetaData(
                objects=[],
                camera=camera,
                scene_dataset="",
                env_param={},
                face_up=False,
                concentrated=False,
                comments="",
                runtime_seed=0,
                baseline_dis=0,
                emitter_dist_l=0,
            )
        else:
            self._meta = data["meta"]
        self._img_size = img_size
        self._device = device
        self._n_pts = n_pts

    def get_per_object(self, obj_idx):
        object_mask = np.equal(self._mask, obj_idx)
        if not object_mask.any():
            raise ValueError(f"mask 里没有实例 {obj_idx}")
        self._depth[self._depth > 4.0] = 0
        if not (self._mask.shape[:2] == self._depth.shape[:2] == self._color.shape[:2]):
            raise ValueError("depth、mask、rgb 尺寸要一致")
        intrinsics = self._meta.camera.intrinsics
        intrinsic_matrix = np.array(
            [
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
        img_width, img_height = self._color.shape[1], self._color.shape[0]
        intrinsic_matrix[0] *= img_width / intrinsics.width
        intrinsic_matrix[1] *= img_height / intrinsics.height
        coord_2d = get_2d_coord_np(img_width, img_height).transpose(1, 2, 0)
        ys, xs = np.argwhere(object_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = get_bbox([np.min(ys), np.min(xs), np.max(ys), np.max(xs)], img_height, img_width)
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_eval(bbox_xyxy, img_height, img_width)
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        roi_rgb_ = crop_resize_by_warp_affine(self._color, bbox_center, scale, self._img_size, interpolation=cv2.INTER_LINEAR)
        roi_rgb = rgb_transform(roi_rgb_)
        mask_target = self._mask.copy().astype(np.float32)
        mask_target[self._mask != obj_idx] = 0.0
        mask_target[self._mask == obj_idx] = 1.0
        roi_mask = crop_resize_by_warp_affine(mask_target, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST)
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(self._depth, bbox_center, scale, self._img_size, interpolation=cv2.INTER_NEAREST)
        roi_depth = np.expand_dims(roi_depth, axis=0)
        depth_valid = roi_depth > 0
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            raise ValueError("裁剪后没有有效深度")
        valid = (np.squeeze(roi_depth, axis=0) > 0) * (np.squeeze(roi_mask, axis=0) > 0)
        xs, ys = np.argwhere(valid).transpose(1, 0)
        valid = valid.reshape(-1)
        pcl_in = depth_to_pcl(roi_depth, intrinsic_matrix, roi_coord_2d, valid)
        if len(pcl_in) < 10:
            raise ValueError(f"点数不够做位姿估计：{len(pcl_in)}")
        ids, pcl_in = sample_points(pcl_in, self._n_pts)
        xs, ys = xs[ids], ys[ids]
        return {
            "pcl_in": torch.as_tensor(pcl_in.astype(np.float32)).contiguous(),
            "roi_rgb": torch.as_tensor(np.ascontiguousarray(roi_rgb), dtype=torch.float32).contiguous(),
            "roi_rgb_": torch.as_tensor(np.ascontiguousarray(roi_rgb_), dtype=torch.uint8).contiguous(),
            "roi_xs": torch.as_tensor(np.ascontiguousarray(xs), dtype=torch.int64).contiguous(),
            "roi_ys": torch.as_tensor(np.ascontiguousarray(ys), dtype=torch.int64).contiguous(),
            "roi_center_dir": torch.as_tensor(
                pixel2xyz(img_height, img_height, bbox_center, intrinsics), dtype=torch.float32
            ).contiguous(),
        }

    def get_objects(self):
        obj_idx = np.unique(self._mask)
        obj_idx = obj_idx[(obj_idx != 0) & (obj_idx != 255)]
        if obj_idx.size == 0:
            raise ValueError("mask 里没有前景实例")
        objects = {}
        for idx in obj_idx:
            obj = self.get_per_object(idx)
            for key, value in obj.items():
                objects.setdefault(key, []).append(value)
        for key, value in objects.items():
            objects[key] = torch.stack(value, dim=0)
        data = {
            "pts": objects["pcl_in"].to(self._device),
            "pts_color": objects["pcl_in"].to(self._device),
            "roi_rgb": objects["roi_rgb"].to(self._device),
            "roi_xs": objects["roi_xs"].to(self._device),
            "roi_ys": objects["roi_ys"].to(self._device),
            "roi_center_dir": objects["roi_center_dir"].to(self._device),
        }
        num_pts = data["pts"].shape[1]
        zero_mean = torch.mean(data["pts"][:, :, :3], dim=1)
        data["zero_mean_pts"] = copy.deepcopy(data["pts"])
        data["zero_mean_pts"][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
        data["pts_center"] = zero_mean
        return data

    @property
    def color(self):
        return self._color
