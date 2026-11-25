import gc
import random

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from configs.config import get_config
from datasets.datasets_infer import InferDataset
from networks.posenet_agent import PoseNet
from networks.reward import sort_poses_by_energy
from utils.metrics import get_rot_matrix
from utils.misc import average_quaternion_batch, get_pose_dim, get_pose_representation
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix


class GenPose2:
    def __init__(self, score_model_path: str, energy_model_path: str, scale_model_path: str):
        self.cfg = self._get_config(score_model_path, energy_model_path, scale_model_path)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.cfg.agent_type = "score"
        self.score_agent = PoseNet(self.cfg)
        self.score_agent.load_ckpt(model_dir=self.cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        self.score_agent.eval()
        self.cfg.agent_type = "energy"
        self.energy_agent = PoseNet(self.cfg)
        self.energy_agent.load_ckpt(model_dir=self.cfg.pretrained_energy_model_path, model_path=True, load_model_only=True)
        self.energy_agent.eval()
        self.cfg.agent_type = "scale"
        self.scale_agent = PoseNet(self.cfg)
        self.scale_agent.load_ckpt(model_dir=self.cfg.pretrained_scale_model_path, model_path=True, load_model_only=True)
        self.scale_agent.eval()

    def _get_config(self, score_model_path: str, energy_model_path: str, scale_model_path: str):
        cfg = get_config()
        cfg.pretrained_score_model_path = score_model_path
        cfg.pretrained_energy_model_path = energy_model_path
        cfg.pretrained_scale_model_path = scale_model_path
        cfg.sampler_mode = ["ode"]
        cfg.T0 = 0.55
        cfg.seed = 0
        cfg.eval_repeat_num = 50
        cfg.dino = "pointwise"
        return cfg

    def _inference_score(self, data: InferDataset, score_agent: PoseNet, prev_pose=None, T0=None):
        all_pred_pose = []
        all_score_feature = []
        for i, batch_sample in enumerate([data.get_objects()]):
            if prev_pose is not None:
                _prev_pose = prev_pose[i].clone()
                _prev_pose[:, -3:] -= batch_sample["pts_center"]
            else:
                _prev_pose = None
            pred_pose, _ = score_agent.pred_func(
                data=batch_sample,
                repeat_num=self.cfg.eval_repeat_num,
                T0=self.cfg.T0 if T0 is None else T0,
                init_x=None if prev_pose is None else _prev_pose,
                return_average_res=False,
                return_process=False,
            )
            all_pred_pose.append(pred_pose)
            all_score_feature.append(
                {
                    "pts_feat": batch_sample["pts_feat"].cpu(),
                    "rgb_feat": None if batch_sample["rgb_feat"] is None else batch_sample["rgb_feat"].cpu(),
                }
            )
        return all_pred_pose, all_score_feature

    def _inference_energy(self, data: InferDataset, energy_agent: PoseNet, all_pred_pose):
        all_pred_energy = []
        for i, batch_sample in enumerate([data.get_objects()]):
            pred_energy = energy_agent.get_energy(
                data=batch_sample,
                pose_samples=all_pred_pose[i],
                T=None,
                mode="test",
                extract_feature=True,
            )
            all_pred_energy.append(pred_energy.cpu())
        return all_pred_energy

    def _aggregate_pose(self, all_pred_pose, all_pred_energy):
        if all_pred_energy is None:
            all_pred_energy = [torch.ones(*(all_pred_pose[i].shape[:2]), 2) for i in range(len(all_pred_pose))]
        all_aggregated_pose = []
        for pred_pose, pred_energy in zip(all_pred_pose, all_pred_energy):
            sorted_pose, _sorted_energy = sort_poses_by_energy(pred_pose, pred_energy)
            bs = pred_pose.shape[0]
            retain_num = int(self.cfg.eval_repeat_num * self.cfg.retain_ratio)
            good_pose = sorted_pose[:, :retain_num, :]
            rot_matrix = get_rot_matrix(good_pose[:, :, :-3].reshape(bs * retain_num, -1), self.cfg.pose_mode)
            quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, retain_num, -1)
            aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz)
            if self.cfg.clustering:
                for j in range(bs):
                    pairwise_distance = 1 - torch.sum(quat_wxyz[j].unsqueeze(0) * quat_wxyz[j].unsqueeze(1), dim=2) ** 2
                    labels = DBSCAN(
                        eps=self.cfg.clustering_eps,
                        min_samples=int(self.cfg.clustering_minpts * retain_num),
                    ).fit(pairwise_distance.cpu().numpy()).labels_
                    if np.any(labels >= 0):
                        bins = np.bincount(labels[labels >= 0])
                        best_label = np.argmax(bins)
                        aggregated_quat_wxyz[j] = average_quaternion_batch(quat_wxyz[j, labels == best_label].unsqueeze(0))[0]
            aggregated_trans = torch.mean(good_pose[:, :, -3:], dim=1)
            aggregated_pose = torch.zeros(bs, 4, 4)
            aggregated_pose[:, 3, 3] = 1
            aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz)
            aggregated_pose[:, :3, 3] = aggregated_trans
            all_aggregated_pose.append(aggregated_pose)
            gc.collect()
        return all_aggregated_pose

    def _inference_scale(self, data: InferDataset, scale_agent: PoseNet, all_score_feature, all_aggregated_pose):
        all_final_pose = []
        all_final_length = []
        for i, batch_sample in enumerate([data.get_objects()]):
            batch_sample.update(
                {key: (None if value is None else value.to(self.cfg.device)) for key, value in all_score_feature[i].items()}
            )
            batch_sample["axes"] = all_aggregated_pose[i][:, :3, :3].to(self.cfg.device)
            cal_mat, length = scale_agent.pred_scale_func(batch_sample)
            final_pose = all_aggregated_pose[i].clone()
            final_pose[:, :3, :3] = cal_mat.cpu()
            all_final_pose.append(final_pose.cpu())
            all_final_length.append(length.cpu())
        return all_final_pose, all_final_length

    def inference(self, data: InferDataset, prev_pose=None, tracking=False, tracking_T0=0.15):
        prev_pose_genpose2 = None
        if prev_pose is not None:
            prev_pose_genpose2 = []
            for item in prev_pose:
                pose_genpose2 = torch.zeros(item.shape[0], get_pose_dim(self.cfg.pose_mode), device=self.cfg.device)
                pose_genpose2[:, :-3] = get_pose_representation(item[:, :3, :3], self.cfg.pose_mode)
                pose_genpose2[:, -3:] = item[:, :3, 3]
                prev_pose_genpose2.append(pose_genpose2)
        infer_T0 = tracking_T0 if tracking and prev_pose is not None else None
        all_pred_pose, all_score_feature = self._inference_score(data, self.score_agent, prev_pose_genpose2, infer_T0)
        if self.cfg.pretrained_energy_model_path is not None:
            all_pred_energy = self._inference_energy(data, self.energy_agent, all_pred_pose)
            all_aggregated_pose = self._aggregate_pose(all_pred_pose, all_pred_energy)
        else:
            all_aggregated_pose = self._aggregate_pose(all_pred_pose, None)
        return self._inference_scale(data, self.scale_agent, all_score_feature, all_aggregated_pose)


def create_genpose2(score_model_path: str, energy_model_path: str, scale_model_path: str):
    return GenPose2(score_model_path, energy_model_path, scale_model_path)
