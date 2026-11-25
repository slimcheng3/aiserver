import numpy as np


def sample_points(pcl, n_pts):
    total_pts_num = pcl.shape[0]
    if total_pts_num < n_pts:
        pcl = np.concatenate(
            [np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[: n_pts % total_pts_num]],
            axis=0,
        )
        ids = np.concatenate(
            [np.tile(np.arange(total_pts_num), n_pts // total_pts_num), np.arange(n_pts % total_pts_num)],
            axis=0,
        )
    else:
        ids = np.random.permutation(total_pts_num)[:n_pts]
        pcl = pcl[ids]
    return ids, pcl


def depth_to_pcl(depth, K, xymap, valid):
    K = K.reshape(-1)
    cx, cy, fx, fy = K[2], K[5], K[0], K[4]
    depth = depth.reshape(-1).astype(np.float32)[valid]
    x_map = xymap[0].reshape(-1)[valid]
    y_map = xymap[1].reshape(-1)[valid]
    real_x = (x_map - cx) * depth / fx
    real_y = (y_map - cy) * depth / fy
    pcl = np.stack((real_x, real_y, depth), axis=-1)
    return pcl.astype(np.float32)


def rgb_transform(rgb):
    rgb_ = np.transpose(rgb, (2, 0, 1)) / 255
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for i in range(3):
        rgb_[i, :, :] = (rgb_[i, :, :] - mean[i]) / std[i]
    return rgb_


def get_bbox(bbox, img_height=480, img_length=640):
    y1, x1, y2, x2 = bbox
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, img_height - 40, img_length - 40)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        rmax += -rmin
        rmin = 0
    if cmin < 0:
        cmax += -cmin
        cmin = 0
    if rmax > img_height:
        rmin -= rmax - img_height
        rmax = img_height
    if cmax > img_length:
        cmin -= cmax - img_length
        cmax = img_length
    return rmin, rmax, cmin, cmax
