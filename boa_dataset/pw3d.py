"""
Mix the video from different datasets
"""

import os
import os.path as osp
import cv2
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import constants
import config
from utils.dataprocess import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, split_into_chunks


def key_3dpw(elem):
    elem = os.path.basename(elem)
    vid = elem.split('_')[1]
    pid = elem.split('_')[2][:-4]
    return int(vid) * 10 + int(pid)


class PW3D(Dataset):
    def __init__(self, options):
        super(PW3D, self).__init__()
        self.options = options  # 包含seqlen=16
        # overlap = 0.75
        overlap = 0.9
        self.set = 'train'
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.stride = int(self.options.seqlen * (1 - overlap))

        # 3DPW
        self.pw3d_img_dir = config.PW3D_ROOT
        # self.pw3d_datas = glob.glob('/home/neu307/liumeilin/boa_data/data_extras/3dpw_[0-9]*_[0-9].npz')
        self.pw3d_datas = glob.glob('/home/neu307/liumeilin/boa_data/data_extras/3dpw_[6]_[0].npz')
        self.pw3d_datas.sort(key=key_3dpw)
        print(f'npz for pw3d: {len(self.pw3d_datas)}')  # 37个npz文件

        # parse data
        self.datas = []
        self.imgnames = []
        self.scales = []
        self.centers = []
        self.smpl_j2ds = []
        self.genders = []
        self.dataset_names = []
        # for 3dhp
        self.gt_j3ds = []
        # for 3dpw
        self.pose, self.betas, self.smpl_j2ds, self.op_j2ds = [], [], [], []
        # 将npz文件读取并存储相关参数
        for i in range(len(self.pw3d_datas)):
            self.datas.append(self.pw3d_datas[i])
            imgnames, scales, centers, pose, betas, smpl_j2ds, op_j2ds, genders, gt_j3ds, dataset_name = self.parse_3dpw(
                self.pw3d_datas[i])
            # 一共37个文件夹，每个文件夹里的视频帧有序号但不是完全连续的，有一些断帧
            self.imgnames.append(imgnames)
            self.scales.append(scales)
            self.centers.append(centers)
            self.smpl_j2ds.append(smpl_j2ds)
            self.genders.append(genders)
            self.gt_j3ds.append(gt_j3ds)
            self.pose.append(pose)
            self.betas.append(betas)
            self.op_j2ds.append(op_j2ds)
            self.dataset_names.append(dataset_name)
        assert len(self.datas) == len(self.pw3d_datas)
        # record it 
        with open(osp.join(self.options.expdir, self.options.expname, 'seq_order.record'), 'w') as f:
            for data_name in self.datas:
                f.write(data_name + '\n')
        # 直接把37个文件夹里的所有视频帧连结在一起，一共35515帧--存在文件夹之间视频终端/不连续的情况?--可以改进
        self.imgnames = np.concatenate(self.imgnames, axis=0)
        self.scales = np.concatenate(self.scales, axis=0)
        self.centers = np.concatenate(self.centers, axis=0)
        self.smpl_j2ds = np.concatenate(self.smpl_j2ds, axis=0)
        self.genders = np.concatenate(self.genders, axis=0)
        self.gt_j3ds = np.concatenate(self.gt_j3ds, axis=0)
        self.pose = np.concatenate(self.pose, axis=0)
        self.betas = np.concatenate(self.betas, axis=0)
        self.op_j2ds = np.concatenate(self.op_j2ds, axis=0)
        self.dataset_names = np.concatenate(self.dataset_names, axis=0)
        self.vid_indices = split_into_chunks(self.imgnames, self.options.seqlen, self.stride)

    def __len__(self):
        return self.scales.shape[0]

    def __getitem__(self, index):
        start_index, end_index = self.vid_indices[index]  # 时序16步长16时，有655个数组

        is_train = self.set == 'train'

        scale = self.scales[start_index:end_index + 1].copy()
        center = self.centers[start_index:end_index + 1].copy()
        op_j2d = self.op_j2ds[start_index:end_index + 1].copy()  # 387,49,3
        theta = self.pose[start_index:end_index + 1].copy() # 387,72
        beta = self.betas[start_index:end_index + 1].copy() # 387,10
        imgname = self.imgnames[start_index:end_index + 1].copy()
        smpl_j2d = self.smpl_j2ds[start_index:end_index + 1].copy() # 387,49,3
        gender = self.genders[start_index:end_index + 1].copy()  # 387,
        dataset_name = self.dataset_names[start_index:end_index + 1].copy()  # 387
        j3d = self.gt_j3ds[start_index:end_index + 1].copy()  # 387,24,4

        op_j2d_tensor = np.zeros((self.options.seqlen, 49,3), dtype=np.float16)
        image_tensor = np.zeros((self.options.seqlen, 3,224,224), dtype=np.float16)
        theta_tensor = np.zeros((self.options.seqlen, 72), dtype=np.float16)
        beta_tensor = np.zeros((self.options.seqlen, 10), dtype=np.float16)
        smpl_j2d_tensor = np.zeros((self.options.seqlen, 49,3), dtype=np.float16)
        bbox_tensor = np.zeros((self.options.seqlen, 3), dtype=np.float16)

        # 读取图片
        # 标签均来自npz文件,而image数据从官网下载的数据集中取,存在不对应的情况,需进行处理
        for idx in range(self.options.seqlen):
            # image = self.read_image(imgname, idx)
            imgnamepath = os.path.join(self.pw3d_img_dir, imgname[idx])
            # image[:,:,::-1]的作用是对颜色通道把RGB转换成BGR
            image = cv2.imread(imgnamepath)[:, :, ::-1].copy().astype(np.float32)

            # ori image, no aug
            flip = 0  # flipping
            pn = np.ones(3)  # per channel pixel-noise
            rot = 0  # rotation
            sc = 1  # scaling
            op_j2d[idx], image, theta[idx], beta[idx], smpl_j2d[idx] = self.process_sample(image.copy(),
                                                                                           theta[idx],
                                                                                           beta[idx],
                                                                                           op_j2d[idx],
                                                                                           smpl_j2d[idx],
                                                                                           center[idx],
                                                                                           scale[idx],
                                                                                           flip, pn, rot, sc,
                                                                                           is_train=False)
            op_j2d_tensor[idx] = op_j2d[idx]
            image_tensor[idx] = image
            theta_tensor[idx] = theta[idx]
            beta_tensor[idx] = beta[idx]
            smpl_j2d_tensor[idx] = smpl_j2d[idx]
            bbox_tensor[idx] = np.stack([center[idx][0], center[idx][1], scale[idx] * 200])

        # item = {
        #     'op_j2d': op_j2d_tensor,
        #     'image': image_tensor,
        #     'pose': theta_tensor,
        #     'betas': beta_tensor,
        #     'smpl_j2d': smpl_j2d_tensor,
        #     'gender': gender,
        #     'imgname': imgname,
        #     'dataset_name': dataset_name,
        #     'j3d': j3d,
        #     'bbox': bbox_tensor,
        # }
        item = {
            # 'op_j2d': torch.from_numpy(op_j2d_tensor).float(),
            'op_j2d': op_j2d_tensor,
            'image': image_tensor,
            'pose': theta_tensor,
            'betas': beta_tensor,
            'smpl_j2d': smpl_j2d_tensor,
            'gender': gender,
            # 'imgname': imgname.ToTensor(),
            # 'imgname': imgname,
            # 'dataset_name': dataset_name,
            'j3d': j3d,
            'bbox': bbox_tensor,
        }

        return item

    def process_sample(self, image, pose, beta, keypoints, smpl_j2ds, center, scale, flip, pn, rot, sc, is_train):
        # labeled keypoints
        kp2d = torch.from_numpy(
            self.j2d_processing(keypoints, center, sc * scale, rot, flip, is_train=is_train)).float()
        smpl_j2ds = torch.from_numpy(
            self.j2d_processing(smpl_j2ds, center, sc * scale, rot, flip, is_train=is_train)).float()
        img = self.rgb_processing(image, center, sc * scale, rot, flip, pn, is_train=is_train)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)  # 3,224,224
        # img = torch.cat((img, seqlen), dim=0).to(self.device)
        pose = torch.from_numpy(self.pose_processing(pose, rot, flip, is_train=is_train)).float()
        betas = torch.from_numpy(beta).float()
        return kp2d, img, pose, betas, smpl_j2ds

    def read_image(self, imgname, index):
        # 读入所有图片路径
        if self.dataset_names[index] == '3dpw':
            imgname = os.path.join(self.pw3d_img_dir, imgname)
        elif self.dataset_names[index] == '3dhp':
            imgname = os.path.join(self.hp3d_img_dir, imgname)
        img = cv2.imread(imgname)[:, :, ::-1].copy().astype(np.float32)
        return img

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, is_train):
        # 图像中心对齐，裁剪成224*224
        rgb_img = crop(rgb_img.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        if is_train and flip:
            rgb_img = flip_img(rgb_img)
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, is_train):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / constants.IMG_RES - 1.
        # flip the x coordinates
        if is_train and f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def pose_processing(self, pose, r, f, is_train):
        """Process SMPL theta parameters and apply all augmentation transforms."""
        if is_train:
            # rotation or the pose parameters
            pose[:3] = rot_aa(pose[:3], r)
            # flip the pose parameters
            if f:
                pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def parse_3dpw(self, dataname):
        data = np.load(dataname)
        imgnames = data['imgname']
        scales = data['scale']
        centers = data['center']
        pose = data['pose'].astype(np.float)
        betas = data['shape'].astype(np.float)
        smpl_j2ds = data['j2d']
        op_j2ds = data['op_j2d']
        # Get gender data, if available
        try:
            gender = data['gender']
            genders = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            genders = -1 * np.ones(len(imgnames)).astype(np.int32)
        gt_j3ds = np.zeros((scales.shape[0], 24, 4))
        dataset_name = ['3dpw'] * scales.shape[0]
        return imgnames, scales, centers, pose, betas, smpl_j2ds, op_j2ds, genders, gt_j3ds, dataset_name
