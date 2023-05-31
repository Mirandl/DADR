"""
Dynaboa
"""

import os
import errno
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import random
import joblib
import numpy as np
import os.path as osp
import learn2learn as l2l

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize

from utils.dataprocess import crop, transform, rot_aa
import config
import constants
from model import SMPL
from model.vibe import vibe
from utils.smplify.prior import MaxMixturePrior
from utils.geometry import batch_rodrigues, perspective_projection, rotation_matrix_to_angle_axis
from boa_dataset.pw3d import PW3D
from boa_dataset.internet_data import Internet_dataset
from render_demo import Renderer, convert_crop_cam_to_orig_img


class BaseAdaptor():
    def __init__(self, options):
        self.options = options
        self.exppath = osp.join(self.options.expdir, self.options.expname)
        os.makedirs(self.exppath + '/mesh', exist_ok=True)
        os.makedirs(self.exppath + '/image', exist_ok=True)
        os.makedirs(self.exppath + '/result', exist_ok=True)
        self.summary_writer = SummaryWriter(self.exppath)  # `SummaryWriter` 类提供了一个高级 API，用于在给定目录中创建事件文件，并向其中添加摘要和事件。
        # 该类异步更新文件内容。 这允许训练程序调用方法以直接从训练循环将数据添加到文件中，而不会减慢训练速度。
        self.device = torch.device('cuda')  # 表示将构建的张量或者模型分配到相应的设备上。
        # set seed
        self.seed_everything(self.options.seed)  # 22-设定生成随机数的种子，目的是为了让结果具有重复性，重现结果。

        self.options.mixtrain = self.options.lower_level_mixtrain or self.options.upper_level_mixtrain  # 1

        if self.options.retrieval:
            # # load basemodel's feature聚类中心
            self.load_h36_cluster_res()

        if self.options.retrieval:
            self.h36m_dataset = SourceDataset(datapath='/home/neu307/liumeilin/boa_data/retrieval_res'
                                                       '/h36m_random_sample_center_10_10.pt')

        # set model
        self.set_model_optim()
        if self.options.use_meanteacher:
            self.set_teacher()

        # set dataset
        self.set_dataloader()  # 在多线程时需要将dataloader封装起来 warp，否则会报错

        # set criterion
        self.set_criterion()

        self.setup_smpl()

    def get_h36m_data(self, indice):
        item_i = self.h36m_dataset[indice]
        return {k: v for k, v in item_i.items()}

    def load_h36_cluster_res(self, ):
        ########## 0.1
        # self.h36m_cluster_res = joblib.load('/home/neu307/liumeilin/boa_data/retrieval_res/cluster_res_random_sample_center_10_10_potocol2.pt')
        self.h36m_cluster_res = joblib.load('/home/neu307/liumeilin/boa_data/retrieval_res/cluster_res_random_sample_center_10_10_potocol2.pt')

        # self.hrnet_pretrain = torch.load('/home/neu307/liumeilin/boa_data/retrieval_res/hrnet_pretrain.pth')
        # self.hrnet_w32_conv_pare_mosh = torch.load('/home/neu307/liumeilin/boa_data/retrieval_res/hrnet_w32_conv_pare_mosh.pth')
        # self.hrnet_w32_conv_pare_coco = torch.load('/home/neu307/liumeilin/boa_data/retrieval_res/hrnet_w32_conv_pare_coco.pth')
        # self.hrnet_w32_conv_pare = torch.load('/home/neu307/liumeilin/boa_data/retrieval_res/hrnet_w32_conv_pare.pth')

        self.centers = self.h36m_cluster_res['centers']
        self.centers = torch.from_numpy(self.centers).float().to(self.device)
        self.index = self.h36m_cluster_res['index']  # 数据集中图片的index
        self.h36m_base_features = np.concatenate(
            joblib.load('/home/neu307/liumeilin/boa_data/retrieval_res/h36m_feats_random_sample_center_10_10.pt'), axis=0)  # (31176,2048)
        # 按中心排序的特征

    def move_dict_to_device(dict, device, tensor2float=False):
        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                if tensor2float:
                    dict[k] = v.float().to(device)
                else:
                    dict[k] = v.to(device)
    # 建议使用model.to(device)的方式，这样可以显示指定需要使用的计算资源，特别是有多个GPU的情况下。

    def retrieval(self, features, seqlen):  # 在源域找6张？最接近的作为3D关节点监督
        h36mdata_list = []
        for i in range(seqlen):
            feature = features[i, :].unsqueeze(0)  # 取当前一帧
            # 开始回溯
            dists = 1 - F.cosine_similarity(feature, self.centers)
            pos_cluster = torch.argsort(dists)[0].item()  # 返回排序后的值所对应的下标  24
            indices = self.index[pos_cluster]
            pos_indices = random.sample(indices, self.options.sample_num)
            for x in pos_indices:  # 28504,28518
                h36mdata_list.append(self.get_h36m_data(x))  # 找到seqlen组原始数据

        h36m_batch = h36mdata_list[0]
        if len(h36mdata_list) > 1:
            for h36m_dataitem in h36mdata_list[1:]:
                for k, v in h36m_dataitem.items():
                    # if k == 'oriimage':
                    #     h36m_batch[k] = np.concatenate([h36m_batch[k], v], axis=0)
                    if isinstance(v, torch.Tensor):
                        h36m_batch[k] = torch.cat([h36m_batch[k], v], dim=0)
        # 注意！这里只对之后需要用到的参数进行了seqlen的累加，其他参数还是第一组的数据
        h36m_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in h36m_batch.items()}

        return h36m_batch

    def seed_everything(self, seed):
        """
        ensure reproduction
        """
        random.seed(seed)
        os.environ['PYHTONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        print('---> seed has been set')

    def set_model_optim(self, ):
        """
        setup model and optimizer
        """
        # checkpoint_h = torch.load(self.options.model_file_hmr)  # 各层参数 data/basemodel.pt
        checkpoint_v = torch.load(self.options.model_file_vibe)
        checkpoint_h = torch.load(self.options.model_file_hmr)
        model = vibe(config.SMPL_MEAN_PARAMS)
        # model_dict = model.state_dict()
        if self.options.use_boa:
            # VIBE模型
            self.model = l2l.algorithms.MAML(model, lr=self.options.fastlr, first_order=True).to(self.device)
            # 引入HMR关于resnet50和regressor的checkpoint
            # checkpoint_h['model'] = {'module.'+k: v for k, v in checkpoint_h['model'].items()}
            checkpoint_h['model'] = {k.replace('module.', ''): v for k, v in checkpoint_h['model'].items()}
            model_dict = model.state_dict()
            # 引入VIBE关于GRU和regressor的checkpoint，注意，regressor部分和上面是重合的
            # 既然VIBE写在后面，认为真正使用的是VIBE的regressor训练参数
            # checkpoint_v['gen_state_dict'] = {'module.'+k : v for k, v in checkpoint_h['model'].items()}
            # checkpoint_v['gen_state_dict']={k.replace('regressor.', ''): v for k, v in checkpoint_v['gen_state_dict'].items()}
            # checkpoint_v['gen_state_dict']={k.replace('encoder.', ''): v for k, v in checkpoint_v['gen_state_dict'].items()}
            model_dict.update(checkpoint_h['model'])
            model_dict.update(checkpoint_v['gen_state_dict'])
            model_dict = model.state_dict()
            self.model.load_state_dict(model_dict, strict=False)
            # 还是不行--还是需要预训练
        else:
            self.model = model.to(self.device)
            checkpoint_h['model'] = {k.replace('module.', ''): v for k, v in checkpoint_h['model'].items()}
            self.model.load_state_dict(checkpoint_h['model'], strict=True)
            self.model.load_state_dict(checkpoint_v['gen_state_dict'], strict=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options.lr,
                                          betas=(self.options.beta1, self.options.beta2))  # 0.5，0.9
        print('---> model and optimizer have been set')

    def set_dataloader(self, ):
        if self.options.dataset == '3dpw':
            dataset = PW3D(self.options)
            self.imgdir = config.PW3D_ROOT
        else:
            dataset = Internet_dataset()
            self.imgdir = osp.join(config.InternetData_ROOT, 'images')
        # sampler生成一系列的index，而batch_sampler则是将sampler生成的indices打包分组，得到一个又一个batch的index。
        self.dataloader = DataLoader(dataset, batch_size=self.options.batch_size, shuffle=False, num_workers=0)
        # self.dataloader = DataLoader(dataset, batch_size=self.options.batch_size, shuffle=True, num_workers=8)
        # 不并行读取，每个batch不打乱,因为每次读取数据是一帧一帧，如果打乱就是随机取出作为输入了

    def set_criterion(self, ):
        self.gmm_f = MaxMixturePrior(prior_folder='/home/neu307/liumeilin/boa_data/spin_data', num_gaussians=8, dtype=torch.float32).to(
            self.device)
        self.cosembeddingloss = nn.CosineEmbeddingLoss().to(self.device)  # 余弦相似度

    def setup_smpl(self, ):
        self.smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(self.device)
        self.smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(self.device)
        self.smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(self.device)
        self.joint_mapper_h36m = constants.H36M_TO_J14
        self.joint_mapper_gt = constants.J24_TO_J14
        self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    def set_teacher(self, ):
        model = vibe(config.SMPL_MEAN_PARAMS)
        # 加载pth预训练模型
        checkpoint_h = torch.load(self.options.model_file_hmr)  # 各层参数 data/basemodel.pt
        checkpoint_v = torch.load(self.options.model_file_vibe)

        # pretrained_dict_h = torch.load(self.options.model_file_hmr)
        # pretrained_dict_v = torch.load(self.options.model_file_vibe)
        # pretrained_dict_h1 = pretrained_dict_h['model']
        # pretrained_dict_v1 = pretrained_dict_v['gen_state_dict']

        # model_dict = model.state_dict()  # 后加,查看现有网络结构的名字和对应的参数
        for param in model.parameters():
            param.detach_()  # 将param从创建它的图中分离，并把它设置成叶子tensor，将param的grad_fn的值设置为None,这样就不会再与前一个节点x关联。
            # 并将param的requires_grad设置为False，这样进行backward()时就不会求param的梯度
        self.teacher = model.to(self.device)
        model_dict = model.state_dict()
        # 重新制作预训练的权重，主要是减去参数不匹配的层
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'encoder.gru' not in k)}
        # pretrained_dict_v1 = {k.replace('regressor.', ''): v for k, v in pretrained_dict_v1.items()}
        # pretrained_dict_g = {k.replace('encoder.', ''): v for k, v in pretrained_dict_v1.items()}
        # # 更新权重
        # model_dict.update(pretrained_dict_g)
        # model_dict.update(pretrained_dict_h1)
        # checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        # 将checkpoint中键值k的module.换为空格
        self.teacher.load_state_dict(model_dict, strict=False)

    def projection(self, cam, s3d, eps=1e-9):
        cam_t = torch.stack([cam[:, 1], cam[:, 2],
                             2 * constants.FOCAL_LENGTH / (constants.IMG_RES * cam[:, 0] + eps)], dim=-1)
        camera_center = torch.zeros(s3d.shape[0], 2, device=self.device)
        s2d = perspective_projection(s3d,
                                     rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(s3d.shape[0], -1,
                                                                                                   -1),
                                     translation=cam_t,
                                     focal_length=constants.FOCAL_LENGTH,
                                     camera_center=camera_center)
        s2d_norm = s2d / (constants.IMG_RES / 2.)  # to [-1,1]
        return {'ori': s2d, 'normed': s2d_norm}

    def get_hist(self, ):  # 调取5帧前对应的图片和2D标签
        infos = self.history[self.global_step - self.options.interval]
        return torch.from_numpy(infos['image']).to(self.device), torch.from_numpy(infos['s2d']).to(self.device)

    def save_hist(self, image, s2d):
        self.history[self.global_step] = {'image': image.detach().cpu().numpy(),
                                          's2d': s2d.detach().cpu().numpy()}

    def decode_smpl_params(self, poses, beta, gender='neutral', pose2rot=False):
        # 需要判断当前输入视频帧数,只取最后一帧
        if gender == 'neutral':
            smpl_out = self.smpl_neutral(betas=beta, body_pose=poses[:, 1:], global_orient=poses[:, 0].unsqueeze(1),
                                         pose2rot=pose2rot)
        elif gender == 'male':
            smpl_out = self.smpl_male(betas=beta, body_pose=poses[:, 1:], global_orient=poses[:, 0].unsqueeze(1),
                                      pose2rot=pose2rot)
        elif gender == 'female':
            smpl_out = self.smpl_female(betas=beta, body_pose=poses[:, 1:], global_orient=poses[:, 0].unsqueeze(1),
                                        pose2rot=pose2rot)
        return {'s3d': smpl_out.joints, 'vts': smpl_out.vertices}

    def update_teacher(self, teacher, model):
        """
        teacher = teacher * alpha + model * (1 - alpha)
        In general, I set alpha to be 0.1.
        """
        factor = self.options.alpha
        for param_t, param in zip(teacher.parameters(), model.parameters()):
            # param_t.data.mul_(factor).add_(1 - factor, param.data)
            param_t.data.mul_(factor).add_(param.data, alpha=1 - factor)

    def excute(self, ):
        pass

    def adaptation(self):
        pass

    def cal_feature_diff(self, features_i, features_j):
        sims_dict = {}
        mean_cos_sim = 0
        for i, (feat_i, feat_j) in enumerate(zip(features_i, features_j)):
            cos_sim = F.cosine_similarity(feat_i.flatten(), feat_j.flatten(), dim=0, eps=1e-12)
            mean_cos_sim += cos_sim
            sims_dict[i] = {'cos': cos_sim.item(), }
        self.fit_losses['feat_sim/cos_sim'] = mean_cos_sim / i
        return sims_dict

    def num_process(self, pred_rotmat, pred_shape, pred_cam):
        # to reduce time dimension
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])  # 前两维相乘，逗号是将int改为字符串格式，再连接后面两个维度
        # flatten for weight vectors
        flatten = lambda x: x.reshape(-1)
        # accumulate all predicted thetas from IEF
        accumulate_thetas = lambda x: torch.cat([output['theta'] for output in x], 0)

        return pred_rotmat, pred_shape, pred_cam

    def lower_level_adaptation(self, seqlen, image, gt_keypoints_2d, h36m_batch, learner=None):
        batch_size = image.shape[0]
        gt_keypoints_2d = gt_keypoints_2d.to(torch.float32)
        pred_rotmats, pred_shapes, pred_cams, init_features = learner(image, need_feature=True)  # init_features=16
        # 此处需要对多帧进行处理（每次取一帧进行参数计算，最后再加一起求loss）
        new_pred_s3d = []
        new_pred_s2d = []
        new_pred_vertices = []
        new_pred_rotmat = []
        new_pred_shape = []

        for i in range(seqlen):
            pred_rotmat = pred_rotmats[:, i, :]  # 取当前一帧的
            # pred_rotmat = torch.squeeze(pred_rotmat, dim=1)  # 删除第一维
            pred_shape = pred_shapes[:, i, :]  # 取当前一帧的
            # pred_shape = torch.squeeze(pred_shape, dim=1)  # 删除第一维
            pred_cam = pred_cams[:, i, :]  # 取当前一帧的
            # pred_cam = torch.squeeze(pred_cam, dim=1)  # 删除第一维

            smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
            pred_s3d = smpl_out['s3d']
            pred_vertices = smpl_out['vts']
            pred_s2d = self.projection(pred_cam, pred_s3d)['normed']

            #  取25以后的，且1-3维坐标反向，倒着取，再在第三维加一个维度
            # 加一起
            new_pred_s3d.append(pred_s3d.unsqueeze(1))  # (3,1,49,3)
            new_pred_s2d.append(pred_s2d.unsqueeze(1))
            new_pred_vertices.append(pred_vertices.unsqueeze(1))
            new_pred_rotmat.append(pred_rotmat.unsqueeze(1))
            new_pred_shape.append(pred_shape.unsqueeze(1))

        pred_s3d = torch.cat(new_pred_s3d, dim=1)  # (3,16,49,3)
        pred_s2d = torch.cat(new_pred_s2d, dim=1)
        # pred_vertices = torch.cat(new_pred_vertices, dim=1)
        pred_rotmat = torch.cat(new_pred_rotmat, dim=1)
        pred_shape = torch.cat(new_pred_shape, dim=1)
        conf = gt_keypoints_2d[:, :, 25:, -1].unsqueeze(-1).clone()  # [1,16,24,1]

        # 对处理后的三个参数计算loss
        if self.options.use_frame_losses_lower:
            # calculate losses
            # 2D keypoint loss
            s2dloss = (F.mse_loss(pred_s2d[:, :, 25:, :], gt_keypoints_2d[:, :, 25:, :-1], reduction='none') * conf).mean()
            # shape prior constraint
            shape_prior = self.cal_shape_prior(pred_shape)  # 12.9769
            # pose prior constraint
            pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)  # 137.3873
            loss = s2dloss * self.options.s2dloss_weight + \
                   shape_prior * self.options.shape_prior_weight + \
                   pose_prior * self.options.pose_prior_weight
            # 0.3478
            self.kp2dlosses_lower.append(s2dloss.item())
            self.fit_losses['ll/s2dloss'] = s2dloss
            self.fit_losses['ll/shape_prior'] = shape_prior
            self.fit_losses['ll/pose_prior'] = pose_prior
            self.fit_losses['ll/unlabelloss'] = loss

        if self.options.use_temporal_losses_lower:  # 0
            if self.options.use_meanteacher:
                teacherloss = self.cal_teacher_loss(image, pred_rotmat, pred_shape, pred_s2d, pred_s3d)
                if self.options.use_frame_losses_lower:
                    loss += teacherloss * self.options.teacherloss_weight
                else:
                    loss = teacherloss * self.options.teacherloss_weight

            if self.options.use_motion and (self.global_step - self.options.interval) > 0:
                motionloss = self.cal_motion_loss(learner, pred_s2d[:, 25:], gt_keypoints_2d[:, 25:], prefix='ul')
                loss += motionloss * self.options.motionloss_weight

        if self.options.retrieval:  # 从源域找类似标签
            # h36m_batch = self.retrieval(init_features[5], seqlen)  # [3*16,2048]
            h36m_batch = self.retrieval(init_features[6], seqlen)  # [3*16,2048]
        # 当多帧时，也输入多帧一起求loss
        if self.options.lower_level_mixtrain:
            lableloss, label_feats = self.adapt_on_labeled_data(learner, h36m_batch, seqlen, prefix='ll')
            loss += lableloss * self.options.labelloss_weight

        return loss, init_features

    def upper_level_adaptation(self, seqlen, image, gt_keypoints_2d, h36m_batch, learner=None):
        image = image.to(torch.float32)
        batch_size = image.shape[0]
        gt_keypoints_2d = gt_keypoints_2d.to(torch.float32)
        pred_rotmats, pred_shapes, pred_cams, init_features = learner(image, need_feature=True)
        # 此处需要对多帧进行处理（合并）
        new_pred_s3d = []
        new_pred_s2d = []
        new_pred_vertices = []
        new_pred_rotmat = []
        new_pred_shape = []
        for i in range(seqlen):
            pred_rotmat = pred_rotmats[:, i, :]  # 取当前一帧的
            pred_shape = pred_shapes[:, i, :]  # 取当前一帧的
            pred_cam = pred_cams[:, i, :]  # 取当前一帧的

            smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
            pred_s3d = smpl_out['s3d']
            pred_vertices = smpl_out['vts']
            pred_s2d = self.projection(pred_cam, pred_s3d)['normed']
            #  取25以后的，且1-3维坐标反向，倒着取，再在第三维加一个维度
            # 加一起
            new_pred_s3d.append(pred_s3d.unsqueeze(1))  # (3,1,49,3)
            new_pred_s2d.append(pred_s2d.unsqueeze(1))
            new_pred_vertices.append(pred_vertices.unsqueeze(1))
            new_pred_rotmat.append(pred_rotmat.unsqueeze(1))
            new_pred_shape.append(pred_shape.unsqueeze(1))

        pred_s3d = torch.cat(new_pred_s3d, dim=1)  # (1,16,49,3)
        pred_s2d = torch.cat(new_pred_s2d, dim=1)  # (1,16,49,2)
        # pred_vertices = torch.cat(new_pred_vertices, dim=1)
        pred_rotmat = torch.cat(new_pred_rotmat, dim=1)  # (1,16,24,3,3)
        pred_shape = torch.cat(new_pred_shape, dim=1)  # (1,16,10)
        conf = gt_keypoints_2d[:, :, 25:, -1].unsqueeze(-1).clone()  # [1,16,24,1]

        if self.options.use_frame_losses_upper:
            # calculate losses
            # 2D keypoint loss
            s2dloss = (F.mse_loss(pred_s2d[:, :, 25:], gt_keypoints_2d[:, :, 25:, :-1], reduction='none') * conf).mean()  # 0.0157
            # shape prior constraint
            shape_prior = self.cal_shape_prior(pred_shape)  # 0.8080
            # pose prior constraint
            pose_prior = self.cal_pose_prior(pred_rotmat, pred_shape)  # 131.7622
            loss = s2dloss * self.options.s2dloss_weight + \
                   shape_prior * self.options.shape_prior_weight + \
                   pose_prior * self.options.pose_prior_weight
            # 0.1698
            self.kp2dlosses_upper[self.global_step] = s2dloss.item()
            self.fit_losses['ul/s2dloss'] = s2dloss
            self.fit_losses['ul/shape_prior'] = shape_prior
            self.fit_losses['ul/pose_prior'] = pose_prior
            self.fit_losses['ul/unlabelloss'] = loss
        ########
        # 通过教师模型来记录上一帧的模型参数，再将当前帧输入模型得到的指标和当前帧输入教师模型得到的指标进行loss
        ########
        if self.options.use_temporal_losses_upper:  # 1
            if self.options.use_meanteacher:  # loss特别大？？！
                teacherloss = self.cal_teacher_loss(image, seqlen, pred_rotmat, pred_shape, pred_s2d, pred_s3d)  # 0.0474
                if self.options.use_frame_losses_upper:
                    loss += teacherloss * self.options.teacherloss_weight  # 0.1745
                else:
                    loss = teacherloss * self.options.teacherloss_weight
            # ！！还需要更正教师模型和时序损失在多帧输入上的变化？？
            if self.options.use_motion and (self.global_step - 15 - self.options.interval) >= 0:
                motionloss = self.cal_motion_loss(learner, seqlen, pred_s2d[:, :, 25:], gt_keypoints_2d[:, :, 25:], prefix='ul')
                loss += motionloss * self.options.motionloss_weight

        ###########
        # 在源域找最接近的图像的标签作为监督
        ###########
        if self.options.retrieval:  # 找batch
            # h36m_batch = self.retrieval(init_features[5], seqlen)
            h36m_batch = self.retrieval(init_features[6], seqlen)
        if self.options.upper_level_mixtrain:  # 参与训练求loss
            lableloss, label_feats = self.adapt_on_labeled_data(learner, h36m_batch, seqlen, prefix='ul')
            loss += lableloss * self.options.labelloss_weight

        return loss, init_features

    def cal_teacher_loss(self, image, seqlen, pred_rotmat, pred_shape, pred_s2d, pred_s3d):
        """
        we calculate same loss items as SPIN. 
        """
        ema_rotmats, ema_shapes, ema_cams = self.teacher(image)

        new_ema_rotmat=[]
        new_ema_shape=[]
        new_ema_cam=[]
        new_ema_s2d=[]
        new_ema_pred_s3d=[]
        for i in range(seqlen):
            ema_rotmat = ema_rotmats[:, i, :]  # 取当前一帧的 [1,24,3,3]
            ema_shape = ema_shapes[:, i, :]  # 取当前一帧的 [1,10]
            ema_cam = ema_cams[:, i, :]  # 取当前一帧的 [1,3]
            # 在生成smpl时需要单帧才行,需要遍历
            ema_smpl_out = self.decode_smpl_params(ema_rotmat, ema_shape)
            ema_pred_s3d = ema_smpl_out['s3d']
            ema_pred_vts = ema_smpl_out['vts']
            # 投影
            ema_s2d = self.projection(ema_cam, ema_pred_s3d)['normed']
            # 合一起
            new_ema_rotmat.append(ema_rotmat.unsqueeze(1))
            new_ema_shape.append(ema_shape.unsqueeze(1))
            new_ema_cam.append(ema_cam.unsqueeze(1))
            new_ema_s2d.append(ema_s2d.unsqueeze(1))
            new_ema_pred_s3d.append(ema_pred_s3d.unsqueeze(1))

        ema_rotmat = torch.cat(new_ema_rotmat, dim=1)
        ema_shape = torch.cat(new_ema_shape, dim=1)
        ema_cam = torch.cat(new_ema_cam, dim=1)
        ema_s2d = torch.cat(new_ema_s2d, dim=1)
        ema_pred_s3d = torch.cat(new_ema_pred_s3d, dim=1)

        # 2d and 3d kp losses
        s2dloss = F.mse_loss(pred_s2d, ema_s2d)
        s3dloss = F.mse_loss(ema_pred_s3d, pred_s3d)
        # beta and theta losses
        shape_loss = F.mse_loss(pred_shape, ema_shape)
        pose_loss = F.mse_loss(pred_rotmat, ema_rotmat)

        loss = s2dloss * 5 + s3dloss * 5 + shape_loss * 0.001 + pose_loss * 1
        self.fit_losses['teacher/s2dloss'] = s2dloss
        self.fit_losses['teacher/s3dloss'] = s3dloss
        self.fit_losses['teacher/shape_loss'] = shape_loss
        self.fit_losses['teacher/pose_loss'] = pose_loss
        self.fit_losses['teacher/loss'] = loss
        return loss

    def adapt_on_labeled_data(self, model, batch, seqlen, prefix='ll'):
        image = batch['img']
        gt_s3d = batch['pose_3d']
        gt_shape = batch['betas']
        gt_pose = batch['pose']
        gt_s2d = batch['keypoints']  # [16,49,3]
        conf = gt_s2d[:, 25:, -1].unsqueeze(-1).clone()  # [16,24,1]
        # 由于从源域提取的数据，虽然按seqlen合并了，但没有batch_size维度
        #------------ 在image中指定位置N加上一个维数为1的维度??作为batchsize，但是如果batchsize不是1，怎么办?
        image = torch.unsqueeze(image, 0)  # [1,16,3,224,224]
        gt_s3d = torch.unsqueeze(gt_s3d, 0)
        gt_shape = torch.unsqueeze(gt_shape, 0)
        gt_pose = torch.unsqueeze(gt_pose, 0)
        gt_s2d = torch.unsqueeze(gt_s2d, 0)
        conf = torch.unsqueeze(conf, 0)

        pred_rotmats, pred_shapes, pred_cams, label_feats = model(image, need_feature=True)  # 多帧
        new_pred_s2d = []
        new_pred_s3d = []
        for i in range(seqlen):
            pred_rotmat = pred_rotmats[:, i, :]  # 取当前一帧的 [1,24,3,3]
            pred_shape = pred_shapes[:, i, :]  # 取当前一帧的 [1,10]
            pred_cam = pred_cams[:, i, :]  # 取当前一帧的 [1,3]
            # 在生成smpl时需要单帧才行,需要遍历
            smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
            pred_s3d = smpl_out['s3d']  # [1,49,3]
            # pred_vertices = smpl_out['vts']
            pred_s2d = self.projection(pred_cam, pred_s3d)['normed']  # [1,49,2]
            # 加一起
            new_pred_s2d.append(pred_s2d.unsqueeze(1))
            new_pred_s3d.append(pred_s3d.unsqueeze(1))  # (1,49,3)
        pred_s2ds = torch.cat(new_pred_s2d, dim=1)
        pred_s3ds = torch.cat(new_pred_s3d, dim=1)  # (1,16,49,3)
        # shape and pose losses
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)  # 转换！[16,24,3,3]
        gt_rotmat = torch.unsqueeze(gt_rotmat, 0)  # 加了一个维度，方便与pred_rotmat一致 [1, 16,24,3,3]
        pose_loss = F.mse_loss(pred_rotmats, gt_rotmat)  # 0.1035
        shape_loss = F.mse_loss(pred_shapes, gt_shape)  # 0.8157
        # 2d kp loss
        # pred_s2d = self.projection(pred_cam, pred_s3ds)['normed']  # 上面包含了这步
        s2dloss = (F.mse_loss(pred_s2ds[:, :, 25:], gt_s2d[:, :, 25:, :-1], reduction='none') * conf).mean()  # 0.1178
        # 3d kp loss
        s3dloss = self.cal_s3d_loss(pred_s3ds[:, :, 25:], gt_s3d[:, :, :, :-1], conf)
        assert gt_s3d.shape[2] == 24
        # -------------------权重设置？？？5是怎么来的
        loss = s2dloss * 5 + s3dloss * 5 + shape_loss * 0.001 + pose_loss * 1  # 0.9919
        self.fit_losses[f'{prefix}/labled_s2dloss'] = s2dloss
        self.fit_losses[f'{prefix}/labled_s3dloss'] = s3dloss
        self.fit_losses[f'{prefix}/labled_shape_loss'] = shape_loss
        self.fit_losses[f'{prefix}/labled_pose_loss'] = pose_loss
        self.fit_losses[f'{prefix}/labled_loss'] = loss
        return loss, label_feats

    def cal_motion_loss(self, model, seqlen, pred_s2ds, gt_s2ds, prefix='ul'):
        hist_images, hist_s2ds = self.get_hist()
        hist_pred_rotmats, hist_pred_shapes, hist_pred_cams = model(hist_images)

        new_pred_motion=[]
        new_gt_motion=[]
        new_conf=[]

        for i in range(seqlen):
            hist_pred_rotmat = hist_pred_rotmats[:, i, :]
            hist_pred_shape = hist_pred_shapes[:, i, :]
            hist_pred_cam = hist_pred_cams[:, i, :]
            pred_s2d = pred_s2ds[:, i, :]
            gt_s2d= gt_s2ds[:, i, :]
            hist_s2d = hist_s2ds[:, i, :]
            hist_smpl_out = self.decode_smpl_params(hist_pred_rotmat, hist_pred_shape)
            hist_pred_s3d = hist_smpl_out['s3d']

            # cal motion loss
            hist_pred_s2d = self.projection(hist_pred_cam, hist_pred_s3d)['normed']
            pred_motion = pred_s2d - hist_pred_s2d[:, 25:]
            gt_motion = gt_s2d[:, :, :-1] - hist_s2d[:, 25:, :-1]
            # cal non-zero confidence
            conf1 = hist_s2d[:, 25:, -1].unsqueeze(-1).clone()
            conf2 = gt_s2d[:, :, -1].unsqueeze(-1).clone()
            one = torch.tensor([1.]).to(self.device)
            zero = torch.tensor([0.]).to(self.device)
            conf = torch.where((conf1 + conf2) == 2, one, zero)

            new_pred_motion.append(pred_motion.unsqueeze(1))
            new_gt_motion.append(gt_motion.unsqueeze(1))
            new_conf.append(conf.unsqueeze(1))
        pred_motion = torch.cat(new_pred_motion, dim=1)
        gt_motion = torch.cat(new_gt_motion, dim=1)
        conf = torch.cat(new_conf, dim=1)

        motion_loss = (F.mse_loss(pred_motion, gt_motion, reduction='none') * conf).mean()
        self.fit_losses[f'{prefix}/motion_loss'] = motion_loss
        return motion_loss

    def cal_shape_prior(self, pred_betas):
        return (pred_betas ** 2).sum(dim=-1).mean()  # 求和后最后一维求平均值

    def cal_pose_prior(self, pred_rotmat, betas):
        # gmm prior
        body_pose = rotation_matrix_to_angle_axis(pred_rotmat[:, :, 1:].contiguous().view(-1, 3, 3)).contiguous().view(-1,
                                                                                                                    69)
        pose_prior_loss = self.gmm_f(body_pose, betas).mean()
        return pose_prior_loss

    def cal_s3d_loss(self, pred_s3d, gt_s3d, conf):
        """ 
        align the s3d and then cal the mse loss
        Input: (N,24,2)
        """
        # 以下第二维都加了 :,
        gt_hip = (gt_s3d[:, :, 2] + gt_s3d[:, :, 3]) / 2
        gt_s3d = gt_s3d - gt_hip[:, :, None, :]
        pred_hip = (pred_s3d[:, :, 2] + pred_s3d[:, :, 3]) / 2
        pred_s3d = pred_s3d - pred_hip[:, :, None, :]
        loss = (conf * F.mse_loss(pred_s3d, gt_s3d, reduction='none')).mean()
        return loss

    # def inference(self, batch, model, need_feature=False):
    def inference(self, batch, model, J_regressor=None):
        pass

    def save_results(self, vts, cam_trans, images, name, bbox, prefix=None):
        vts = vts.clone().detach().cpu().numpy()
        cam_trans = cam_trans.clone().detach().cpu().numpy()
        images = images.clone().detach()
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
        images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))
        for i in range(vts.shape[0]):
            oriimg = cv2.imread(os.path.join(self.imgdir, name[i]))
            ori_h, ori_w = oriimg.shape[:2]
            bbox = bbox.cpu().numpy()
            ori_pred_cams = convert_crop_cam_to_orig_img(cam_trans, bbox, ori_w, ori_h)
            renderer = Renderer(resolution=(ori_w, ori_h), orig_img=True, wireframe=False)
            rendered_image = renderer.render(oriimg, vts[i], ori_pred_cams[i], color=np.array([205, 129, 98]) / 255,
                                             mesh_filename='demo.obj')
            cv2.imwrite(osp.join(self.exppath, 'image', f'{prefix}_{self.global_step + i}.png'), rendered_image)

    def write_summaries(self, losses):
        for loss_name, val in losses.items():
            # self.summary_writer.add_scalar(loss_name, val, self.global_step)
            self.summary_writer.add_scalar(loss_name, val, self.epoch+1)


class SourceDataset(Dataset):
    def __init__(self, datapath):
        super(SourceDataset, self).__init__()
        self.img_dir = config.H36M_ROOT
        self.data = joblib.load(datapath)
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        # == parse data == #
        self.imgname = self.data['imgname']
        # import ipdb;ipdb.set_trace()
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.pose = self.data['pose'].astype(np.float)
        self.betas = self.data['shape'].astype(np.float)
        self.pose_3d = self.data['S']
        keypoints_gt = self.data['part']
        keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)
        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)
        self.length = self.scale.shape[0]

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        pose = self.pose[index].copy()
        betas = self.betas[index].copy()

        # Load image
        imgname = os.path.join(self.img_dir, self.imgname[index])
        img = self.read_image(imgname)
        item['oriimage'] = img.copy()
        orig_shape = np.array(img.shape)[:2]

        # no augmentation
        rot, sc = 0, 1

        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc * scale, rot)).float().unsqueeze(
            0)
        img = self.rgb_processing(img, center, sc * scale, rot)
        item['oriimage2'] = [img.copy(), center, sc * scale, rot]

        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)

        item['img'] = img.unsqueeze(0)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot)).float().unsqueeze(0)
        item['betas'] = torch.from_numpy(betas).float().unsqueeze(0)
        item['imgname'] = imgname
        S = self.pose_3d[index].copy()
        item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot)).float().unsqueeze(0)
        return item

    def __len__(self):
        return len(self.imgname)

    def j2d_processing(self, kp, center, scale, r):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / constants.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp

    def read_image(self, imgname):
        img = cv2.imread(imgname)
        if not img.any():
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                imgname)
        return img[:, :, ::-1].copy().astype(np.float32)  # 列表数组左右翻转,把RGB(或BRG)转换成BGR(或者RGB)

    def rgb_processing(self, rgb_img, center, scale, rot):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
        return rgb_img

    def pose_processing(self, pose, r):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def j3d_processing(self, S, r):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        S = S.astype('float32')
        return S
