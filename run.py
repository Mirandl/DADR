"""
dynaboa
"""

import os
import time
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch
import constants
from progress.bar import Bar
from utils.pose_utils import (
    compute_similarity_transform_batch,
    compute_accel,
    compute_error_accel,
)
from base_adaptor import BaseAdaptor

# from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', type=str, default='exps')
parser.add_argument('--expname', type=str, default='3dpw')
parser.add_argument('--dataset', type=str, default='3dpw', choices=['3dpw', 'internet'])
parser.add_argument('--seed', type=int, default=22, help='random seed')
parser.add_argument('--seq_seed', type=int, default=22, help='random seed')
parser.add_argument('--model_file_hmr', type=str, default='data/basemodel.pt')
parser.add_argument('--model_file_vibe', type=str, default='/home/neu307/liumeilin/boa_data/model_best.pth.tar')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--save_res', type=int, default=0, choices=[0, 1], help='save middle mesh and image')

parser.add_argument('--lr', type=float, default=3e-6, help='learning rate of the upper-level')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 of adam')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch for start -later added')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch for end -later added')
parser.add_argument('--epoch_flag', type=int, default=0, help='flag for whether to retrival in each epoch-later added')
parser.add_argument('--retrieval_type', type=str, default='inner_feature',
                    help='feature type to retrival in each epoch-later added,inner_feature or outer_theta')
parser.add_argument('--num_iters_per_epoch', type=int, default=372, help='iter for each epoch -later added')
parser.add_argument('--seqlen', type=int, default=16, help='sequence per batch -later added')

# boa
parser.add_argument('--use_boa', type=int, default=1, choices=[0, 1], help='use boa')
parser.add_argument('--fastlr', type=float, default=8e-6,
                    help='fast learning rate, which is the parameter of lower-level')
parser.add_argument('--inner_step', type=int, default=1, help='steps of inner loop')
parser.add_argument('--record_lowerlevel', type=int, default=1, help='record results of the lower level?')
parser.add_argument('--s2dloss_weight', type=float, default=10)
parser.add_argument('--shape_prior_weight', type=float, default=2e-6)
parser.add_argument('--pose_prior_weight', type=float, default=1e-4)

parser.add_argument('--use_frame_losses_lower', type=int, default=1, choices=[0, 1],
                    help='whether use frame-wise losses at lower level')
parser.add_argument('--use_frame_losses_upper', type=int, default=1, choices=[0, 1],
                    help='whether use frame-wise losses at upper level')
parser.add_argument('--use_temporal_losses_lower', type=int, default=0, choices=[0, 1],
                    help='whether use temporal-wise losses at lower level')
parser.add_argument('--use_temporal_losses_upper', type=int, default=1, choices=[0, 1],
                    help='whether use temporal-wise losses at upper level')

parser.add_argument('--sample_num', type=int, default=1, help='sample_num')
parser.add_argument('--retrieval', type=int, default=1, choices=[0, 1], help='use retrieval')

parser.add_argument('--dynamic_boa', type=int, default=1, choices=[0, 1], help='dynamic boa')
parser.add_argument('--cos_sim_threshold', type=float, default=3.1e-4, help='cos sim threshold')
parser.add_argument('--optim_steps', type=int, default=7, help='steps of the boa for the current image')

# mix training
parser.add_argument('--lower_level_mixtrain', type=int, default=1, choices=[0, 1], help='use mix training')
parser.add_argument('--upper_level_mixtrain', type=int, default=1, choices=[0, 1], help='use mix training')
parser.add_argument('--mixtrain', type=int, help='use mix training')
parser.add_argument('--labelloss_weight', type=float, default=0.1, help='weight of h36m loss')

# teacher
parser.add_argument('--use_meanteacher', type=int, default=1, choices=[0, 1], help='1: use mean teacher')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha * teacher + (1-alpha) * model, scale: 0-1')
parser.add_argument('--teacherloss_weight', type=float, default=0.1)

# motion
parser.add_argument('--use_motion', type=int, default=1, choices=[0, 1], help='1: use mean teacher')
parser.add_argument('--interval', type=int, default=5, help='interval of temporal loss, scale: >= 1')
parser.add_argument('--motionloss_weight', type=float, default=0.8)


class Adaptor(BaseAdaptor):

    def excute(self, ):
        for epoch in range(self.options.start_epoch, self.options.end_epoch):
            self.epoch = epoch  # 后加
            self.epoch_flag = 0
            self.sims = []
            self.feat_sims = {}
            self.optim_step_record = []
            mpjpe_all, pampjpe_all = [], []
            pve_all = []
            self.mpjpe_statistics, self.pampjpe_statistics = [[] for i in range(len(self.dataloader))], [[] for i in
                                                                                                         range(
                                                                                                             len(self.dataloader))]  # 35515个[] 分成32个batch-> 1110个[]
            self.mpjpe_all_lower, self.pampjpe_all_lower = [[] for i in range(self.options.inner_step)], [[] for i in
                                                                                                          range(
                                                                                                              self.options.inner_step)]  # 1个[]
            self.history = {}
            self.kp2dlosses_lower = []
            self.kp2dlosses_upper = {}
            self.load = False
            self.stride = 1

            timer = {
                'data': 0,
                'forward': 0,
                'loss': 0,
                'batch': 0,
            }
            start = time.time()

            self.dataloader_iter = None
            self.dataloader_iter = iter(self.dataloader)
            # 如果更改了窗口大小和步长，options.num_iters_per_epoch需要重新计算并改变
            bar = Bar(f'Epoch {self.epoch + 1}/{self.options.end_epoch}', fill='#',
                      max=self.options.num_iters_per_epoch)
            # 进度条
            for i in range(self.options.num_iters_per_epoch):
                # Dirty solution to reset an iterator
                dataloader = None
                if self.dataloader_iter:  # do
                    try:
                        dataloader = next(self.dataloader_iter)  # 给target2d赋初值
                    except StopIteration:
                        self.dataloader_iter = iter(self.dataloader)
                        dataloader = next(self.dataloader_iter)
                    BaseAdaptor.move_dict_to_device(dataloader, self.device)
                self.global_step = i

                timer['data'] = time.time() - start
                start = time.time()

                # 按所有输入帧依次读取，不去重
                #   for step, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                #       self.global_step = step  # 索引，第n组图片
                # iter访问迭代对象时只返回元素，enumerate除了元素batch还会返回索引step，或者叫做访问顺序
                self.fit_losses = {}
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in dataloader.items()}
                # 如果v是向量就传到GPU上，否则还是v
                # Step1: adaptation
                # self.model.eval()
                self.model.train()  # 预测阶段，自动把BN和DropOut固定住。也会加上torch.no_grad()来关闭梯度的计算。
                mpjpe, pampjpe, pve = self.adaptation(batch)  # 138-152;87-94;11317

                timer['forward'] = time.time() - start
                start = time.time()

                # Step2: inference
                # 这里保存了16帧的loss-每次取16帧的平均值
                self.fit_losses['metrics/mpjpe'] = torch.tensor(mpjpe.mean())
                self.fit_losses['metrics/pampjpe'] = torch.tensor(pampjpe.mean())
                self.fit_losses['metrics/pve'] = torch.tensor(pve.mean())

                # self.write_summaries(self.fit_losses)
                # 每次只保存一个batch的平均误差值
                mpjpe_all.append(mpjpe.mean())
                pampjpe_all.append(pampjpe.mean())
                pve_all.append(pve.mean())

                timer['loss'] = time.time() - start
                start = time.time()

                timer['batch'] = timer['data'] + timer['forward'] + timer['loss']
                start = time.time()
                summary_string = f'({i + 1}/{self.options.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                                 f'ETA: {bar.eta_td:}'
                for k, v in timer.items():
                    summary_string += f' | {k}: {v:.2f}'

                # ！！！原来是每200张图片输出一次step误差，那改成16张为一个batch之后，这个数需要调整
                # if (self.global_step + 1) % 200 == 0:
                if (self.global_step + 1) % 10 == 0:
                    # 只打印10组batch中的每组平均误差值的平均值
                    print(
                        f'\nStep:{self.global_step}: MPJPE:{np.mean(mpjpe_all)}, PAMPJPE:{np.mean(pampjpe_all)}, PVE:{np.mean(pve_all)}')
                if self.load:
                    self.load_ckpt()
                bar.suffix = summary_string
                bar.next()

            self.write_summaries(self.fit_losses)

            bar.finish()
            # logout the results
            print('--- Final ---')
            print(
                f'\nStep:{self.global_step}: MPJPE:{np.mean(mpjpe_all)}, PAMPJPE:{np.mean(pampjpe_all)}, PVE:{np.mean(pve_all)}')
            for i in range(self.options.inner_step):
                print(
                    f'\nLower-level  Step:{i} MPJPE:{np.mean(self.mpjpe_all_lower[i])}, PAMPJPE:{np.mean(self.pampjpe_all_lower[i])}')

            # save results
            joblib.dump({'kp2dloss': self.kp2dlosses_lower}, osp.join(self.exppath, 'lowerlevel_kp2dloss.pt'))
            joblib.dump({'kp2dloss': self.kp2dlosses_upper}, osp.join(self.exppath, 'upperlevel_kp2dloss.pt'))

            joblib.dump({'mpjpe': mpjpe_all, 'pampjpe': pampjpe_all, 'pve': pve_all}, osp.join(self.exppath, 'res.pt'))
            joblib.dump({'mpjpe': self.mpjpe_all_lower, 'pampjpe': self.pampjpe_all_lower},
                        osp.join(self.exppath, 'lower_res.pt'))
            joblib.dump({'mpjpe': self.mpjpe_statistics, 'pampjpe': self.pampjpe_statistics},
                        osp.join(self.exppath, 'steps_statistic_res.pt'))
            joblib.dump({'feat': self.feat_sims}, osp.join(self.exppath, 'feat_sims.pt'))
            joblib.dump({'step': self.optim_step_record}, osp.join(self.exppath, 'optim_step_record.pt'))
            with open(osp.join(self.exppath, 'res.txt'), 'w') as f:
                f.write(
                    f'Step:{self.global_step}: MPJPE:{np.mean(mpjpe_all)}, PAMPJPE:{np.mean(pampjpe_all)}, PVE:{np.mean(pve_all)}\n')
                for i in range(self.options.inner_step):
                    f.write(
                        f'Lower-level  Step:{i} MPJPE:{np.mean(self.mpjpe_all_lower[i])}, PAMPJPE:{np.mean(self.pampjpe_all_lower[i])}\n')

    def adaptation(self, batch):
        image = batch['image']  # 1,3,224,224
        gt_keypoints_2d = batch['smpl_j2d']  # 1,49,3->bs,16,49,3
        self.save_hist(image, gt_keypoints_2d)  # 每输入一张图，保留图和2D标签

        # 本程序需要输入的是图像[32,3,224,224]，经特征处理后是[16，2048]
        # 而预训练模型vibe需要同时输入2d数据feature[19,16,2048]和3d数据feature[13,16,2048]，拼接后为[32,16,2048],
        # 32=batch size,16=seqlen
        # 由此产生维度上的差异
        if self.options.use_boa:
            with torch.no_grad():
                _, _, _, init_features = self.model(image, need_feature=True)
            h36m_batch = None
            # step 1, clone model
            learner = self.model.clone()
            # learner = self.model.clone().detach()
            # step 2, lower probe
            for i in range(self.options.inner_step):
                lower_level_loss, _ = self.lower_level_adaptation(self.options.seqlen, image, gt_keypoints_2d,
                                                                  h36m_batch, learner)  # 0.2884
                learner.adapt(lower_level_loss)  # 要保证里面的参数统一是float32，即类型一致！有一个float16的中间参数就会导致更新失败
                # to evaluate the lower-level model
                mpjpe, pampjpe, _ = self.inference(self.options.seqlen, batch, learner)
                # 这里保存了
                self.fit_losses[f'metrics/lower_{i}_mpjpe'] = torch.tensor(mpjpe[15])  # 151-156,比原来的122大了30！！
                self.fit_losses[f'metrics/lower_{i}_pampjpe'] = torch.tensor(pampjpe[15])  # 104-109，比原来的53.85大了一倍！
                self.mpjpe_all_lower[i].append(mpjpe)
                self.pampjpe_all_lower[i].append(pampjpe)
            # step 3, upper update
            # self.model.train()
            upper_level_loss, _ = self.upper_level_adaptation(self.options.seqlen, image, gt_keypoints_2d, h36m_batch,
                                                              learner)  # 0.2737
            # loss实际上已经默认求平均了
            self.optimizer.zero_grad()  # 清空之前的梯度
            # self.model.train()
            # upper_level_loss.clone().detach()
            upper_level_loss.backward()  # 反向传播，计算当前梯度
            self.optimizer.step()  # 根据梯度更新网络参数
            if self.options.use_meanteacher:
                # update the mean teacher
                self.update_teacher(self.teacher, self.model)

            # record pamjpe and mpjpe
            mpjpe, pampjpe, pve = self.inference(self.options.seqlen, batch, self.model)
            self.mpjpe_statistics[self.global_step] = [mpjpe, ]  # 151-157，比原来的122大了30！！
            self.pampjpe_statistics[self.global_step] = [pampjpe, ]  # 104-109，比原来的52.01大了一倍！

            if self.options.dynamic_boa:  # 重复优化，适应关键帧
                # cal similarity to judge whether this sample needs more optimization
                with torch.no_grad():
                    _, _, _, adapted_features = self.model(image, need_feature=True)
                    feat_sim_dict = self.cal_feature_diff(init_features, adapted_features)
                    feat_12 = feat_sim_dict[12]['cos']  # 取两次regressor后的feature？
                    self.feat_sims[self.global_step] = [feat_sim_dict]

                # while the 1-feat_12 not converge, continual optimizing.
                self.optimized_step = 0
                while 1 - feat_12 > self.options.cos_sim_threshold:
                    self.optimized_step += 1
                    if self.optimized_step > self.options.optim_steps:
                        # maximun optimzation step, stop the optimization.
                        break
                    # 这里是把训练不好的16张图全都重新送到上层网络，重新训练
                    upper_level_loss, adapted_features = self.upper_level_adaptation(self.options.seqlen, image,
                                                                                     gt_keypoints_2d, h36m_batch,
                                                                                     self.model)
                    self.optimizer.zero_grad()
                    upper_level_loss.backward()
                    self.optimizer.step()
                    if self.options.use_meanteacher:
                        self.update_teacher(self.teacher, self.model)
                    with torch.no_grad():
                        init_features = adapted_features  # 将新特征存为新的初始特征，便于下一轮计算相似度
                        _, _, _, adapted_features = self.model(image, need_feature=True)
                        feat_sim_dict = self.cal_feature_diff(init_features, adapted_features)
                        feat_12 = feat_sim_dict[12]['cos']
                        self.feat_sims[self.global_step].append(feat_sim_dict)
                    # record pamjpe and mpjpe
                    mpjpe, pampjpe, pve = self.inference(self.options.seqlen, batch, self.model)
                    self.mpjpe_statistics[self.global_step].append(mpjpe)
                    self.pampjpe_statistics[self.global_step].append(pampjpe)
                self.optim_step_record.append(self.optimized_step)
            return mpjpe, pampjpe, pve
        else:  # 只有底层通道
            h36m_batch = None
            lower_level_loss, _ = self.lower_level_adaptation(self.options.seqlen, image, gt_keypoints_2d, h36m_batch,
                                                              self.model)
            self.optimizer.zero_grad()
            lower_level_loss.backward()
            self.optimizer.step()
            mpjpe, pampjpe, pve = self.inference(self.options.seqlen, batch, self.model)
            return mpjpe, pampjpe, pve

    def inference(self, seqlen, batch, model, need_feature=False):
        image = batch['image'].type(torch.float32)
        gt_poses = batch['pose'].type(torch.float32)
        gt_betas = batch['betas'].type(torch.float32)
        genders = batch['gender']
        # ???
        # model.eval()
        with torch.no_grad():
            if need_feature:
                pred_rotmats, pred_shapes, pred_cam, features = model(image, need_feature)
            else:
                pred_rotmats, pred_shapes, pred_cam = model(image, need_feature)
        # 求当前smpl输出--四步法
        new_gt_keypoints_3d = []
        new_pred_keypoints_3d = []
        new_S1_hat = []
        new_S2 = []
        new_gt_vertices = []
        new_pred_vertices = []
        new_mpjpe_s = []
        new_pampjpe_s = []
        new_pve_s = []

        for i in range(seqlen):
            # 取当前一帧的
            gt_pose = gt_poses[:, i, :]
            gt_beta = gt_betas[:, i, :]
            pred_rotmat = pred_rotmats[:, i, :]
            pred_shape = pred_shapes[:, i, :]
            gender = genders[:, i]
            # smpl参数处理
            smpl_out = self.decode_smpl_params(pred_rotmat, pred_shape)
            pred_joints = smpl_out['s3d']
            pred_vertices = smpl_out['vts']

            # 这里pred_vertices只借用了第一维的batch size
            J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
                self.device)  # [1,17,6890]
            gt_vertices = self.smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                         betas=gt_beta).vertices  # [1,6890,3]
            gt_vertices_female = self.smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                  betas=gt_beta).vertices  # [1,6890,3]
            gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]  # 啥意思？

            # 这一步是分帧计算再合并还是这样合起来计算，直接这样计算的第二维度是16。有可能影响最后loss！
            # 下面决定分帧计算,第二维度的16都没了：
            gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)  # [1,16,17,3]
            gt_pelvis = gt_keypoints_3d[:, [0], :].clone()  # [1,16,1,3]
            # gt_pelvis = (gt_keypoints_3d[:, [2], :] + gt_keypoints_3d[:, [3], :]) / 2.0
            gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_h36m, :]  # [1,16,14,3]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis  # [1,16,14,3]
            # 这一步是分帧计算再合并还是这样合起来计算，直接这样计算的第二维度是16。有可能影响最后loss！
            # Get 14 predicted joints from the mesh
            # 处理维度不同的tensor结构进行相乘
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)  # [1,16,17,3]
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()  # [1,16,1,3] 骨盆
            # pred_pelvis = (pred_keypoints_3d[:, [2], :] + pred_keypoints_3d[:, [3], :]) / 2.0
            pred_keypoints_3d = pred_keypoints_3d[:, self.joint_mapper_h36m, :]  # [1,16,14,3]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis  # [1,16,14,3]

            # ----------------------------1. MPJPE----------------------------------#
            mpjpe_s = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

            # ----------------------------2. PA-MPJPE--------------------------------------------#
            S1 = pred_keypoints_3d.cpu().numpy()  # [1,14,3]
            S2 = gt_keypoints_3d.cpu().numpy() # [1,14,3]
            # 由于这里涉及到矩阵转置，因此决定都分帧计算，以免时间维度的干扰
            S1_hat = compute_similarity_transform_batch(S1, S2)  # [1,14,3]
            pampjpe_s = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1)).mean(axis=-1)

            # ----------------------------3. compute PVE-----------------------------------------#
            smpl_out = self.smpl_neutral(betas=gt_beta, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3],
                                         pose2rot=True)
            gt_vertices = smpl_out.vertices.detach().cpu().numpy()  # [1,6890,3]

            pve_s = np.sqrt(np.sum((gt_vertices - pred_vertices.detach().cpu().numpy()) ** 2, axis=2)).mean(axis=-1)

            # 加一起
            # new_gt_keypoints_3d.append(gt_keypoints_3d.unsqueeze(1))  # (1,1,14,3)
            # new_pred_keypoints_3d.append(pred_keypoints_3d.unsqueeze(1))  # (1,1,14,3)
            # new_S1_hat.append(np.expand_dims(S1_hat, axis=1))  # (1,1,14,3)
            # new_S2.append(np.expand_dims(S2, axis=1))  # (1,1,14,3)
            # new_gt_vertices.append(np.expand_dims(gt_vertices, axis=1))  # (1,1,6890,3)
            # new_pred_vertices.append(pred_vertices.unsqueeze(1))
            new_mpjpe_s.append(mpjpe_s)
            new_pampjpe_s.append(pampjpe_s)
            new_pve_s.append(pve_s)
        # 合并
        # gt_keypoints_3d = torch.cat(new_gt_keypoints_3d, dim=1)  # (1,16,14,3)
        # pred_keypoints_3d = torch.cat(new_pred_keypoints_3d, dim=1)  # (1,16,14,3)
        # S1_hat = np.concatenate(new_S1_hat, axis=1)  # (1,16,14,3)
        # S2 = np.concatenate(new_S2, axis=1)  # (1,16,14,3)
        # gt_vertices = np.concatenate(new_gt_vertices, axis=1)  # (1,16,6890,3)
        # pred_vertices = torch.cat(new_pred_vertices, dim=1)  # (1,16,6890,3)
        mpjpe = np.concatenate(new_mpjpe_s, axis=0)
        pampjpe = np.concatenate(new_pampjpe_s, axis=0)
        pve = np.concatenate(new_pve_s, axis=0)

        # ----------------------------1. MPJPE--比原来大0.03----------------------------------#
        # mpjpe = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        # ----------------------------2. PA-MPJPE--比原来大0.05--------------------------------------------#
        # pampjpe = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1)).mean(axis=-1)
        # ----------------------------3. compute PVE,原来0.09，现在13.17?-----------------------------------------#
        # pve = np.sqrt(np.sum((gt_vertices - pred_vertices.detach().cpu().numpy()) ** 2, axis=2)).mean(axis=1)
        # accel = np.mean(compute_accel(pred_keypoints_3d)) * 1000
        # accel_err = np.mean(compute_error_accel(joints_pred=pred_keypoints_3d, joints_gt=gt_keypoints_3d)) * 1000


        # to cache results
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * 5000. / (constants.IMG_RES * pred_cam[:, 0] + 1e-9)], dim=-1)
        cached_results = {'verts': pred_vertices.cpu().numpy(),
                          'cam': pred_cam_t.cpu().numpy(),
                          'rotmat': pred_rotmat.cpu().numpy(),
                          'beta': pred_shape.cpu().numpy()}
        joblib.dump(cached_results, osp.join(self.exppath, 'result', f'Pred_{self.global_step}.pt'))

        # visulize results
        if self.options.save_res:
            self.save_results(pred_vertices, pred_cam, image, batch['imgname'], batch['bbox'], mpjpe * 1000,
                              pampjpe * 1000, prefix='Pred')
        if need_feature:
            return mpjpe * 1000, pampjpe * 1000, pve * 1000, features
            # return mpjpe * 1000, pampjpe * 1000, pve * 10, features
        else:
            return mpjpe * 1000, pampjpe * 1000, pve * 1000
            # return mpjpe * 1000, pampjpe * 1000, pve * 10


if __name__ == '__main__':
    options = parser.parse_args()
    exppath = osp.join(options.expdir, options.expname)  # exps/3dpw
    # os.makedirs(exppath, exist_ok=False)
    argsDict = options.__dict__
    with open(f'{exppath}/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    adaptor = Adaptor(options)
    adaptor.excute()
