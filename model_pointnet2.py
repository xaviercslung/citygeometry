from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
from pointnet2_utils import *

MODELS_EXT = '.dms'

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
#     print('----', xyz.shape)
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=True)
        return new_points

class PointNetEncoder(nn.Module):
    def __init__(self, k=128, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.k = k
        self.sa1 = PointNetSetAbstraction(npoint=32, radius=0.2, nsample=8, in_channel=5 + 5, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=16, radius=0.4, nsample=8, in_channel=128 + 5, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=8, radius=0.8, nsample=8, in_channel=256 + 5, mlp=[256, 256, 512],
                                          group_all=True)
        self.fp3 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = torch.nn.Conv1d(128, 100, 1)
        self.conv2 = torch.nn.Conv1d(100, 50, 1)
        self.conv3 = torch.nn.Conv1d(50, 25, 1)
        self.conv4 = torch.nn.Conv1d(25, self.k, 1)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)
        self.bn3 = nn.BatchNorm1d(25)

    def forward(self, x):
        B, _, _ = x.shape
        l0_points = x
        l0_xyz = x[:, :, :]

        batchsize = x.size()[0]
        n_pts = x.size()[2]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)

        x = x.view(batchsize, n_pts, self.k)
        return x

class MLP_Decoder(nn.Module):
    def __init__(self, n_feature=128, M=20):
        super(MLP_Decoder, self).__init__()
        self.n_feature = n_feature
        self.M = M
        # self.M = get_MLP_layers((in_channel, n_feature, n_feature*2, out_channel))
        self.fc_1 = nn.Linear(n_feature, n_feature)
        self.fc_2 = nn.Linear(n_feature, n_feature // 2)
        self.fc_3 = nn.Linear(n_feature // 2, 13 * M)

    def get_parameter(self, y):
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope = torch.chunk(
            y, 13, 2)

        pi = F.softmax(pi)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        sigma_a = torch.exp(sigma_a)
        sigma_b = torch.exp(sigma_b)
        sigma_slope = torch.exp(sigma_slope)
        rho_xy = torch.tanh(rho_xy)
        rho_ab = torch.tanh(rho_ab)

        mu_x = torch.sigmoid(mu_x)
        mu_y = torch.sigmoid(mu_y)
        mu_a = torch.sigmoid(mu_a)
        mu_b = torch.sigmoid(mu_b)
        mu_slope = torch.sigmoid(mu_slope)

        return [pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope]

    def forward(self, feature):
        y = self.fc_1(feature)
        y = F.relu(y)
        y = self.fc_2(y)
        y = F.relu(y)
        y = self.fc_3(y)
        parameter = self.get_parameter(y)

        return parameter


class SaveableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save_to_drive(self, name=None):
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)
        name = name if name is not None else self.DEFAULT_SAVED_NAME
        torch.save(self.state_dict(), os.path.join(self.MODELS_DIR, name + MODELS_EXT))

    def load_from_drive(model, name=None, model_dir=None, **kwargs):
        name = name if name is not None else model.DEFAULT_SAVED_NAME
        loaded = model(**kwargs)
        loaded.load_state_dict(torch.load(os.path.join(model_dir, name + MODELS_EXT)))
        loaded.eval()
        return loaded


class pointnetbaseline(SaveableModule):
    def __init__(self, device, M=20, model_folder='log', save_name='baseline', n_feature=128):
        super(pointnetbaseline, self).__init__()

        self.n_feature = n_feature
        self.M = M
        self.encoder = PointNetEncoder(k=n_feature)
        self.decoder = MLP_Decoder(n_feature, M)

        self.MODELS_DIR = model_folder
        self.DEFAULT_SAVED_NAME = save_name
        self.device = device

    def make_target(self, batch):

        dx = torch.stack([batch[:, 0, :]] * self.M, 1).transpose(2, 1)
        dy = torch.stack([batch[:, 1, :]] * self.M, 1).transpose(2, 1)
        da = torch.stack([batch[:, 2, :]] * self.M, 1).transpose(2, 1)
        db = torch.stack([batch[:, 3, :]] * self.M, 1).transpose(2, 1)
        dslope = torch.stack([batch.data[:, 4, :]] * self.M, 1).transpose(2, 1)

        return [dx, dy, da, db, dslope]

    def bivariate_normal_pdf(self, point, parameter):
        dx, dy, da, db, dslope = [point[i] for i in range(len(point))]
        _, mu_x, mu_y, sigma_x, sigma_y, rho_xy, mu_a, mu_b, sigma_a, sigma_b, rho_ab, mu_slope, sigma_slope = [
            parameter[i] for i in range(len(parameter))]

        index_1 = torch.isnan(mu_x).any()
        index_2 = torch.isnan(mu_y).any()
        index_3 = torch.isnan(mu_a).any()
        index_4 = torch.isnan(mu_b).any()
        index_5 = torch.isnan(mu_slope).any()

        if index_1:
            print("mu_x")
            print(mu_x)
        if index_2:
            print("mu_y")
            print(mu_y)
        if index_3:
            print("mu_a")
            print(mu_a)
        if index_4:
            print("mu_b")
            print(mu_b)
        if index_5:
            print("mu_slope")
            print(mu_slope)

        pi = torch.asin(torch.tensor(1.)) * 2

        z_x = ((dx - mu_x) / sigma_x) ** 2
        z_y = ((dy - mu_y) / sigma_y) ** 2
        z_xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
        Z_xy = z_x + z_y - 2 * rho_xy * z_xy
        exp_xy = torch.exp(-Z_xy / (2 * (1 - rho_xy ** 2)))
        norm_xy = 2 * pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)
        result_xy = exp_xy / norm_xy

        z_a = ((da - mu_a) / sigma_a) ** 2
        z_b = ((db - mu_b) / sigma_b) ** 2
        z_ab = (da - mu_a) * (db - mu_b) / (sigma_a * sigma_b)
        Z_ab = z_a + z_b - 2 * rho_ab * z_ab
        exp_ab = torch.exp(-Z_ab / (2 * (1 - rho_ab ** 2)))
        norm_ab = 2 * pi * sigma_a * sigma_b * torch.sqrt(1 - rho_ab ** 2)
        result_ab = exp_ab / norm_ab

        z_slope = ((dslope - mu_slope) / sigma_slope) ** 2
        exp_slope = torch.exp(-z_slope / 2)
        norm_slope = torch.sqrt(2 * pi) * sigma_slope
        result_slope = exp_slope / norm_slope

        return result_xy, result_ab, result_slope

    def reconstruction_loss(self, point, parameter):

        pdf_xy, pdf_ab, pdf_slope = self.bivariate_normal_pdf(point, parameter)

        batch_size = pdf_xy.shape[0]
        N_points = pdf_xy.shape[1]

        pi = parameter[0]

        LS_xy = -torch.sum(torch.log(1e-5 + torch.sum(pi * pdf_xy, 2)))
        LS_ab = -torch.sum(torch.log(1e-5 + torch.sum(pi * pdf_ab, 2)))
        LS_slope = -torch.sum(torch.log(1e-5 + torch.sum(pi * pdf_slope, 2)))

        LS = (LS_xy + LS_ab + LS_slope) / (batch_size * N_points)

        return LS

    def forward(self, X):
        point = self.make_target(X.float())
        x = self.encoder(X.float())
        parameter = self.decoder(x)
        loss = self.reconstruction_loss(point, parameter)
        return loss

    def loss_on_loader(self, loader, device):
        # calculate loss on all data
        total = 0.0
        num = 0
        with torch.no_grad():
            for (i, inputs) in enumerate(loader):
                inputs = inputs.to(device)
                inputs = inputs.float()

                loss = self.forward(inputs)
                total += loss
                num += 1
        return total / num