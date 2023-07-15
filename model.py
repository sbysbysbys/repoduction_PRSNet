import numpy as np
import torch
import torch.nn as nn
import utils 

#卷积最大池编码
class CaMP(nn.Module):
    def __init__(self):
        super(CaMP, self).__init__()

        self.conv1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv5 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        # print(x.size())
        x = self.maxpool1(x)
        # print(x.size())
        x = self.relu1(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.relu5(x)
        # print(x.size())
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)

# 全连接层
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

        self.fc1 = nn.Linear(64, 32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        # print(x.size())
        x = self.relu1(x)
        x = self.fc2(x)
        # print(x.size())
        x = self.relu2(x)
        x = self.fc3(x)
        # print(x.size())
        return x

#PRSNet
class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet,self).__init__()

        self.camp = CaMP()
        self.fcp1 = FC()
        self.fcp2 = FC()
        self.fcp3 = FC()
        self.fcq1 = FC()
        self.fcq2 = FC()
        self.fcq3 = FC()

    def forward(self,x,p1,p2,p3,q1,q2,q3):
        x = self.camp(x)
        x = x.view(x.size(0),-1)
        p1 = torch.tensor(p1) + self.fcp1(x)
        p2 = torch.tensor(p2) + self.fcp2(x)
        p3 = torch.tensor(p3) + self.fcp3(x)
        q1 = torch.tensor(q1) + self.fcq1(x)
        q2 = torch.tensor(q2) + self.fcq2(x)
        q3 = torch.tensor(q3) + self.fcq3(x)

        return p1,p2,p3,q1,q2,q3

# Loss
import yaml
class PRSNetLoss:
    def __init__(self):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.cfg_model = config["model"]
        self.wr = self.cfg_model["weight"]

    def __call__(self, voxel,sample,cp, p1, p2, p3, q1, q2, q3):
        voxel = torch.split(voxel, 1, dim=0)
        sample = torch.split(sample, 1, dim=0)
        cp = torch.split(cp, 1, dim=0)
        p1 = torch.split(p1, 1, dim=0)
        p2 = torch.split(p2, 1, dim=0)
        p3 = torch.split(p3, 1, dim=0)
        q1 = torch.split(q1, 1, dim=0)
        q2 = torch.split(q2, 1, dim=0)
        q3 = torch.split(q3, 1, dim=0)
        # print("  ")
        # print(voxel[0].size())
        # print(sample[0].size())
        # print(cp[0].size())
        # print(p1[0].size())
        data = []
        losses = []
        for i in range(len(voxel)):
            # print("batch_idx = ", i+1)
            data.append({"voxel": voxel[i].squeeze(0), "sample": sample[i].squeeze(0), "cp": cp[i].squeeze(0)})
            Lsd_plane, Lsd_quat, Lr_plane, Lr_quat = utils.losses(data[i], p1[i].squeeze(0), p2[i].squeeze(0), p3[i].squeeze(0), q1[i].squeeze(0), q2[i].squeeze(0), q3[i].squeeze(0))
            loss = Lsd_plane + Lsd_quat + (Lr_plane + Lr_quat) * self.wr
            losses.append(loss.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        # print(losses.size())
        losses = torch.mean(losses)
        return losses


