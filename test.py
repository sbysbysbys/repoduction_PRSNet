
import yaml
import os
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy
import utils
import model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

config_dir = config["dir"]
config_train = config["train"]
config_test = config["test"]
if_double_check = config_test["double_check"]
test_data_dir = config_dir['test_data_dir']
results_dir = config_dir['results_dir']

test_dataset_path = []
for file in os.listdir(test_data_dir):
    test_dataset_path.append(os.path.join(test_data_dir,file))

test_dataset = []
for data_path in test_dataset_path:
    try:
        data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
    except Exception as e:
        print("test::data_preparation::cannot load data from ",data_path,e)
    
    sample = data['surfaceSamples']
    voxel = data['Volume']
    cp = data['closestPoints']

    voxel=torch.from_numpy(voxel).float().unsqueeze(0)
    sample=torch.from_numpy(sample).float().t()
    cp=torch.from_numpy(cp).float().reshape(-1,3)

    test_data = {
        "voxel": voxel,
        "sample": sample,
        "cp": cp,
        "path": data_path
    }
    test_dataset.append(test_data)

class VoxelDataset(Dataset):
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, idx):
        return self.test_dataset[idx]

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

test_dataset = VoxelDataset(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=config_train["batch_size"], shuffle=False)

print("test::start testing...")

net = model.PRSNet().to(device)

checkpoint_dir = config_dir['checkpoints_dir']
checkpoint_file = "model_epoch_" + str(config_train["epochs"] + config_train["epochs_decay"]) + ".pth"
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
checkpoint = torch.load(checkpoint_path)

net.load_state_dict(checkpoint)
net.eval()

criterion = model.PRSNetLoss()
test_losses = []

check=0

name_idx = 1
for batch in test_dataloader:
    voxel = batch["voxel"].to(device)
    sample = batch["sample"].to(device)
    cp = batch["cp"].to(device)
    data_path = batch["path"]

    param_p1 = [1, 0, 0, 0]
    param_p2 = [0, 1, 0, 0]
    param_p3 = [0, 0, 1, 0]
    param_q1 = [0, 1, 0, 0]
    param_q2 = [0, 0, 1, 0]
    param_q3 = [0, 0, 0, 1]

    with torch.no_grad():
        p1, p2, p3, q1, q2, q3 = net(voxel, param_p1, param_p2, param_p3, param_q1, param_q2, param_q3)
        loss = criterion(voxel, sample, cp, p1, p2, p3, q1, q2, q3)
        test_losses.append(loss.item())

        p1 = torch.split(p1, 1, dim=0)
        p2 = torch.split(p2, 1, dim=0)
        p3 = torch.split(p3, 1, dim=0)
        q1 = torch.split(q1, 1, dim=0)
        q2 = torch.split(q2, 1, dim=0)
        q3 = torch.split(q3, 1, dim=0)
        print(len(data_path))
        voxel = torch.split(voxel, 1, dim=0)
        sample = torch.split(sample, 1, dim=0)
        cp = torch.split(cp, 1, dim=0)
        
        # 检查数据
        if check == 0:
            print("checking data......")
            print("p1 = ",p1[0])
            print("p2 = ",p2[0])
            print("p3 = ",p3[0])
            print("q1 = ",q1[0])
            print("q2 = ",q2[0])
            print("q3 = ",q3[0])
            data = {
                "voxel": voxel[0].squeeze(0),
                "sample": sample[0].squeeze(0),
                "cp": cp[0].squeeze(0)
                    }
            sp1,sp2,sp3,sq1,sq2,sq3 = utils.init_planes_and_quats(data,p1[0].squeeze(0),p2[0].squeeze(0),p3[0].squeeze(0),q1[0].squeeze(0),q2[0].squeeze(0),q3[0].squeeze(0))
            print("checking Lsd......")
            print("p1_Lsd = ",sp1.Lsd())
            print("p2_Lsd = ",sp2.Lsd())
            print("p3_Lsd = ",sp3.Lsd())
            print("q1_Lsd = ",sq1.Lsd())
            print("q2_Lsd = ",sq2.Lsd())
            print("q3_Lsd = ",sq3.Lsd())
            print("checking Lr......")
            print("Lr_plane = ",utils.LrPlane(sp1,sp2,sp3))
            print("Lr_quat = ",utils.LrQuat(sq1,sq2,sq3))
            Lsd_plane, Lsd_quat, Lr_plane, Lr_quat, La = utils.losses(data, p1[0].squeeze(0), p2[0].squeeze(0), p3[0].squeeze(0), q1[0].squeeze(0), q2[0].squeeze(0), q3[0].squeeze(0))
            print("Lsd_plane = ",Lsd_plane)
            print("Lsd_quat = ",Lsd_quat)
            print("Lr_plane = ",Lr_plane)
            print("Lr_quat = ",Lr_quat)
            print("La = ",La)
            check=1

        # 评估并保存
        idx = 0
        for path_idx in range(len(data_path)):
            alter_plane = [1,1,1,1]
            alter_quat = [1,1,1,1]
            data = {
                "voxel": voxel[idx].squeeze(0),
                "sample": sample[idx].squeeze(0),
                "cp": cp[idx].squeeze(0)
                    }
            p = [p1[idx].squeeze(0),p2[idx].squeeze(0),p3[idx].squeeze(0)]
            q = [q1[idx].squeeze(0),q2[idx].squeeze(0),q3[idx].squeeze(0)]
            if if_double_check:
                alter_plane,alter_quat = utils.double_check(data,p1[idx].squeeze(0),p2[idx].squeeze(0),p3[idx].squeeze(0),q1[idx].squeeze(0),q2[idx].squeeze(0),q3[idx].squeeze(0))
            try:
                test_data = sio.loadmat(data_path[idx], verify_compressed_data_integrity=False)
            except Exception as e:
                print("test::data_preparation::cannot load data from ",data_path,e)

            model = {'voxel':test_data['Volume'], 'vertices':test_data['vertices'], 'faces':test_data['faces']}
            for i in range(1,4):
                if(alter_plane[i]==1):
                    model['plane'+str(i)] = p[i-1].cpu().numpy()
                if(alter_quat[i]==1):
                    model['quat'+str(i)] = q[i-1].cpu().numpy()
            sio.savemat(os.path.join(results_dir, 'test_data_'+str(name_idx)+'.mat'), model)
            name_idx += 1
            idx += 1
            print("saving test_data_",name_idx)
        



# Calculate the average test loss
avg_test_loss = sum(test_losses) / len(test_losses)

print("test::average loss: ", avg_test_loss)

