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
# print(config_dir)
train_data_dir = config_dir['train_data_dir']
# print(train_data_dir)

train_dataset_path = []
for file in os.listdir(train_data_dir):
    train_dataset_path.append(os.path.join(train_data_dir,file))

i=0
train_dataset = []
for data_path in train_dataset_path:
    # print(data_path)
    try:
        data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
    except Exception as e:
        print("train::data_preparation::cannot load data from ",data_path,e)
    
    sample = data['surfaceSamples']
    voxel = data['Volume']
    cp = data['closestPoints']

    voxel=torch.from_numpy(voxel).float().unsqueeze(0)
    sample=torch.from_numpy(sample).float().t()
    cp=torch.from_numpy(cp).float().reshape(-1,3)
    # if i == 0:
    #     # print(voxel[0][0][0])
    #     # print("voxel = ",voxel.size(),":",voxel)
    #     # print("sample = ",sample.size(),":",sample)
    #     # print("cp = ",cp.size(),":",cp)
    # i = 1

    train_data = {
        "voxel": voxel,
        "sample": sample,
        "cp": cp
    }
    train_dataset.append(train_data)
    # p1,p2,p3,q1,q2,q3 = utils.init_planes_and_quats(train_data,param_p1,param_p2,param_p3,param_q1,param_q2,param_q3)

class VoxelDataset(Dataset):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def __len__(self):
        return len(self.train_dataset)
    
    def __getitem__(self, index):
        return self.train_dataset[index]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("train::device = ",device)

print("train::data_preparation::loading data...")
dataset = VoxelDataset(train_dataset)
dataloader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=True, num_workers=config_train["num_workers"])
print("train::data_preparation::data loaded")

net = model.PRSNet().to(device)

criterion = model.PRSNetLoss()
# criterion = criterion.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=config_train["lr"])

print("train::start training...")
for epoch in range(1,config_train["epochs"]+config_train["epochs_decay"]+1):
    losses = []
    for batch in dataloader:
        voxel = batch["voxel"].to(device)
        sample = batch["sample"].to(device)
        cp = batch["cp"].to(device)

        param_p1 = [1,0,0,0]
        param_p2 = [0,1,0,0]
        param_p3 = [0,0,1,0]
        param_q1 = [0,1,0,0]
        param_q2 = [0,0,1,0]
        param_q3 = [0,0,0,1]
        p1,p2,p3,q1,q2,q3 = net(voxel,param_p1,param_p2,param_p3,param_q1,param_q2,param_q3)
        loss = criterion(voxel,sample,cp,p1,p2,p3,q1,q2,q3)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = sum(losses) / len(losses)
    print("epoch: {} loss: ".format(epoch),avg_loss)

    if epoch > config_train["epochs"]:
        for param_group in optimizer.param_groups:
            param_group['lr'] -= config_train["lr"] / config_train["epochs_decay"]

    if epoch % config_train["save_epoch_freq"] == 0:

        file_path = os.path.join(config_dir["checkpoints_dir"], "model_epoch_{}.pth".format(epoch))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(net.state_dict(), file_path)
        print("save_epoch: {}".format(epoch))




