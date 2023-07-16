import numpy as np
import torch
from scipy.spatial.transform import Rotation
import yaml
# 平面对称
# ax+by+cz+d = 0
class SymmetryPlane:
    def __init__(self,data,plane):
        self.data = data
        self.voxel = data["voxel"]
        self.points = data["sample"]
        self.cp = data["cp"]

        # print("data = ",data)
        # print("SymmetryPlane::plane = ",plane.size())
        self.n = plane[0:3]
        # print("SymmetryPlane::self.n = ",self.n)
        self.len = torch.sqrt(torch.sum(self.n**2))
        # print("SymmetryPlane::self.len = ",self.len)
        self.n_unit = self.n/self.len
        self.d = plane[3]
        # print("SymmetryPlane::self.d = ",self.d)

    # 对称点
    def sympoint(self,p):
        # print(p.size())
        # print(self.n)
        sym_points = p - 2*(torch.dot(p,self.n)+self.d)/self.len * self.n_unit
        return sym_points
    
    # 对称点集
    def sympoints(self):
        distances = (torch.matmul(self.points, self.n) + self.d)/self.len
        # print("distance = ",distances[500])
        symmetric_distances = -2 * distances
        # print("symmetric_distance = ",symmetric_distances[500]) 
        symmetric_points = self.points + symmetric_distances.view(-1, 1) * self.n_unit
        return symmetric_points


    # 最短距离 # 对称距离损失
    def Lsd(self):
        # print("point = ",self.points[500])
        sympoints = self.sympoints()
        # print("sympoint = ",sympoints[500])
        # 如何判断在哪个体素中？
        in_voxels = which_voxel(sympoints)
        # print("in_voxel = ",in_voxels[500])
        min_dises = min_distance(sympoints,in_voxels,self.cp)
        # print("min_dis = ",min_dises[500])
        return torch.sum(min_dises)

# 平面的正则化损失
def LrPlane(p1,p2,p3):
    identity_matrix = torch.eye(3)
    p_array = torch.stack((p1.n_unit,p2.n_unit,p3.n_unit))
    p_array_t = torch.transpose(p_array, 0, 1)
    lr_plane = torch.matmul(p_array,p_array_t)-identity_matrix
    return frobenius_norm(lr_plane)
    

 # 轴对称
class SymmetryQuat:
    def __init__(self, data, quat):
        self.data = data
        self.voxel = data["voxel"]
        self.points = data["sample"]
        self.cp = data["cp"]

        self.quat = quat
        self.n = quat[1:4]
        self.w = quat[0]
        self.len = torch.sqrt(torch.sum(self.n**2))
        self.n_unit = self.n/self.len

    # 对称点
    def sympoint(self, p):
        q_conj = torch.conj(self.quat)
        p_quat = torch.tensor([0, p[0], p[1], p[2]], device=self.quat.device)
        sym_quat = self.quat * p_quat * q_conj
        return torch.tensor([sym_quat[1], sym_quat[2], sym_quat[3]])

    # 对称点集
    def sympoints(self):
        # print("point = ",self.points[500].item())
        rotated_points = rotate_points(self.points, self.quat)
        # print("rotated_point = ",rotated_points[500].item())
        return rotated_points


    # 最短距离 / 对称距离损失
    def Lsd(self):
        # print("point = ",self.points[500])
        sympoints = self.sympoints()
        # print("sympoint = ",sympoints[500])
        # 如何判断在哪个体素中？
        in_voxels = which_voxel(sympoints)
        # print("in_voxel = ",in_voxels[500])
        min_dises = min_distance(sympoints,in_voxels,self.cp)
        # print("min_dis = ",min_dises[500])
        return torch.sum(min_dises)
    
    def La(self):
        return self.w**2/self.len**2
        
    
import torch

# 四元数旋转点云
def rotate_points(points, quat):
    quat = torch.unsqueeze(quat, dim=0)
    point_quat = torch.cat([torch.zeros_like(points[:, :1]), points], dim=1)
    rotated_quat = quaternion_multiply(quaternion_multiply(quat, point_quat), quaternion_inverse(quat))
    rotated_points = rotated_quat[:, 1:]

    return rotated_points

# 四元数乘法
def quaternion_multiply(q1, q2):
    # print("point_quat = ",q1.size())
    # print("quat = ",q2.size())
    w2, x2, y2, z2 = torch.unbind(q2, dim=1)
    w1, x1, y1, z1 = torch.unbind(q1, dim=1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)

# 四元数的共轭
def quaternion_inverse(q):
    w, x, y, z = torch.unbind(q, dim=1)
    norm = w*w + x*x + y*y + z*z
    return torch.stack([w/norm, -x/norm, -y/norm, -z/norm], dim=1)


# 轴对称的正则化损失
def LrQuat(q1,q2,q3):
    identity_matrix = torch.eye(3)
    q_array = torch.stack((q1.n_unit,q2.n_unit,q3.n_unit))
    q_array_t = torch.transpose(q_array, 0, 1)
    lr_quat = torch.matmul(q_array,q_array_t)-identity_matrix
    return frobenius_norm(lr_quat)
    

# frobenius规范
def frobenius_norm(matrix):
    squared_sum = torch.sum(torch.abs(matrix)**2)
    norm = torch.sqrt(squared_sum)
    return norm

#在那个体素中
def which_voxel(points):
    in_voxels = points*32+16
    in_voxels = torch.floor(in_voxels)
    in_voxels = torch.clamp(in_voxels, 0, 31)
    in_voxels = in_voxels[:, 0] * 32 * 32 + in_voxels[:, 1] * 32 + in_voxels[:, 2]
    # print("which_voxel::in_voxels_size =  ", in_voxels.size())
    return in_voxels

#距离
def min_distance(points, in_voxels, cp):
    closest_points = torch.gather(cp, 0, in_voxels.unsqueeze(1).repeat(1, 3).long())
    # print("closest_point = ",closest_points[500])
    distance = torch.norm(points - closest_points, dim=1)
    return distance


#初始化三个plane和三个quat
def init_planes_and_quats(data,param_p1,param_p2,param_p3,param_q1,param_q2,param_q3):
    p1 = SymmetryPlane(data,param_p1)
    p2 = SymmetryPlane(data,param_p2)
    p3 = SymmetryPlane(data,param_p3)
    # quat1 = Rotation.from_quat(param_q1)
    # quat2 = Rotation.from_quat(param_q2)
    # quat3 = Rotation.from_quat(param_q3)
    q1 = SymmetryQuat(data,param_q1)
    q2 = SymmetryQuat(data,param_q2)
    q3 = SymmetryQuat(data,param_q3)
    return p1,p2,p3,q1,q2,q3

# 损失函数
# 转动的幅度是0的时候是什么情况?
# 是否希望转动的幅度稍微大一些
# 接近于0或者是360的时候 相当于cos值绝对值接近于一？
def losses(data,param_p1,param_p2,param_p3,param_q1,param_q2,param_q3):
    p1,p2,p3,q1,q2,q3 = init_planes_and_quats(data,param_p1,param_p2,param_p3,param_q1,param_q2,param_q3)
    # with open("config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    # cfg_model = config["model"]
    # wr = cfg_model["weight"]
    # print("wr = ",wr)

    Lsd_plane = p1.Lsd() + p2.Lsd() + p3.Lsd()
    Lsd_quat = q1.Lsd() + q2.Lsd() + q3.Lsd()
    Lr_plane = LrPlane(p1,p2,p3)
    Lr_quat = LrQuat(q1,q2,q3)
    La = q1.La() + q2.La() + q3.La()

    return Lsd_plane,Lsd_quat,Lr_plane,Lr_quat,La

# 删除一些多余的结果
def double_check(data,param_p1,param_p2,param_p3,param_q1,param_q2,param_q3):
    p1,p2,p3,q1,q2,q3 = init_planes_and_quats(data,param_p1,param_p2,param_p3,param_q1,param_q2,param_q3)
    p = [p1,p2,p3]
    q = [q1,q2,q3]

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    cfg_model = config["model"]
    dc = cfg_model["dc"]
    da = cfg_model["da"]
    # print("dc = ",dc)
    min_angle = cfg_model["min_angle"]
    # print("min_angle = ",min_angle)

    angle_p12 = lines_angle(p1.n,p2.n)
    angle_p13 = lines_angle(p1.n,p3.n)
    angle_p23 = lines_angle(p1.n,p2.n)
    alter_plane = alter(p1,p2,p3,angle_p12,angle_p13,angle_p23,min_angle)

    angle_q12 = lines_angle(q1.quat[1:3],q2.quat[1:3])
    angle_q13 = lines_angle(q1.quat[1:3],q3.quat[1:3])
    angle_q23 = lines_angle(q2.quat[1:3],q3.quat[1:3])
    alter_quat = alter(q1,q2,q3,angle_q12,angle_q13,angle_q23,min_angle)

    for i in range(1,4):
        if alter_plane[i] == 1:
            if p[i-1].Lsd() > dc:
                alter_plane[i] = 0
        if alter_quat[i] == 1:
            if q[i-1].Lsd() > dc or q[i-1].La() > da:
                alter_quat[i] = 0
    print(p1.Lsd(),p2.Lsd(),p3.Lsd())
    print(q1.Lsd(),q2.Lsd(),q3.Lsd())
    print("alter_plane = ",alter_plane)
    print("alter_quat = ",alter_quat)
    
    return alter_plane,alter_quat
        


# 夹角
def lines_angle(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# 筛选相邻成员
def alter(s1,s2,s3,angle_s12,angle_s13,angle_s23,min_angle):
    alter_res = [1,1,1,1]
    angles = [angle_s12,angle_s13,angle_s23]
    s = [s1,s2,s3]
    for i in range(1,4):
        for j in range(i+1,4):
            if alter_res[i] == 1 and alter_res[j] == 1:
                if abs(angles[i+j-3]-90) > 90 - min_angle:
                    if s[i-1].Lsd() > s[j-1].Lsd():
                        alter_res[i] = 0
                    else:
                        alter_res[j] = 0
    
    return alter_res



#  pip install tensorflow==2.4.1 tensorboard==2.4.1 torch==1.7.1 torchsummary==1.5.1 scipy==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/



    





    
