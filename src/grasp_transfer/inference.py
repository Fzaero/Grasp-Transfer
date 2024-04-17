import numpy as np
import cv2
from numpy.core.numeric import full
import open3d as o3d
from transforms3d.quaternions import mat2quat
import sys
from dataio import *
from networks import *
from torch.utils.data import DataLoader
from visualize import *
from trimesh.viewer import windowed
import time
from sdf_estimator import *
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from utility import *
from scipy.spatial.distance import cdist
import copy
import yaml
import transforms3d
from tqdm import tqdm
import pickle 

import sys
import os
grasp_transfer_path = os.getenv("GRASP_TRANSFER_SOURCE_DIR")
repo_path = os.getenv("REPO_DIR")

model = MyNet(64,latent_dim = 128, hyper_hidden_features=256,hidden_num=128)
model.to(device=torch.device('cuda:0'))
part_weights = {
    "rim": repo_path + "/outputs/rim/model",
    "handle": repo_path + "/outputs/handle/model",
    "cuboid": repo_path + "/outputs/cuboid/model"
}

def load_part_weights(part_name):
    if part_name not in part_weights.keys():
        return "Error"
    temp_ordered_dict = OrderedDict()
    weight =torch.load(part_weights[part_name])
    for key in weight:
        temp_ordered_dict[key] =weight[key]
        
    model.load_state_dict(temp_ordered_dict)#my_model
    model.eval()
    
def estimate_mesh_from_model_with_embedding(model, epoch,embedding, resolution = 64, scale = 1 ):
    with torch.no_grad():
        N=resolution
        max_batch=64 ** 3
        model.eval()

        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * scale * voxel_size) + scale *voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * scale * voxel_size) + scale *voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * scale * voxel_size) + scale *voxel_origin[0]

        num_samples = N ** 3

        samples.requires_grad = False

        head = 0

        while head < num_samples:
            sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
            samples[head : min(head + max_batch, num_samples), 3] = (
                model.inference(sample_subset,embedding,epoch)['model_out']
                .squeeze()#.squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)
        v,t,n,_ = measure.marching_cubes(sdf_values.numpy(),0.035,step_size = 1,spacing=[2/N,2/N,2/N])
        return scale*(v-1),t,n

def get_grasps_for_pcd(pcd, part_name, candidates, candidate_for_obj, return_traj=False,epochs=100):
    load_part_weights(part_name)
    with torch.no_grad():
        RT1 = np.eye(4)
        RT1[:3,:] = model.get_refine_poses(torch.tensor([0]).cuda(),False).cpu().numpy().reshape(3,4)
        
    embedding_traj = list()
    pose_traj = list()
    transformations = copy.deepcopy(candidates)
    model_transfer = MyNetTransferPCD(len(transformations)*2*candidate_for_obj, model,
                                      latent_dim = 128, embedding = torch.zeros(1,128).cuda())
    
    model_transfer.to(device=torch.device('cuda:0'))
    optim = torch.optim.Adam([
                    {'params': model_transfer.latent_codes.parameters(), 'lr': 1e-2},
                    {'params': model_transfer.se3_refine.parameters(), 'lr': 1e-2},
                    {'params': model_transfer.affine_tr.parameters(), 'lr': 1e-2}
                ],
        lr=1e-2)
    
    transfer_dataset = PCD_Alignment_Dataset_Random_Sampling(pcd,candidate_for_obj,
                                                             transformations)
    transfer_dataloader = DataLoader(transfer_dataset, shuffle=False,batch_size=32, num_workers=0, drop_last = False)
    total_steps=0
    epochs=100

    with tqdm(total=len(transfer_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            model_transfer.train()
            for step, (model_input, gt) in enumerate(transfer_dataloader):
                start_time = time.time()
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                losses = model_transfer(model_input,gt,2000)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                pbar.update(1)
                pbar.set_postfix(loss=train_loss, time=time.time() - start_time, epoch=epoch)
                total_steps += 1
            RTs,_ = model_transfer.get_refine_poses(torch.arange(0,len(transformations)*2*candidate_for_obj).cuda())
            RTs = RTs.cpu().detach().numpy()
            curr_embeddings = model_transfer.latent_codes.weight.data.cpu().detach().clone().numpy()
            embedding_traj.append(curr_embeddings)
            pose_traj.append(RTs)
            transfer_dataset.set_sampling_RT(RTs[:,:3,:4])

    final_losses = dict()
    for key_name in ['total','sdf','embeddings_constraint']:
        final_losses[key_name] = np.zeros((len(transformations)*2*candidate_for_obj))
    model_transfer.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(transfer_dataloader):
            start_time = time.time()
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            losses = model_transfer(model_input,gt,2000,reduction=False)
            for loss_name, loss in losses.items():
                final_losses[loss_name][step*32:(step+1)*32]+=loss.cpu().numpy()
                final_losses['total'][step*32:(step+1)*32]+=loss.cpu().numpy()
    best_index = np.argmin(final_losses['total'])
    correction_fix = np.eye(4)
    if best_index >len(transformations)*candidate_for_obj-1:  
        correction_fix[:3,:3] = R.from_rotvec(np.array([-np.pi, 0 , 0])).as_matrix() 
    correction_fix = correction_fix.reshape(1,4,4)
    RT2_inv = find_inverse_RT_4x4(copy.deepcopy(transfer_dataset.scale_transformation))
    RT3_inv = find_inverse_RT_4x4(copy.deepcopy(transfer_dataset.canonical_pose_transformation))
    RT4_inv = copy.deepcopy(transfer_dataset.base_transforms[best_index//candidate_for_obj])
    RT5_inv = find_inverse_RT_4x4(transfer_dataset.initial_reference[best_index])
    RT6_inv = np.array([find_inverse_RT_3x4(pose_traj[i][best_index]) for i in range(epochs)])
    mesh_transformation_traj = RT2_inv@RT3_inv@RT4_inv@RT5_inv@RT6_inv
    grasp_poses_traj = mesh_transformation_traj@RT1.reshape(1,4,4)@correction_fix
    grasp_pose_R_traj = np.array([transforms3d.affines.decompose((grasp_poses_traj[i]))[1] for i in range(epochs)])

    grasp_poses_traj[:,:3,:3] = grasp_pose_R_traj
    if not return_traj:
        return mesh_transformation_traj[-1], grasp_poses_traj[-1], embedding_traj[-1][best_index], model_transfer
    else:
        return mesh_transformation_traj, grasp_poses_traj, embedding_traj[:][best_index], model_transfer