import torch
from skimage import measure
import numpy as np

''' Adapted from Siren
'''
def estimate_mesh_from_model(model, epoch, subject_idx, resolution = 64, scale = 1 ):
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
        if subject_idx==-1:
            subject_idx = torch.Tensor([range(64)]).squeeze().long().cuda()[None,...]
            embedding = model.get_latent_code(subject_idx)
            embedding=embedding.mean(dim=1)
        else:
            subject_idx = torch.Tensor([subject_idx]).squeeze().long().cuda()[None,...]
            embedding = model.get_latent_code(subject_idx)
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
def estimate_points_from_model(model, epoch, subject_idx, resolution = 64, scale = 1 ):
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
        if subject_idx==-1:
            subject_idx = torch.Tensor([range(64)]).squeeze().long().cuda()[None,...]
            embedding = model.get_latent_code(subject_idx)
            embedding=embedding.mean(dim=1)
        else:
            subject_idx = torch.Tensor([subject_idx]).squeeze().long().cuda()[None,...]
            embedding = model.get_latent_code(subject_idx)
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
        return samples[np.abs(sdf_values)<0.01,:3]       
    
def estimate_mesh_traj_from_model(model, epoch, subject_list,num_of_steps = 20, resolution = 64,scale = 0.6 ):
    with torch.no_grad():
        N=resolution
        max_batch=64 ** 3
        model.eval()
        
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        mesh_list=list()
        for i in range(len(subject_list)):
            subject_idx = torch.Tensor([subject_list[i]]).squeeze().long().cuda()[None,...]
            if i == len(subject_list)-1:
                subject_idx2 = torch.Tensor([subject_list[0]]).squeeze().long().cuda()[None,...]
            else:
                subject_idx2 = torch.Tensor([subject_list[i+1]]).squeeze().long().cuda()[None,...]    
            embedding1 = model.get_latent_code(subject_idx)
            embedding2 = model.get_latent_code(subject_idx2)    
            for interp in range(num_of_steps):
                embedding = embedding1+(embedding2-embedding1)/(num_of_steps-1.0)*(interp)
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
                v,t,n,_ = measure.marching_cubes(sdf_values.numpy(),0.01,step_size = 1,spacing=[2/N,2/N,2/N])
                mesh_list.append((v-1,t,n))
        return mesh_list
    
# def estimate_mesh_traj_from_model(model, epoch, subject_idx,subject_idx2, resolution = 64,scale = 0.6 ):
#     with torch.no_grad():
#         N=resolution
#         max_batch=64 ** 3
#         model.eval()
#         subject_idx = torch.Tensor([subject_idx]).squeeze().long().cuda()[None,...]
#         subject_idx2 = torch.Tensor([subject_idx2]).squeeze().long().cuda()[None,...]
#         embedding1 = model.get_latent_code(subject_idx)
#         embedding2 = model.get_latent_code(subject_idx2)
        
        
#         # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
#         voxel_origin = [-1, -1, -1]
#         voxel_size = 2.0 / (N - 1)

#         overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
#         mesh_list=list()
#         for interp in range(50):
#             embedding = embedding1+(embedding2-embedding1)/49.0*(interp)
#             samples = torch.zeros(N ** 3, 4)

#             # transform first 3 columns
#             # to be the x, y, z index
#             samples[:, 2] = overall_index % N
#             samples[:, 1] = (overall_index.long() / N) % N
#             samples[:, 0] = ((overall_index.long() / N) / N) % N

#             # transform first 3 columns
#             # to be the x, y, z coordinate
#             samples[:, 0] = (samples[:, 0] * scale * voxel_size) + scale *voxel_origin[2]
#             samples[:, 1] = (samples[:, 1] * scale * voxel_size) + scale *voxel_origin[1]
#             samples[:, 2] = (samples[:, 2] * scale * voxel_size) + scale *voxel_origin[0]

#             num_samples = N ** 3

#             samples.requires_grad = False

#             head = 0


#             while head < num_samples:
#                 sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()[None,...]
#                 samples[head : min(head + max_batch, num_samples), 3] = (
#                     model.inference(sample_subset,embedding,epoch)['model_out']
#                     .squeeze()#.squeeze(1)
#                     .detach()
#                     .cpu()
#                 )
#                 head += max_batch

#             sdf_values = samples[:, 3]
#             sdf_values = sdf_values.reshape(N, N, N)
#             v,t,n,_ = measure.marching_cubes(sdf_values.numpy(),0.01,step_size = 1,spacing=[2/N,2/N,2/N])
#             mesh_list.append((v-1,t,n))
#         return mesh_list
