import numpy as np
import torch
from os.path import join
from torch.utils.data import Dataset
import trimesh
from camera import *
import glob
import open3d as o3d
import copy
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from utility import *

def compute_unit_sphere_transform(mesh):
    """
    Reference: https://github.com/marian42/mesh_to_sdf/issues/23#issuecomment-779287297
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale

def compute_unit_sphere_transform_pcd(pcd_points):
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -np.mean(pcd_points,axis=0)
    scale = 1 / np.max(np.linalg.norm(pcd_points + translation, axis=1))
    return translation, scale

class Shape_SDF_Dataset(Dataset):
    def __init__(self, shapes_path,number_of_scenes,num_of_samples):
        """
        Shapes_path (<str>): Path to Aligned Shape Meshes
        """
        paths_to_objects = glob.glob(shapes_path)
        self.obj_list = [obj_path for obj_path in np.random.choice(paths_to_objects,number_of_scenes,replace=False)]
        self.meshes = list()
        self.sdf_points_pos = list()
        self.sdf_points_neg = list()
        self.sdf_values_pos = list()
        self.sdf_values_neg = list()
        self.pointclouds = list()
        self.pointcloud_normals = list()
        
        self.num_of_samples = num_of_samples
        self.number_of_scenes = number_of_scenes
        for obj_path in self.obj_list:
            m = trimesh.load(obj_path +'/model.obj', process=True,force='mesh')
            translation,scale = compute_unit_sphere_transform(m)
            self.meshes.append(m)
            samples, fid  = m.sample(250000, return_index=True)
            ### Reference For Normal Estimation : https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179        
            # compute the barycentric coordinates of each sample
            bary = trimesh.triangles.points_to_barycentric(
                triangles=m.triangles[fid], points=samples)
            # interpolate vertex normals from barycentric coordinates
            normals = trimesh.unitize((m.vertex_normals[m.faces[fid]] *
                                      trimesh.unitize(bary).reshape(
                                          (-1, 3, 1))).sum(axis=1))
            samples = (samples + translation)  * scale 
            self.pointclouds.append(samples)            
            self.pointcloud_normals.append(normals)            
            npz_file = np.load(join(obj_path,'sdf_values.npz'))
            points =  (npz_file['points'] + translation)  * scale 
            sdf_values = npz_file['sdf'] * scale
            self.sdf_points_pos.append(points[sdf_values>0])
            self.sdf_points_neg.append(points[sdf_values<0])   
            pos_sdf_values = sdf_values[sdf_values>0]
            neg_sdf_values = sdf_values[sdf_values<0]
            pos_sdf_values = pos_sdf_values*20
            pos_sdf_values[pos_sdf_values>=1]=0.999999999999
            neg_sdf_values = neg_sdf_values*20
            neg_sdf_values[neg_sdf_values<=-1]=-0.999999999999            
            
            self.sdf_values_pos.append(pos_sdf_values)
            self.sdf_values_neg.append(neg_sdf_values)

    def __len__(self):
        return len(self.meshes)
    def __getitem__(self,index):
        random_points_on_surface = np.random.choice(np.arange(250000),self.num_of_samples//2)
        random_points_pos = np.random.choice(np.arange(len(self.sdf_values_pos[index])),self.num_of_samples//4)
        random_points_neg = np.random.choice(np.arange(len(self.sdf_values_neg[index])),self.num_of_samples//4)
        
        
        x = torch.from_numpy(np.vstack([self.pointclouds[index][random_points_on_surface],
                       self.sdf_points_pos[index][random_points_pos],
                       self.sdf_points_neg[index][random_points_neg]])).float()

        y = {'sdf':torch.from_numpy(np.vstack([np.zeros((self.num_of_samples//2,1)),
                             self.sdf_values_pos[index][random_points_pos].reshape(-1,1),
                             self.sdf_values_neg[index][random_points_neg].reshape(-1,1)])).float(),
             'normals': torch.from_numpy(np.vstack([self.pointcloud_normals[index][random_points_on_surface],
                                                   -1*np.ones((self.num_of_samples//2,3))])).float(),
             }
        observations =  {'coords': x,
            'sdf': y['sdf'],
            'normals': y['normals'],
            'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'sdf':observations['sdf'] ,
        'normals': observations['normals']}

        return observations, ground_truth
class Shape_SDF_Dataset_Noisy(Dataset):
    def __init__(self, obj_list,number_of_scenes,num_of_samples,random_poses=None):
        """
        Shapes_path (<str>): Path to Aligned Shape Meshes
        """
        #
        self.obj_list = obj_list
        self.meshes = list()
        self.meshes_transformed = list()
        
        self.sdf_points_pos = list()
        self.sdf_points_neg = list()
        self.sdf_values_pos = list()
        self.sdf_values_neg = list()
        self.pointclouds = list()
        self.pointcloud_normals = list()
        if random_poses is None:
            se3_noise = torch.randn(number_of_scenes,6)*torch.Tensor([0.25,0.25,0.25,0.1,0.1,0.1])
            se3_noise[0,:] = 0 # Initial scene is not noisy
            random_poses = lie.se3_to_SE3(se3_noise).numpy()[:,:3,:]
        self.random_poses = random_poses
        self.num_of_samples = num_of_samples
        self.number_of_scenes = number_of_scenes
        for obj_ind, obj_path in enumerate(self.obj_list):
            m = trimesh.load(obj_path +'/model_watertight.obj', process=True,force='mesh')
            translation,scale = compute_unit_sphere_transform(m)
            self.meshes.append(m)
            m2 = m.copy()
            pose_transform = np.eye(4)
            pose_transform[:3,:] = random_poses[obj_ind]
            m2.apply_transform(pose_transform)
            self.meshes_transformed.append(m2)
            ### Reference For Normal Estimation : https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179        
            samples, fid  = m.sample(250000, return_index=True)
            # compute the barycentric coordinates of each sample
            bary = trimesh.triangles.points_to_barycentric(
                triangles=m.triangles[fid], points=samples)
            # interpolate vertex normals from barycentric coordinates
            normals = trimesh.unitize((m.vertex_normals[m.faces[fid]] *
                                      trimesh.unitize(bary).reshape(
                                          (-1, 3, 1))).sum(axis=1))
            samples = (samples + translation)  * scale 
            samples = random_poses[obj_ind] @ to_hom_np(samples).T
            normals = random_poses[obj_ind] @ to_hom_np_with0(normals).T
            self.pointclouds.append(samples.T)            
            self.pointcloud_normals.append(normals.T)            
            npz_file = np.load(join(obj_path,'sdf_values.npz'))
            points =  (random_poses[obj_ind] @ to_hom_np((npz_file['points'] + translation)  * scale).T).T
            sdf_values = npz_file['sdf'] * scale
            self.sdf_points_pos.append(points[sdf_values>0])
            self.sdf_points_neg.append(points[sdf_values<0])   
            pos_sdf_values = sdf_values[sdf_values>0]
            neg_sdf_values = sdf_values[sdf_values<0]
            pos_sdf_values = pos_sdf_values
            pos_sdf_values[pos_sdf_values>=1]=0.999999999999
            neg_sdf_values = neg_sdf_values
            neg_sdf_values[neg_sdf_values<=-1]=-0.999999999999            
            
            self.sdf_values_pos.append(pos_sdf_values)
            self.sdf_values_neg.append(neg_sdf_values)
    def __len__(self):
        return len(self.meshes)
    def __getitem__(self,index):
        random_points_on_surface = np.random.choice(np.arange(250000),self.num_of_samples//2)
        random_points_pos = np.random.choice(np.arange(len(self.sdf_values_pos[index])),self.num_of_samples//4)
        random_points_neg = np.random.choice(np.arange(len(self.sdf_values_neg[index])),self.num_of_samples//4)
        
        x = torch.from_numpy(np.vstack([self.pointclouds[index][random_points_on_surface],
                       self.sdf_points_pos[index][random_points_pos],
                       self.sdf_points_neg[index][random_points_neg]])).float()

        y = {'sdf':torch.from_numpy(np.vstack([np.zeros((self.num_of_samples//2,1)),
                             self.sdf_values_pos[index][random_points_pos].reshape(-1,1),
                             self.sdf_values_neg[index][random_points_neg].reshape(-1,1)])).float(),
             'normals': torch.from_numpy(np.vstack([self.pointcloud_normals[index][random_points_on_surface],
                                                   -1*np.ones((self.num_of_samples//2,3))])).float(),
             }
        observations =  {'coords': x,
            'sdf': y['sdf'],
            'normals': y['normals'],
            'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'sdf':observations['sdf'] ,
        'normals': observations['normals']}

        return observations, ground_truth    
    
class Part_SDF_Dataset(Dataset):
    def __init__(self, obj_list, number_of_scenes,num_of_samples):
        """
        Shapes_path (<str>): Path to Aligned Shape Meshes
        """
        self.obj_list = obj_list
        self.meshes = list()
        self.sdf_points_pos = list()
        self.sdf_points_neg = list()
        self.sdf_values_pos = list()
        self.sdf_values_neg = list()
        self.pointclouds = list()
        self.pointcloud_normals = list()
        self.sampling_RTs = np.zeros((number_of_scenes,3,4))
        self.epoch = 0
        self.num_of_samples = num_of_samples
        self.number_of_scenes = number_of_scenes
        for obj_ind in range(self.number_of_scenes):
            obj_path = self.obj_list[obj_ind]
            m = trimesh.load(obj_path +'/model_watertight.obj', process=True,force='mesh')
            translation,scale = compute_unit_sphere_transform(m)
            self.meshes.append(m)
            samples, fid  = m.sample(250000, return_index=True)
            ### Reference For Normal Estimation : https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179        
            # compute the barycentric coordinates of each sample
            bary = trimesh.triangles.points_to_barycentric(
                triangles=m.triangles[fid], points=samples)
            # interpolate vertex normals from barycentric coordinates
            normals = trimesh.unitize((m.vertex_normals[m.faces[fid]] *
                                      trimesh.unitize(bary).reshape(
                                          (-1, 3, 1))).sum(axis=1))
            samples = (samples + translation)  * scale 
            self.pointclouds.append(samples)            
            self.pointcloud_normals.append(normals)            
            npz_file = np.load(join(obj_path,'sdf_values.npz'))
            points =  (npz_file['points'] + translation)  * scale 
            sdf_values = npz_file['sdf']  #* scale
            self.sdf_points_pos.append(points[sdf_values>0])
            self.sdf_points_neg.append(points[sdf_values<0])   
            pos_sdf_values = sdf_values[sdf_values>0]
            neg_sdf_values = sdf_values[sdf_values<0]
            pos_sdf_values = pos_sdf_values
            pos_sdf_values[pos_sdf_values>=0.999999999999]=0.999999999999
            neg_sdf_values = neg_sdf_values
            neg_sdf_values[neg_sdf_values<=-0.999999999999]=-0.999999999999            
            self.sampling_RTs[obj_ind,:3,:3] = np.eye(3)
            self.sdf_values_pos.append(pos_sdf_values)
            self.sdf_values_neg.append(neg_sdf_values)

    def __len__(self):
        return self.number_of_scenes
    
    def pick_ref_frame(self):
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(self.pointclouds[0][:16000])
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(o3d_pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        picked_point = vis.get_picked_points()
        _,scale = compute_unit_sphere_transform(self.meshes[0])
        picked_point = self.pointclouds[0][picked_point[0]]/scale
        RT = np.eye(4)
        RT[:3,3] = -np.array(picked_point)# + np.array([0.05,0,0])
        return RT#@ tr
    def transform_all_pc_with_Rt(self,RT):
        self.meshes_transformed=list()
        for ind in range(len(self.pointclouds)):
            RT_copy = np.copy(RT)
            m2 = self.meshes[ind].copy()
            _,scale = compute_unit_sphere_transform(self.meshes[ind])
            m2.apply_transform(RT)
            self.meshes_transformed.append(m2)  
            RT_copy[:3,3] = RT_copy[:3,3] *scale                       
            points = RT_copy @ to_hom_np(self.pointclouds[ind]).T
            self.pointclouds[ind] = (points.T)[:,:3]
            points = RT_copy @ to_hom_np(self.sdf_points_pos[ind]).T
            self.sdf_points_pos[ind] = (points.T)[:,:3]    
            points = RT_copy @ to_hom_np(self.sdf_points_neg[ind]).T
            self.sdf_points_neg[ind] = (points.T)[:,:3]
            
    def sample_within_sphere(self,ind,refRT = np.eye(4)[:3,:],radius=0.5):
        points = refRT @ to_hom_np(np.copy(self.pointclouds[ind])).T
        points = (points.T)
        pc_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]   
        points = refRT @ to_hom_np(np.copy(self.sdf_points_pos[ind])).T
        points = (points.T)[:,:3]
        sdf_pos_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]   
        points = refRT @ to_hom_np(np.copy(self.sdf_points_neg[ind])).T
        points = (points.T)[:,:3]
        sdf_neg_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]   
        
        return pc_inds,sdf_pos_inds,sdf_neg_inds
    def set_sampling_RT(self,RTs):
        self.sampling_RTs = RTs
    def set_curr_epoch(self,epoch):
        self.epoch = epoch        
    def __getitem__(self,index):
        pc_inds,sdf_pos_inds,sdf_neg_inds = self.sample_within_sphere(index,self.sampling_RTs[index],
                                                                      radius = 0.5)
        m = self.meshes[index].copy()
        H = np.eye(4)
        H[:3,:] = self.sampling_RTs[index]
        m.apply_transform(H)
        _,scale = compute_unit_sphere_transform(m)
        del m
        
        num_of_pos_sdf = self.num_of_samples//4
        num_of_neg_sdf = self.num_of_samples//4
        if len(sdf_pos_inds)<self.num_of_samples//16:
            num_of_pos_sdf = 0
            num_of_neg_sdf = self.num_of_samples//2
        if len(sdf_neg_inds)<self.num_of_samples//16:
            num_of_neg_sdf = 0   
            num_of_pos_sdf = self.num_of_samples//2

        
        random_points_on_surface = pc_inds[np.random.choice(np.arange(len(pc_inds)),self.num_of_samples//2)]
        
        random_points_pos = sdf_pos_inds[np.random.choice(np.arange(len(sdf_pos_inds)),num_of_pos_sdf)]
        random_points_neg = sdf_neg_inds[np.random.choice(np.arange(len(sdf_neg_inds)),num_of_neg_sdf)]
        
        
        x = torch.from_numpy(np.vstack([self.pointclouds[index][random_points_on_surface],
                       self.sdf_points_pos[index][random_points_pos],
                       self.sdf_points_neg[index][random_points_neg]])).float()

        y = {'sdf':torch.from_numpy(np.vstack([np.zeros((self.num_of_samples//2,1)),
                             scale*self.sdf_values_pos[index][random_points_pos].reshape(-1,1),
                             scale*self.sdf_values_neg[index][random_points_neg].reshape(-1,1)])).float(),
             'normals': torch.from_numpy(np.vstack([self.pointcloud_normals[index][random_points_on_surface],
                                                   -1*np.ones((self.num_of_samples//2,3))])).float(),
             }
        observations =  {'coords': x,
            'sdf': y['sdf'],
            'normals': y['normals'],
            'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'sdf':observations['sdf'] ,
        'normals': observations['normals']}
        return observations, ground_truth         
    
    
class Part_Search_SDF_Dataset(Dataset):
    def __init__(self, shapes_path,number_of_poses,num_of_samples):
        """
        """
        path_to_objs = glob.glob(shapes_path)
        self.obj_path =np.random.choice(path_to_objs,1)[0]
        
        self.initial_reference = np.zeros((number_of_poses,4,4))        
        self.sampling_RTs = np.zeros((number_of_poses,3,4))
        
        self.num_of_samples = num_of_samples
        self.number_of_poses = number_of_poses
        self.number_of_scenes = number_of_poses # For using the visualization function
        m = trimesh.load(self.obj_path +'/model_watertight.obj', process=True,force='mesh')
        translation,scale = compute_unit_sphere_transform(m)
        self.mesh_base = m
        samples, fid  = m.sample(100000, return_index=True)
        ### Reference For Normal Estimation : https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179     
        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(
            triangles=m.triangles[fid], points=samples)
        # interpolate vertex normals from barycentric coordinates
        normals = trimesh.unitize((m.vertex_normals[m.faces[fid]] *
                                  trimesh.unitize(bary).reshape(
                                      (-1, 3, 1))).sum(axis=1))
        samples = (samples + translation)  * scale 
        self.pointcloud_base = samples            
        self.pointcloud_normals_base = normals            
        npz_file = np.load(join(self.obj_path,'sdf_values.npz'))
        points =  (npz_file['points'] + translation)  * scale 
        sdf_values = npz_file['sdf'] * scale

        self.sdf_points_pos_base = points[sdf_values>0]
        self.sdf_points_neg_base = points[sdf_values<0]

        pos_sdf_values = sdf_values[sdf_values>0]
        neg_sdf_values = sdf_values[sdf_values<0]    
        self.sdf_values_pos = pos_sdf_values
        self.sdf_values_neg = neg_sdf_values     
        self.sampling_RTs[:,:3,:3] = np.eye(3)
        self.initial_reference[:,:4,:4] = np.eye(4)
        _,scale = compute_unit_sphere_transform(self.mesh_base)
        
        random_points = np.copy(self.pointcloud_base[np.random.choice(np.arange(100000),number_of_poses)])
        self.random_points = random_points
        self.meshes_transformed = list()
        self.pointclouds = list()
        self.pointcloud_normals = list()        
        self.sdf_points_pos = list()
        self.sdf_points_neg = list()
        
        for pose_ind in range(number_of_poses):
            RT = self.initial_reference[pose_ind]
            RT_copy = np.copy(RT)                                   
            pos_translate = np.copy(-random_points[pose_ind])
            points = np.copy(self.pointcloud_base) + pos_translate
            points = RT_copy @ to_hom_np(points).T
            self.pointclouds.append((points.T)[:,:3])
            points = np.copy(self.sdf_points_pos_base) + pos_translate
            points = RT_copy @ to_hom_np(points).T
            self.sdf_points_pos.append((points.T)[:,:3])  
            points = np.copy(self.sdf_points_neg_base) + pos_translate
            points = RT_copy @ to_hom_np(points).T
            self.sdf_points_neg.append((points.T)[:,:3])

    def __len__(self):
        return self.number_of_poses
    
    def sample_within_sphere(self,ind,refRT = np.eye(4)[:3,:]):
        points = refRT @ to_hom_np(np.copy(self.pointclouds[ind])).T
        points = (points.T)
        pc_inds = np.where(
            np.linalg.norm(self.pointclouds[ind],axis=1)<1
            )[0]   
        points = refRT @ to_hom_np(np.copy(self.sdf_points_pos[ind])).T
        points = (points.T)[:,:3]
        sdf_pos_inds = np.where(
            np.linalg.norm(self.sdf_points_pos[ind],axis=1)<1
            )[0]   
        points = refRT @ to_hom_np(np.copy(self.sdf_points_neg[ind])).T
        points = (points.T)[:,:3]
        sdf_neg_inds = np.where(
            np.linalg.norm(self.sdf_points_neg[ind],axis=1)<1
            )[0]   
        
        return pc_inds,sdf_pos_inds,sdf_neg_inds
    def set_sampling_RT(self,RTs):
        self.sampling_RTs = RTs
    def __getitem__(self,index):
        
        
        pc_inds,sdf_pos_inds,sdf_neg_inds = self.sample_within_sphere(index,self.sampling_RTs[index])
        
        random_points_on_surface = pc_inds[np.random.choice(np.arange(len(pc_inds)),self.num_of_samples//2,replace=True),]
        random_points_pos = sdf_pos_inds[np.random.choice(np.arange(len(sdf_pos_inds)),self.num_of_samples//4,replace=True)]
        random_points_neg = sdf_neg_inds[np.random.choice(np.arange(len(sdf_neg_inds)),self.num_of_samples//4,replace=True)]
        
        
        x = torch.from_numpy(np.vstack([self.pointclouds[index][random_points_on_surface],
                       self.sdf_points_pos[index][random_points_pos],
                       self.sdf_points_neg[index][random_points_neg]])).float()

        y = {'sdf':torch.from_numpy(np.vstack([np.zeros((self.num_of_samples//2,1)),
                             self.sdf_values_pos[random_points_pos].reshape(-1,1),
                             self.sdf_values_neg[random_points_neg].reshape(-1,1)])).float(),
             'normals': torch.from_numpy(np.vstack([self.pointcloud_normals[index][random_points_on_surface],
                                                   -1*np.ones((self.num_of_samples//2,3))])).float(),
             }
        observations =  {'coords': x,
            'sdf': y['sdf'],
            'normals': y['normals'],
            'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'sdf':observations['sdf'] ,
        'normals': observations['normals']}
        return observations, ground_truth    
       
class Part_Search_SDF_DatasetCloseby(Dataset):
    def __init__(self, obj_path,number_of_poses,num_of_samples,R_init=np.eye(4)): # 
        """
        """
#        path_to_objs = glob.glob(shapes_path)
        self.obj_path = obj_path#np.random.choice(path_to_objs,1)[0]
        
        self.initial_reference = np.zeros((number_of_poses,4,4))        
        self.sampling_RTs = np.zeros((number_of_poses,3,4))
        
        self.num_of_samples = num_of_samples
        self.number_of_poses = number_of_poses
        self.number_of_scenes = number_of_poses # For using the visualization function
        m = trimesh.load(self.obj_path +'/model_watertight.obj', process=True,force='mesh')
        m.apply_transform(R_init)
        translation,scale = compute_unit_sphere_transform(m)
        self.mesh_base = m
        samples, fid  = m.sample(100000, return_index=True)
        ### Reference For Normal Estimation : https://github.com/mikedh/trimesh/issues/1285#issuecomment-880986179        
        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(
            triangles=m.triangles[fid], points=samples)
        # interpolate vertex normals from barycentric coordinates
        normals = trimesh.unitize((m.vertex_normals[m.faces[fid]] *
                                  trimesh.unitize(bary).reshape(
                                      (-1, 3, 1))).sum(axis=1))
        samples = (samples + translation)  * scale 
        self.pointcloud_base = samples            
        self.pointcloud_normals_base = normals            
        npz_file = np.load(join(self.obj_path,'sdf_values.npz'))
        points =  (R_init[:3,:]@to_hom_np((npz_file['points'] + translation)).T).T  * scale 
        sdf_values = npz_file['sdf'] #* scale

        self.sdf_points_pos_base = points[sdf_values>0]
        self.sdf_points_neg_base = points[sdf_values<0]

        pos_sdf_values = sdf_values[sdf_values>0]
        neg_sdf_values = sdf_values[sdf_values<0]
                  
        self.sdf_values_pos = pos_sdf_values
        self.sdf_values_neg = neg_sdf_values     
        self.sampling_RTs[:,:3,:3] = np.eye(3)
        self.initial_reference[:,:4,:4] = np.eye(4)
        _,scale = compute_unit_sphere_transform(self.mesh_base)
        
        
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(self.pointcloud_base[:8000])
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(o3d_pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        picked_point = vis.get_picked_points()
        picked_point = self.pointcloud_base[picked_point[0]]
        random_points = np.zeros((number_of_poses,3))
        random_points[:,:3]=picked_point
        random_points[:] + np.random.randn(number_of_poses,3) * 0.1
        self.random_points = random_points
        se3_noise = torch.randn(number_of_poses,6)*torch.Tensor([1.5,1.5,1.5,0.0,0.0,0.0])
        random_poses = lie.se3_to_SE3(se3_noise).numpy()
        self.initial_reference[:,:,:] = random_poses
        self.random_poses = random_poses
        self.meshes_transformed = list()
        self.pointclouds = list()
        self.pointcloud_normals = list()
        self.sdf_points_pos = list()
        self.sdf_points_neg = list()
        
        for pose_ind in range(number_of_poses):

            RT = self.initial_reference[pose_ind]
            RT_copy = np.copy(RT)   
            RT_copy[:3,3] = RT_copy[:3,3] / scale                       
            pos_translate = np.copy(-random_points[pose_ind])
            points = np.copy(self.pointcloud_base) + pos_translate
            points = RT @ to_hom_np(points).T
            self.pointclouds.append((points.T)[:,:3])
            points = np.copy(self.sdf_points_pos_base) + pos_translate
            points = RT @ to_hom_np(points).T
            self.sdf_points_pos.append((points.T)[:,:3])  
            points = np.copy(self.sdf_points_neg_base) + pos_translate
            points = RT @ to_hom_np(points).T
            self.sdf_points_neg.append((points.T)[:,:3])
            normals_ = RT @ to_hom_np_with0(self.pointcloud_normals_base).T
            self.pointcloud_normals.append((normals_.T)[:,:3])            
            m2 = m.copy()
            m2.apply_translation(pos_translate/scale)
            m2.apply_transform(RT_copy)
            self.meshes_transformed.append(m2)  
    def __len__(self):
        return self.number_of_poses
    
    def sample_within_sphere(self,ind,refRT = np.eye(4)[:3,:],radius = 0.5):
        points = refRT @ to_hom_np(np.copy(self.pointclouds[ind])).T
        points = (points.T)
        pc_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]   
        points = refRT @ to_hom_np(np.copy(self.sdf_points_pos[ind])).T
        points = (points.T)[:,:3]
        sdf_pos_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]   
        points = refRT @ to_hom_np(np.copy(self.sdf_points_neg[ind])).T
        points = (points.T)[:,:3]
        sdf_neg_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]   
        
        return pc_inds,sdf_pos_inds,sdf_neg_inds
    def set_sampling_RT(self,RTs):
        self.sampling_RTs = RTs
    def __getitem__(self,index):
        
        
        pc_inds,sdf_pos_inds,sdf_neg_inds = self.sample_within_sphere(index,self.sampling_RTs[index])
        m = self.mesh_base.copy()
        H = np.eye(4)
        H[:3,:] = self.sampling_RTs[index]
        m.apply_transform(H)
        _,scale = compute_unit_sphere_transform(m)
        del m
        
#         random_points_on_surface = pc_inds[np.random.choice(np.arange(len(pc_inds)),self.num_of_samples,replace=True),]

        random_points_on_surface = pc_inds[np.random.choice(np.arange(len(pc_inds)),self.num_of_samples//2,replace=True),]
        random_points_pos = sdf_pos_inds[np.random.choice(np.arange(len(sdf_pos_inds)),self.num_of_samples//4,replace=True)]
        random_points_neg = sdf_neg_inds[np.random.choice(np.arange(len(sdf_neg_inds)),self.num_of_samples//4,replace=True)]
        
        
        x = torch.from_numpy(np.vstack([self.pointclouds[index][random_points_on_surface],
                       self.sdf_points_pos[index][random_points_pos],
                       self.sdf_points_neg[index][random_points_neg]])).float()
    
        y = {'sdf':torch.from_numpy(np.vstack([np.zeros((self.num_of_samples//2,1)),
                             scale*self.sdf_values_pos[random_points_pos].reshape(-1,1),
                             scale*self.sdf_values_neg[random_points_neg].reshape(-1,1)])).float(),
             'normals': torch.from_numpy(np.vstack([self.pointcloud_normals[index][random_points_on_surface],
                                                   -1*np.ones((self.num_of_samples//2,3))])).float(),
             }
        observations =  {'coords': x,
            'sdf': y['sdf'],
            'normals': y['normals'],
            'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'sdf':observations['sdf'] ,
        'normals': observations['normals']}
        return observations, ground_truth       
class PCD_Alignment_Dataset_Random_Sampling(Dataset):
    def __init__(self, pcd_original, number_of_poses_for_each_tr, candidate_transformations,randomness_level= 0.2 ): # 
        """
        pcd_original: open3d_pointcloud
        """
        pcd = copy.deepcopy(pcd_original)
        
        translation,scale = compute_unit_sphere_transform_pcd(np.asarray(pcd.points))
        TR = np.eye(4)
        TR[:3,:3] *=scale 
        TR[:3,3] =TR[:3,:3]@translation
        pcd = pcd.transform(TR)
        trl = np.eye(4)
        pcd_points = np.asarray(pcd.points)     
        
        min_bound = np.min(pcd_points, axis=0)
        max_bound = np.max(pcd_points, axis=0) 
        
        self.scale_transformation = TR
        self.canonical_pose_transformation = trl
        
        self.number_of_tr = len(candidate_transformations)* 2
        self.number_of_poses_for_each_tr = number_of_poses_for_each_tr
        number_of_poses = self.number_of_tr*number_of_poses_for_each_tr
        self.number_of_poses=number_of_poses
        self.initial_reference = np.zeros((number_of_poses,4,4))        
        self.sampling_RTs = np.zeros((number_of_poses,3,4))
        self.num_of_samples = 1000
        translation , scale = compute_unit_sphere_transform_pcd(pcd_points)
        
        self.pointcloud_bases = list()
        self.off_surface_points_GT_pose = np.concatenate([np.random.uniform(min_bound[0]-0.3, max_bound[0]+0.3, (2000,1)),
                                                          np.random.uniform(min_bound[1]-0.3, max_bound[1]+0.3, (2000,1)),
                                                          np.random.uniform(min_bound[2]+0.00001,     max_bound[2]+0.3, (2000,1))],axis=1)
        self.off_surface_udf = cdist(self.off_surface_points_GT_pose,pcd_points).min(axis=1)
        self.off_surface_points_GT_pose = self.off_surface_points_GT_pose
        self.off_surface_udf = self.off_surface_udf
        
        self.off_surface_points_bases = list()
        self.base_transforms= list()
        for TR in candidate_transformations:
            for correction in [0,1]:
                TR_corrected = copy.deepcopy(TR)
                correction_fix = np.eye(4)
                correction_fix[:3,:3] = R.from_rotvec(np.array([correction*np.pi, 0 , 0])).as_matrix()
                TR_corrected= TR_corrected@correction_fix 
                self.base_transforms.append(TR_corrected)
                self.pointcloud_bases.append((find_inverse_RT_4x4(TR_corrected)[:3,:]@(to_hom_np(pcd_points)).T).T) 
                self.off_surface_points_bases.append((find_inverse_RT_4x4(TR_corrected)[:3,:]@(to_hom_np(self.off_surface_points_GT_pose)).T).T) 
                
        self.sampling_RTs[:,:3,:3] = np.eye(3)
        self.initial_reference[:,:4,:4] = np.eye(4)
        se3_noise = torch.randn(number_of_poses,6)*torch.Tensor([randomness_level,randomness_level,
                                                                 randomness_level,randomness_level,
                                                                 randomness_level,randomness_level])
        random_poses = lie.se3_to_SE3(se3_noise).numpy()
        scale_tr =  np.random.uniform(0.5,2,(number_of_poses,1,1))
        random_poses[:,:3,:3]= random_poses[:,:3,:3]*scale_tr
        self.initial_reference[:,:,:] = random_poses
        self.pointclouds = list()
        self.off_surface_points = list()        
        self.is_bad_index = np.zeros((number_of_poses))
        for candidate_base_ind in range(self.number_of_tr): ## pose_ind//self.number_of_poses_for_each_tr
            for candidate_pose_ind in range(self.number_of_poses_for_each_tr): ## pose_ind % self.number_of_poses_for_each_tr
                pose_ind = candidate_base_ind*self.number_of_poses_for_each_tr + candidate_pose_ind
                RT_copy = np.copy(self.initial_reference[pose_ind])
                points = np.copy(self.pointcloud_bases[candidate_base_ind])
                points = RT_copy @ to_hom_np(points).T
                self.pointclouds.append((points.T)[:,:3])
                points = np.copy(self.off_surface_points_bases[candidate_base_ind])
                points = RT_copy @ to_hom_np(points).T
                self.off_surface_points.append((points.T)[:,:3])
                            
    def __len__(self):
        return self.number_of_poses
    
    def sample_within_sphere(self,ind,refRT = np.eye(4)[:3,:]):
        radius = 0.5
        points = refRT @ to_hom_np(np.copy(self.pointclouds[ind])).T
        points = (points.T)
        pc_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]   
        points = refRT @ to_hom_np(np.copy(self.off_surface_points[ind])).T
        points = (points.T)
        offs_inds = np.where(
            np.linalg.norm(points,axis=1)<radius
            )[0]           
        return pc_inds,offs_inds
    def set_sampling_RT(self,RTs):
        self.sampling_RTs = RTs
    def __getitem__(self,index):
        candidate_id = index//self.number_of_poses_for_each_tr
        
        pc_inds,offs_inds = self.sample_within_sphere(index,self.sampling_RTs[index])
        
        if len(pc_inds) == 0:
            pc_inds=np.zeros((100),dtype=int)
        if len(offs_inds) == 0:
            offs_inds=np.zeros((100),dtype=int)
        m = self.pointcloud_bases[candidate_id].copy()
        points = (self.sampling_RTs[index] @ to_hom_np(m).T).T
        _,scale = compute_unit_sphere_transform_pcd(points)
        del m
        if len(pc_inds)<self.num_of_samples//4:
            self.is_bad_index[index] = 1        
        else:
            self.is_bad_index[index] = 0
            
        random_points_on_surface = pc_inds[np.random.choice(np.arange(len(pc_inds)),
                                                            self.num_of_samples//2,replace=True),]
        random_points_off_surface = offs_inds[np.random.choice(np.arange(len(offs_inds)),
                                                            self.num_of_samples//2,replace=True),]
        
        coords = torch.from_numpy(np.vstack([
            self.pointclouds[index][random_points_on_surface],
            self.off_surface_points[index][random_points_off_surface],
        ])).float()      
        sdf = torch.from_numpy(np.vstack([
            np.zeros((self.num_of_samples//2,1)),
            scale*self.off_surface_udf[random_points_off_surface].reshape(-1,1),
        ])).float()
        observations =  {'coords': coords,
            'sdf': sdf,
            'instance_idx':torch.Tensor([index]).squeeze().long()}
    
        ground_truth = {'sdf':observations['sdf']}
        return observations, ground_truth    