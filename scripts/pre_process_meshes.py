
from os import listdir,mkdir
from os.path import isfile, join, isdir
import subprocess

from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np
import glob

def compute_unit_sphere_transform(mesh):
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale
meshes_path = "chairs/*"
## READING THE FILES
mesh_files_path = glob.glob(meshes_path) 
#print(mesh_files_path)
for mesh_path in mesh_files_path:
    if True:#not isfile(mesh_path+"/sdf_values.npz"):
        print(mesh_path)
        mesh = trimesh.load(join(mesh_path,'model_watertight.obj'), process=True,force='mesh')
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=25000, scan_count=100, scan_resolution=400,sign_method='normal')
        # colors = np.zeros(points.shape)
        # colors[sdf < 0, 2] = 1
        # colors[sdf > 0, 0] = 1
        # cloud = pyrender.Mesh.from_points(points, colors=colors)
        # scene = pyrender.Scene()
        # scene.add(cloud)
        # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
        translation, scale = compute_unit_sphere_transform(mesh)        
        points = (points / scale) - translation
        sdf /= scale    
        np.savez(join(mesh_path,"sdf_values"), points=points, sdf=sdf)