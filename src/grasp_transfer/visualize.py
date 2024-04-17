import plotly.graph_objs as go
import numpy as np
import trimesh
from matplotlib import cm
from dataio import compute_unit_sphere_transform
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
import sys, os
import plotly.offline  
repo_path = os.getenv("REPO_DIR")

def show_mesh(vertices, faces,colors=[]):
    x, y, z = zip(*vertices)
    xt, yt, zt = zip(*faces)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, 
                y=y, 
                z=z, 
                i = list(xt),
                j = list(yt),
                k = list(zt),
                colorscale='jet',
                intensity=colors,
            ),

    ])
    fig.show()
    return fig

def show_2_mesh(vertices, faces,vertices2, faces2,colors=[]):
    x, y, z = zip(*vertices)
    xt, yt, zt = zip(*faces)
    
    x2, y2, z2 = zip(*vertices2)
    xt2, yt2, zt2 = zip(*faces2)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, 
                y=y, 
                z=z, 
                i = list(xt),
                j = list(yt),
                k = list(zt),
                colorscale='jet',
                intensity=colors,
            ),
            go.Mesh3d(
                x=x2, 
                y=y2, 
                z=z2, 
                i = list(xt2),
                j = list(yt2),
                k = list(zt2),
                colorscale='jet',
                intensity=colors,
            ),            

    ])
    fig.show()
    return fig

def generateCube(size_mult = 1):
    extentss = np.array([(1,0.01,0.01),(0.01,1,0.01),(0.01,0.01,1)])
    mults1 = np.array([(0,1,0),(0,0,1),(1,0,0),])
    mults2 = np.array([(0,0,1),(1,0,0),(0,1,0),])
    mesh_list=list()
    for extent_ind,extents in enumerate(extentss):
        for i in [-1,1]:
            for j in [-1,1]:
                mesh = trimesh.primitives.Box(extents=extents*size_mult)
                mesh.apply_translation(i*0.5*size_mult*mults1[extent_ind])
                mesh.apply_translation(j*0.5*size_mult*mults2[extent_ind])
                mesh_list.append(mesh)

    return trimesh.util.concatenate(mesh_list)

def trimesh_show(np_pcd_list, color_list=None, rand_color=False, show=True):
    colormap = cm.get_cmap('brg', len(np_pcd_list))
    # colormap= cm.get_cmap('gist_ncar_r', len(np_pcd_list))
    colors = [
        (np.asarray(colormap(val)) * 255).astype(np.int32) for val in np.linspace(0.05, 0.95, num=len(np_pcd_list))
    ]
    if color_list is None:
        if rand_color:
            color_list = []
            for i in range(len(np_pcd_list)):
                color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist() + [255])
        else:
            color_list = colors
    
    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        tpcd.colors = np.tile(color_list[i], (tpcd.vertices.shape[0], 1))

        tpcd_list.append(tpcd)
    
    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    if show:
        scene.show() 

    return scene
def trimesh_show_green(np_pcd_list, color_list=None, rand_color=False, show=True):
    # colormap= cm.get_cmap('gist_ncar_r', len(np_pcd_list))
    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        tpcd.colors = np.tile(np.array([0,255,0]), (tpcd.vertices.shape[0], 1))

        tpcd_list.append(tpcd)
    
    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    if show:
        scene.show() 

    return scene

def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def get_gripper_simple_mesh(center, R, width, depth, score=1, color=None, trimesh_flag = True):
    '''
    Adapted from https://github.com/graspnet/graspnetAPI/blob/master/graspnetAPI/utils/utils.py
    
    '''
    x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score # red for high score
        color_g = 0
        color_b = 1 - score # blue for low score
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    R_correction =np.array([
        [1,0,0],
        [0,0,1],
        [0,1,0],
    ])
    vertices = np.dot(R_correction@R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
#     colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])
    if trimesh_flag:
        gripper = trimesh.Trimesh(vertices=vertices,faces=triangles,face_colors=[color_r,color_g,color_b])
    else:
        gripper = o3d.geometry.TriangleMesh()
        gripper.vertices = o3d.utility.Vector3dVector(vertices)
        gripper.triangles = o3d.utility.Vector3iVector(triangles)
        colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])
        gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def get_panda_gripper_mesh(grasp_tr):
    r = R.from_rotvec(np.array([0, -np.pi/2, 0])) # [0, -np.pi/2, 0]
    gripper_fix = np.eye(4)
    gripper_fix[:3,:3] = r.as_matrix()
    gripper_fix[:3,3] = np.array([-0.07,0,0])
    gripper_mesh = trimesh.load(repo_path + "/meshes/franka_gripper_collision_mesh.stl")
    gripper_mesh.apply_transform(gripper_fix)
    gripper_mesh.apply_transform(grasp_tr)

    return gripper_mesh

def get_coordinate_frame_mesh(scale=0.1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(scale)
    return trimesh.Trimesh(vertices=np.asarray(frame.vertices),
                        vertex_colors = np.asarray(frame.vertex_colors),faces=np.asarray(frame.triangles))

def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.concatenate([np.stack(np.mgrid[:sidelen,:sidelen], axis=-1),
                                   sidelen//2*np.ones((sidelen,sidelen,1))],axis=-1)[None,...].astype(np.float32)
    pixel_coords /= sidelen    
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 3)*1.2
    return pixel_coords

def contour_plot(model,epoch = 2000):
    xlist = np.linspace(-0.6, 0.6, 64)
    ylist = np.linspace(-0.6, 0.6, 64)
    X, Y = np.meshgrid(xlist, ylist)
    Z = list() #y.reshape(64,64)
    for index in range(8):
        with torch.no_grad():
            subject_idx = torch.Tensor([index]).squeeze().long().cuda()[None,...]
            embedding = model.get_latent_code(subject_idx)    
            samples = get_mgrid(64).cuda()
            samples.requires_grad=False
            y = model.inference(samples,embedding,epoch=epoch)['model_out'].squeeze().detach().cpu().numpy()   
            
            y[y>1]=1
            y[y<-1]=-1
            Z.append(y.reshape(64,64))
            
    fig,ax=plt.subplots(4,2,figsize=(15,25))
    for index in range(8):
        cp = ax[index//2,index%2].contourf(X, Y, Z[index],levels=[-0.4,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1])
        ax[index//2,index%2].set_xlabel(str(index))
        
        fig.colorbar(cp,ax=ax[index//2,index%2]) # Add a colorbar to a plot
    plt.show()
    
def draw_mesh_list(mesh_list,fname):
    fig = go.Figure(data = [
            go.Mesh3d(
                            x=[],y=[],z=[], 
                            i = [],j = [],k =[],
                            colorscale='jet',
                        ),

        ],
        layout=go.Layout(
                title="Interpolation Animation",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    fig.update_layout(scene = dict(
                        xaxis=dict(range=[-1, 1], autorange=False,
                                showgrid= True, zeroline= False,visible= False,  ),
                        yaxis=dict(range=[-1, 1], autorange=False,
                                showgrid= True, zeroline= False,visible= False,  ),
                        zaxis=dict(range=[-1, 1], autorange=False,
                                showgrid= True, zeroline= False,visible= False,  ),
                        )
                     )

    # Frames
    frames=list()
    for i in range(0,len(mesh_list)):
        v,t,_ = mesh_list[i]
        x, y, z = zip(*v)
        xt, yt, zt = zip(*t)

        frames.append(
            go.Frame(data= [
                go.Mesh3d(
                        x=x,y=y,z=z, 
                        i = list(xt),j = list(yt),k =list(zt),
                        color='darkblue',
                        name = 'Predicted Surface Mesh',
                        lighting=dict(ambient=0.6, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2),
                        ),    
                    ],
                    name=f'frame{i}',
                )
            )
    fig.update(frames=frames)




    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
                }


    sliders = [
        {"pad": {"b": 10, "t": 60},
         "len": 0.9,
         "x": 0.1,
         "y": 0,

         "steps": [
                     {"args": [[f.name], frame_args(0)],
                      "label": str(k),
                      "method": "animate",
                      } for k, f in enumerate(fig.frames)
                  ]
         }
            ]

    fig.update_layout(

        updatemenus = [{"buttons":[
                        {
                            "args": [None, frame_args(50)],
                            "label": "Play", 
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "Pause", 
                            "method": "animate",
                      }],

                    "direction": "left",
                    "pad": {"r": 10, "t": 20},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
        )

    fig.update_layout(sliders=sliders)

    camera = dict(
        up=dict(x=0, y=-0.707, z=0.707),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=1, z=-1.25)
    )
    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(scene_camera=camera)
    plotly.offline.plot(fig, filename=fname+'.html', auto_open=False)
    del fig
    
    