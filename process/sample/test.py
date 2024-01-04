target_num_points = 4096
glb_file = '3b870dc505c54074bf60e058fc3eb6d4.glb' #'821feac882454ddf930600c8e94f3184.glb' #'0000ecca9a234cae994be239f6fec552.glb' # '0002e50309b44e409c96f440202d90b3.glb' #'0002c6eafa154e8bb08ebafb715a8d46.glb'
# 1. 从顶点采样，保存所有的顶点和顶点的颜色，然后对保存的点进行下采样
'''
import trimesh
import numpy as np 
import matplotlib.pyplot as plt

def downsample_vertices(vertices, colors, target_num_points):
    num_vertices = len(vertices)
    if num_vertices <= target_num_points:
        return vertices, colors
    # 计算下采样的步长
    step = num_vertices // target_num_points
    
    # 使用更均匀的下采样方法
    sampled_indices = np.linspace(0, num_vertices - 1, target_num_points, dtype=int)
    sampled_vertices = vertices[sampled_indices]
    sampled_colors = colors[sampled_indices]
    
    return sampled_vertices, sampled_colors


file_path = f'/mnt/data_sdb/obj/glbs/000-000/{glb_file}' #   95b7698c86664e049720b60c999125f7.glb
mesh = trimesh.load(file_path, force='mesh')
print(dir(mesh.visual))
vertices = mesh.vertices
colors = mesh.visual.to_color().vertex_colors
print(f"vertices.shape:{vertices.shape}\n colors:{colors.shape}")
sampled_vertices, sampled_colors = downsample_vertices(vertices, colors, target_num_points)
sampled_colors_norm = sampled_colors / 255.0
combined_data = np.hstack((sampled_vertices, sampled_colors_norm))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sampled_vertices[:, 0], sampled_vertices[:, 1], sampled_vertices[:, 2], c=sampled_colors_norm, s=2)
ax.set_title(f"Point Cloud Visualization")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
plt.savefig(f'./figs/{glb_file}_downsample_{target_num_points}.png', dpi=300)  
'''

'''
# 2. 从面采样，每个采样点的颜色取决于距离最近的那个顶点
import trimesh
from scipy.spatial import cKDTree  # 导入 cKDTree
import matplotlib.pyplot as plt

processed_glb = []
no_color_glb = []

file_path = f'/mnt/data_sdb/obj/glbs/000-000/{glb_file}'  
mesh = trimesh.load(file_path, force='mesh')
print(dir(mesh.visual))
point, face_index = trimesh.sample.sample_surface(mesh, count=target_num_points, face_weight=None, sample_color=False, seed=0)
colors_for_points_norm = None
print(f'point.shape:{point.shape}')

if hasattr(mesh.visual, 'face_colors'):
    print(f'face_colors')
    # point, face_index, color = trimesh.sample.sample_surface(mesh, count=2048, face_weight=None, sample_color=True, seed=0)
    face_colors = mesh.visual.face_colors
    print(f'face_colors:{face_colors.shape}\n{face_colors}')
    sampled_point_colors = face_colors[face_index]
    # colors_for_points_norm = sampled_point_colors / 255.0
    print(f'sampled_point_colors:{sampled_point_colors.shape}\n{sampled_point_colors}')
    processed_glb.append(glb_file)

elif hasattr(mesh.visual, 'to_color'):
    print(dir(mesh.visual.to_color()))
    print(f'to_color')

    sampled_point_colors = mesh.visual.to_color().vertex_colors
    # vertices_per_face = mesh.faces[face_index]
    print(f'vertex_colors.shape:{sampled_point_colors.shape}\n sampled_point_colors: {sampled_point_colors}')

    point_tree = cKDTree(mesh.vertices)
    closest_vertices = point_tree.query(point)[1]
    print(f"closest_vertices:{closest_vertices}")

    colors_for_points = sampled_point_colors[closest_vertices]
    # colors_for_points_norm = colors_for_points / 255.0

    print(f"point:{point.shape}\n face_index:{len(face_index)}\n")
    print(f"colors_for_points:{colors_for_points.shape}\n")
    
    processed_glb.append(glb_file)    
    
    
else:
    no_color_glb.append(glb_file)

colors_for_points_norm = colors_for_points / 255.0
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point[:, 0], point[:, 1], point[:, 2], c=colors_for_points_norm, s=2)
ax.set_title(f"Point Cloud Visualization")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
plt.savefig(f'./figs/{target_num_points}/{glb_file}_sampleface.png', dpi=300)  

'''



# 读取glb文件，判断里面包含哪些信息
'''
import trimesh
# 读取GLB文件
file_path = '/mnt/data_sdb/obj/glbs/000-000/95b7698c86664e049720b60c999125f7.glb'  # 替换为你的GLB文件路径
mesh = trimesh.load(file_path)
# 检查是否存在颜色信息
if hasattr(mesh, 'visual') and 'vertex' in mesh.visual:
    print("GLB文件包含颜色信息")
elif hasattr(mesh, 'vertices'):
    print("GLB文件包含顶点信息")    
elif hasattr(mesh, 'vertex_colors'):
    print("GLB文件包含顶点颜色信息")
elif hasattr(mesh, 'face_colors'):
    print("GLB文件包含面颜色信息")
else:
    print("GLB文件不包含颜色信息")   
'''






# 使用trimesh采样
'''
mesh = trimesh.load(glb_file_path)
points, index = trimesh.sample.sample_surface(mesh, num_points)
colors = np.array(mesh.visual.vertex_colors[index]) if hasattr(mesh.visual, 'vertex_colors') else None
'''

# 调用objaverse打印uid和annoations

import objaverse
print("prepare the point clouds ...")
uids = objaverse.load_uids()
print(uids[:10])
annoations = objaverse.load_annotations(uids[:10])
print(annoations[uids[0]]['name'])
print(annoations[uids[0]]['tags'])

# 用open3d
'''
mesh = o3d.io.read_triangle_mesh(glb_file_path)
# pcd = mesh.sample_points_uniformly(number_of_points=num_points)
pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=3)
points = np.asarray(pcd.points)
# print("points:", points)
'''


                        

'''
#    color of mesh
if mesh.has_vertex_colors():
    pcd.colors = o3d.utility.Vector3dVector(mesh.vertex_colors)
    colors = np.asarray(pcd.colors)
    print("colors:", colors)
else:
    print("there is no vertex_colors attribute")
''' 
'''
# plt draw the point clouds
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
plt.tight_layout()
plt.savefig(f'{glb_file}_point_cloud_2.png')                    
'''
    
    
    