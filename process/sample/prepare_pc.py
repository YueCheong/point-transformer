import os
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree
import objaverse
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt


num_points = 4096
glbs_folder = f'/mnt/data_sdb/obj/glbs'  
save_folder = f'/mnt/data_sdb/obj/pcs_{num_points}'

start_time = datetime.now()
for folder_id in os.listdir(glbs_folder):
    glb_path = os.path.join(glbs_folder, folder_id) #/mnt/data_sdb/obj/glbs/000-000
    save_path = os.path.join(save_folder, folder_id) #/mnt/data_sdb/obj/pcs_2048/000-000
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
            
    if os.path.isdir(glb_path):
        for glb_file in os.listdir(glb_path):  
                      
            npz_filename = os.path.splitext(glb_file)[0] + '.npz'    # save the point cloud as .npz file
            npz_path = os.path.join(save_path, npz_filename)  
                              
            if not os.path.exists(npz_path):           
                if glb_file.endswith('.glb'):
                    glb_file_path = os.path.join(glb_path, glb_file)
                    print(f"--process the glb file: {glb_file_path}")
                    try:
                        mesh = trimesh.load(glb_file_path, force='mesh')
                        points, face_indices = trimesh.sample.sample_surface(mesh, count=num_points, face_weight=None, sample_color=False, seed=0)
                        point_colors = None
                        
                        if hasattr(mesh.visual, 'face_colors'):
                            try:
                                face_colors = mesh.visual.face_colors
                                point_colors = face_colors[face_indices]
                            except Exception as e:
                                print(f"**An error in face_colors processing: {e}")
                                point_colors = np.full((num_points, 4), 255)
                        elif hasattr(mesh.visual, 'to_color'): 
                            try:
                                vertex_colors = mesh.visual.to_color().vertex_colors
                                point_tree = cKDTree(mesh.vertices)
                                closest_vertices = point_tree.query(points)[1]
                                point_colors = vertex_colors[closest_vertices]
                            except Exception as e:
                                print(f"**An error occurred in to_color processing: {e}")
                                point_colors = np.full((num_points, 4), 255)
                        else:
                            print(f'the {glb_file} mesh has no face_color and to_color ...')
                            with open('/home/hhfan/code/pc/process/sample/no_color_glb_2.txt', 'a') as f:
                                f.write(f'{glb_file_path}\n')
                            point_colors = np.full((num_points, 4), 255)                            

                        point_colors_norm = point_colors / 255.0
                        combined_points_colors = np.hstack((points, point_colors_norm))
                                      
                        '''
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors_norm, s=2)
                        ax.set_title(f'{glb_file} Visualization')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        plt.show()
                        plt.savefig(f'/home/hhfan/code/pc/process/sample/figs/{num_points}/{glb_file}_sampleface.png', dpi=300)
                        '''                        
                        np.savez(npz_path, points=combined_points_colors)
                        print(f"!!saved npz_path: {npz_path}")
                       
                    except Exception as e:
                        print(f"**An error occurred while processing the file: {e}")
                        with open('/home/hhfan/code/pc/process/sample/failed_glb_2.txt', 'a') as f:
                            f.write(f'{glb_file_path}\n')
            else:
                print(f"there exists {npz_path}, skip the point cloud sampling process!") 

end_time = datetime.now()
exe_time = end_time - start_time
print(f'the execution time is: {exe_time}')









