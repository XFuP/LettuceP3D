import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import argparse


def get_distance(point_cloud):
    #point_cloud: Nx3
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))
    print("密度为：", np.mean(pcd.compute_nearest_neighbor_distance()))


def cluster_dbscan(point_cloud, allToCai_density, min_points=50):
    
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(point_cloud))
    # o3d.visualization.draw_geometries([pcd])
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:

    eps = np.mean(pcd.compute_nearest_neighbor_distance()) * allToCai_density
    labels = np.array(pcd.cluster_dbscan(eps, min_points)) # eps:聚类的邻域距离；min_points：聚类的最小点数             

    # 可视化聚类结果
    if False:
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd], window_name="点云密度聚类",
                                        height=480, width=600,
                                        mesh_show_back_face=0)
    return labels

def group_deal(in_path, out_path, allToCai_density):
    files = glob.glob(os.path.join(in_path, '*.txt'))
    for file in tqdm(files):
        data = np.loadtxt(file)
        labels = cluster_dbscan(data[:,:3], allToCai_density)

        max_label = labels.max()
        
        file_name = os.path.basename(file)[:-4]
        this_out_path = os.path.join(out_path, file_name)
        os.makedirs(this_out_path, exist_ok=True)
        for i in range(max_label + 1):
            ind = np.where(labels == i)[0]
            clusters_data = data[ind]
            out_file_name = file_name + f"-{str(i+1)}.txt"

            np.savetxt(os.path.join(this_out_path, out_file_name), clusters_data, fmt="%f %f %f %d %d %d")


if __name__ == '__main__':
    
    # data = np.loadtxt(r"E:\生菜处理数据\现有苗期\lasdata\n个盆无粘连\1000-1001-1002_a.txt", dtype=np.float32)
    # get_distance(data[:, :3])

    parser = argparse.ArgumentParser('')
    parser.add_argument('--in_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--allToCai_density', type=float)
    args = parser.parse_args()

    file_path = args.in_path
    out_path = args.out_path
    allToCai_density = args.allToCai_density

    group_deal(file_path, out_path, allToCai_density)

