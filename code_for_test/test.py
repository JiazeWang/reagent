import torch.nn as nn
import numpy as np
import torch
import h5py
import os
import open3d as o3d
from mv_utils import PCViews
from PIL import Image
import matplotlib.pyplot as plt
if __name__ == '__main__':
    print('test for PCViews')
    """
    fname = '/data/modelnet40_ply_hdf5_2048/ply_data_train0.h5'
    f = h5py.File(fname, mode='r')
    data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
    labels = f['label'][:].flatten().astype(np.int64)
    print("data.shape:", data[0])
    print("label.shape:", labels[0])
    visdata = data[0][:,:3]
    """
    source = np.load("source.npy")
    visdata = source
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(visdata)
    o3d.visualization.draw_geometries([pcd])
    #visdata = torch.tensor(data[0][:,:3].reshape(1, -1, 3))
    visdata = torch.tensor(visdata.reshape(1, -1, 3))
    print("visdata.shape:", visdata.shape)
    pc_views = PCViews()

    img = pc_views.get_img(points=visdata).numpy()
    fig, axs = plt.subplots(2,3)
    fig.suptitle('test for six views')
    axs[0][0].imshow(img[0], interpolation='none')
    axs[0][1].imshow(img[1], interpolation='none')
    axs[0][2].imshow(img[2], interpolation='none')
    axs[1][0].imshow(img[3], interpolation='none')
    axs[1][1].imshow(img[4], interpolation='none')
    axs[1][2].imshow(img[5], interpolation='none')
    plt.show()
    #plt.imshow(img[], interpolation='none')
    #plt.show()
    #img = Image.fromarray(img[1].numpy(), 'L')
    #img.save('my.png')
    #img.show()
