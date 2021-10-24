import numpy as np
import imageio
coor = np.load("t_true_coordinatesnew.npy")[4].astype(int)
print(coor.shape, coor.max(), coor.min())
picture = np.zeros([128, 128]).reshape(-1)
print(picture.shape)
import cupy
num = 0
for i in coor[:100]:
    picture[int(i)]=255
#picture = picture.reshape([128, 128])
#np.add.at(picture, coor, 255);
picture = picture.astype(np.uint8).reshape([128, 128])
imageio.imwrite('c4.jpg', picture)
