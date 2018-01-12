import os
import sys
import numpy as np
from skimage import io
from skimage import transform

path = os.getcwd()
input_path = sys.argv[1]
input_img = sys.argv[2]

# input_path = 'Aberdeen'
# input_img = '40.jpg'

img_f = []
img_names = os.listdir(input_path)
N = len(img_names)
for i in range(N):
    img = io.imread(os.path.join(input_path,img_names[i]))
    img_f.append(img.flatten())

img_f = np.array(img_f)
img_f = img_f.T
X_mean = [np.mean(img_f[i]) for i in range(len(img_f))]
X_mean = np.array(X_mean)
X_mean = X_mean.reshape(1080000,1)
U, s, V = np.linalg.svd(img_f - X_mean, full_matrices=False)

# np.save('U.npy',U)
# np.save('s.npy',s)
# np.save('V.npy',V)

target_img = io.imread(os.path.join(input_path,input_img))
target_img = target_img.flatten()
target_img = np.array(target_img)
target_img = target_img.reshape(1080000,1)
weight = np.dot(U.T,target_img-X_mean)
# weight = np.dot(img_f[:,10]-X_mean,U)

# # Average Image
# # average = np.zeros(1080000)
# # for i in range(N):
# #     average += img_f[:,i]
# # average /= N
# # average = average.reshape(600,600,3)
# # io.imshow(average)
# # io.show()

reconstruction = np.zeros(1080000)
for i in range(4):
    reconstruction = reconstruction + weight[i]*U[:,i]

reconstruction = reconstruction.reshape(1080000,1)
reconstruction = reconstruction + X_mean
reconstruction -= np.min(reconstruction)
reconstruction /= np.max(reconstruction)
reconstruction = (reconstruction * 255).astype(np.uint8)
reconstruction = reconstruction.reshape(600,600,3)
io.imsave('reconstruction.jpg',reconstruction)
