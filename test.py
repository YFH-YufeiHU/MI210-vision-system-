import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main():
    # fileName = './autonomousVehicle.hdf5'
    # with h5py.File(fileName,'r')as f:
    #     img = f.get('images')[20]
    #     print(img)
    #     print(list(f.keys()))
    #     dataset = f['images']
    #     print(dataset.shape)
    #     print(dataset.dtype)
    #     print(img.shape)
    #
    # #plt.gray()
    # plt.imshow(img,cmap='gray')
    # plt.show()
    # A = np.array([[0,1],[1,3]])
    # print((A==1).nonzero())
    # print(A[0][0])
    # print(A[[0,1],[1,1]])
    # A = np.zeros((96,32*3))
    # print(A.shape)
    # A=np.reshape(A,(3,3,32,32))
    #
    # for i in range(3):
    #     for j in range(3):
    #         print(A[i,j].shape)
    # A=np.zeros((3,))
    # print(A)
    # A=np.array([[1,2],[3,4]])
    # print((A-1)/A)
    # print(np.where(A>2,A,0))
    # print(A.T)
    A = np.array([[[1,1],[1,1]],
                  [[2,2],[2,2]],
                  [[3,3],[3,3]]])
    print(A.shape)
    print(np.mean(A[0,:,:]))
    print(A[1,:,:])
    # print(np.repeat)
    # print(np.mean(A,axis=1))
    # print(np.mean(A,axis=3))
    # print(np.mean(A))
    # #
    # A = []
    # B = np.array([[1,2],[3,4]])
    # C = np.array([[2,1],[5,6]])
    # # print(np.mean(B,axis=0))
    # # print(np.mean(B,axis=1))
    # print(B)
    # print(np.append(A,C).reshape((2,2)))


if __name__=='__main__':
    main()



