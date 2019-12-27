import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

class DataLoader():
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        
        self.input_path = 'D:/DATASET06/TRAIN06/masked_input/'
        self.res_path = 'D:/DATASET06/TRAIN06/sphere/'

        self.reflectance_normal_path = 'D:/DATASET06/TRAIN06/reflectance_normal/'

    def load_data(self, batch_size=1, is_testing=False):
        imgs_A = []
        imgs_B = []
        imgs_C = []

        #forme
        #masked input!
        path_A = glob(self.input_path + '*')
        path_B = glob(self.res_path + '*')

        path_B_suf = '_sphere.png'
        path_C_suf = '_normal.png'

        batch_images = np.random.choice(path_A, size=batch_size)

        for img_path in batch_images:
            # Caution! B next A
            img_B = self.imread(img_path)
            img_A = self.imread(self.res_path + (('_'.join(img_path.split('.')[:-1]) + path_B_suf).split('\\')[1]))
            img_C = self.imread(self.reflectance_normal_path + (('_'.join(img_path.split('.')[:-1]) + path_C_suf).split('\\')[1]))

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)
            img_C = scipy.misc.imresize(img_C, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
                img_C = np.fliplr(img_C)

            # convert to binary image(img_C)(convert format for cv2 also)
            img_C = cv2.cvtColor(np.array(img_C), cv2.COLOR_RGB2GRAY) # scipy, read as RGB format
            img_C = np.array(cv2.threshold(np.array(img_C), 1, 255, cv2.THRESH_BINARY))

            imgs_A.append(img_A)
            imgs_B.append(img_B)
            imgs_C.append(img_C)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        imgs_C = np.array(imgs_C) / 255

        return imgs_A, imgs_B, imgs_C

    def load_batch(self, batch_size=1, is_testing=False):
        path_A = glob(self.input_path + '*')
        path_B = glob(self.res_path + '*')

        path_B_suf = '_sphere.png'
        path_C_suf = '_normal.png'

        self.n_batches = int(len(path_A) / batch_size)

        for i in range(self.n_batches-1):
            batch = path_A[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B, imgs_C = [], [], []
            for img in batch:
                #Caution! B next A
                img_B = self.imread(img)
                #for new dataset
                img_A = self.imread(self.res_path + (('_'.join(img.split('.')[:-1]) + path_B_suf).split('\\')[1]))
                img_C = self.imread(
                    self.reflectance_normal_path + (('_'.join(img.split('.')[:-1]) + path_C_suf).split('\\')[1]))

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)
                img_C = scipy.misc.imresize(img_C, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                        img_C = np.fliplr(img_C)

                # convert to binary image(img_C)(convert format for cv2 also)
                img_C = cv2.cvtColor(np.array(img_C), cv2.COLOR_RGB2GRAY)  # scipy, read as RGB format
                #img_C = np.array(cv2.threshold(img_C, 1, 255, cv2.THRESH_BINARY))
                img_C = cv2.threshold(img_C, 1, 255, cv2.THRESH_BINARY)[1]

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                imgs_C.append(img_C)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            imgs_C = np.array(imgs_C) / 255.

            yield imgs_A, imgs_B, imgs_C



    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
