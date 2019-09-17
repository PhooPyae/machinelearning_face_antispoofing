import keras
import cv2
import numpy as np 
import os

count = 0
data_path = 'data_to_patch'
for folder in os.listdir(data_path):
    if(folder == '.DS_Store'):
        continue
    folder_name = data_path+'/'+folder
    for sub_folder in os.listdir(folder_name):
        if(sub_folder == '.DS_Store'):
            continue
        sub_folder_name = folder_name+'/'+sub_folder
        for files in os.listdir(sub_folder_name):
            if(files == '.DS_Store'):
                continue
            file_name = sub_folder_name+'/'+files
            print(file_name)
            image = cv2.imread(file_name)
            convface = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            sift = cv2.xfeatures2d.SIFT_create()
            kp1,des1 = sift.detectAndCompute(convface,None)
            print('key point and destination',kp1,des1)
            print('------------------')
            m,n,q = np.shape(image)
            for i in range(0,len(kp1)):
                p1 = int(kp1[i].pt[0])
                q1 = int(kp1[i].pt[1])
                if (p1>32 and p1<(m-32)) and (q1>32 and q1<(n-32)):
                    patch = image[p1-32:p1+32,q1-32:q1+32]
                    count += 1
                    patch_name = str(count)+'.jpg'
                    cv2.imwrite('patches/'+folder+'/'+sub_folder+'/'+patch_name, patch)