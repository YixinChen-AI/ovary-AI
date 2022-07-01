import cv2,os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(path='./data',mode='train',verbose=False):
    types_dict = {'恶性':'malign', '良性':'benign', '交界性':'borderline'}
    types = os.listdir(path)
    imgs,masks,type_labels,pathology_labels = [],[],[],[]
    name = []
    for i in types:
        dirnames = os.listdir(os.path.join(path,i))
        for d in dirnames:
            tmp_path = os.path.join(path,i,d,'Images')
            files = os.listdir(tmp_path)
            jpg = [f for f in files if '.jpg' in f]
            for j in jpg:
                jpg_path = os.path.join(tmp_path,j)
                img = cv2.imread(jpg_path)
                try:
                    js = json.load(open(jpg_path.replace('.jpg','.json'),'r'))
                except:
                    print(jpg_path)
                    break
                if len(img.shape)==2:    
                    mask = np.zeros_like(img)
                elif len(img.shape) ==3:
                    mask = np.zeros_like(img[:,:,0])
                for shape in js['shapes']:
#                     print(shape['label'])
                    cv2.fillPoly(mask,[np.int32(shape['points'])],color=255)

                if verbose:
                    plt.figure(figsize=(10,5))
                    plt.subplot(1,2,1);plt.imshow(img,cmap='gray')
                    plt.subplot(1,2,2);plt.imshow(mask,cmap='gray')
                    plt.title(types_dict[i])
                    plt.show()

                imgs.append(img)
                masks.append(mask)
                type_labels.append(types_dict[i])
                pathology_labels.append(0)
                name.append(d)
    return imgs,masks,type_labels,pathology_labels,name

def load_data_detection(path='./data',mode='train',verbose=False):
    types_dict = {'恶性':'exist', '良性':'exist', '交界性':'exist','全切组':'no','正常组':'no'}
    types = os.listdir(path)
    imgs,masks,type_labels,pathology_labels = [],[],[],[]
    name = []
    for i in types:
        dirnames = os.listdir(os.path.join(path,i))
        for d in dirnames:
            tmp_path = os.path.join(path,i,d,'Images')
            files = os.listdir(tmp_path)
            jpg = [f for f in files if '.jpg' in f]
            for j in jpg:
                jpg_path = os.path.join(tmp_path,j)
                img = cv2.imread(jpg_path,0)
                
                mask = np.zeros_like(img)
                if types_dict[i] == 'exist':
                    try:
                        js = json.load(open(jpg_path.replace('.jpg','.json'),'r'))
                    except:
                        print(jpg_path)
                    for shape in js['shapes']:
                        cv2.fillPoly(mask,[np.int32(shape['points'])],color=255)
                
                if verbose:
                    plt.figure(figsize=(10,5))
                    plt.subplot(1,2,1);plt.imshow(img,cmap='gray')
                    plt.subplot(1,2,2);plt.imshow(mask,cmap='gray')
                    plt.title(types_dict[i])
                    plt.show()
                
                imgs.append(img)
                masks.append(mask)
                type_labels.append(types_dict[i])
                pathology_labels.append(0)
                name.append(d)
    return imgs,masks,type_labels,pathology_labels,name

def load_data_pathology(path='./data',mode='train',verbose=False):
    types_dict = {'上皮':0, '生殖':1, '性索间质':2,'炎症':3,'巧囊':4}
    types = os.listdir(path)
    imgs,masks,type_labels,pathology_labels = [],[],[],[]
    name = []
    for i in types:
        dirnames = os.listdir(os.path.join(path,i))
        for d in dirnames:
            tmp_path = os.path.join(path,i,d,'Images')
            files = os.listdir(tmp_path)
            jpg = [f for f in files if '.jpg' in f]
            for j in jpg:
                jpg_path = os.path.join(tmp_path,j)
                img = cv2.imread(jpg_path)
                try:
                    js = json.load(open(jpg_path.replace('.jpg','.json'),'r'))
                except:
                    print(jpg_path)
                    break
                if len(img.shape)==2:    
                    mask = np.zeros_like(img)
                elif len(img.shape) ==3:
                    mask = np.zeros_like(img[:,:,0])
                for shape in js['shapes']:
#                     print(shape['label'])
                    cv2.fillPoly(mask,[np.int32(shape['points'])],color=255)

                if verbose:
                    plt.figure(figsize=(10,5))
                    plt.subplot(1,2,1);plt.imshow(img,cmap='gray')
                    plt.subplot(1,2,2);plt.imshow(mask,cmap='gray')
                    plt.title(types_dict[i])
                    plt.show()

                imgs.append(img)
                masks.append(mask)
                type_labels.append(types_dict[i])
                name.append(d)
    return imgs,masks,type_labels,name