import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np
import glob
import random
from torch.utils import data
import torch
import time

class gaussian_noise(object):
    
    def __init__(self,mean=0,sigma=0.1):
        self.mean = np.random.uniform(0,0.25)
        self.sigma = np.random.uniform(0.1,0.01)
        
    def __call__(self,img):
       
        if np.random.randint(0,2):
            img = np.array(img) / 255.

            h,w,c = img.shape

            noise = np.random.normal(self.mean, self.sigma, img.shape) # 隨機生成高斯 noise (float + float)
            # noise + 原圖
            gaussian_out = img + noise
            # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
            gaussian_out = np.clip(gaussian_out, 0, 1)

            # 原圖: float -> int (0~1 -> 0~255)
            gaussian_out = np.uint8(gaussian_out*255)
            # noise: float -> int (0~1 -> 0~255)
            gaussian_out = Image.fromarray(gaussian_out, 'RGB')
        
            return gaussian_out
        else:
            return img
    
        
class datasetNShot(data.Dataset):

    def __init__(self, root, batchsz, n_way, k_shot, k_query,img_c, img_sz,num_distractor,spy_distractor_num,qry_distractor_num ,data_augmentation_num ,train = False):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param img_sz:
        """
        print("load",root)
        self.root = root
        t0 = time.time()
        self.resize = img_sz
        self.img_c = img_c
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.querysz = self.n_way * self.k_query 
        self.data_augmentation_num = data_augmentation_num
        self.setsz = self.n_way * self.k_shot
        self.num_distractor = num_distractor
        self.spy_distractor_num = spy_distractor_num
        self.qry_distractor_num = qry_distractor_num
        self.distractor_querysz = self.num_distractor * self.qry_distractor_num
        self.distractor_setsz = self.num_distractor * self.spy_distractor_num
        if train :
            self.dataset_type = "train"
        else:
            self.dataset_type = "test"
        if train == True:
            self.origin_transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 # transforms.FiveCrop(self.resize),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 gaussian_noise(mean=0,sigma=0.1),
                                                 # transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.origin_transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),

                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        
        self.data = glob.glob(f"{root}/{self.dataset_type}/*/*.jpg")
        
        self.class_list = [x.split('/')[-2] for x in self.data]
        # self.data = [x for x in self.data]
        self.img_dict = {}
        self.img2label = {}
        self.label2index = {}
        
        label_index = 0
        for img,label in zip(self.data,self.class_list):
            if label in self.img_dict.keys():
                self.img_dict[label].append(img)
                self.img2label[img] = self.label2index[label]
            else:
                self.img2label[img] = label_index
                self.label2index[label] = label_index
                label_index = label_index + 1
                self.img_dict[label] = [img]

        self.cls_num = len(self.data)

        self.create_batch(self.batchsz)
        t1 = time.time()
        print("load complete time",t1-t0)
        


    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        *episode* here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        #label
        self.support_x_batch = []  # support set batch

        self.query_x_batch = []  # query set batch
        # unlabel
        if self.num_distractor:
            self.unlabel_support_x_batch = []

            self.unlabel_query_x_batch = []
        
        for _ in range(batchsz):  # for each batch
            selected_cls = np.random.choice(list(self.img_dict.keys()), self.n_way + self.num_distractor, False)  # no duplicate
            label_cls = np.random.choice(selected_cls,self.n_way,False)
            unlabel_cls = list(set(selected_cls) - set(label_cls))
            np.random.shuffle(label_cls)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            unlabel_support_x = []
            unlabel_query_x = []
            for idx,cls in enumerate(label_cls):
                # 2. select k_shot + k_query for each class
                if len(self.img_dict[cls]) >= (self.k_shot + self.k_query):
                    selected_imgs_idx = np.random.choice(len(self.img_dict[cls]), self.k_shot + self.k_query, False)
                else:
                    selected_imgs_idx = np.random.choice(len(self.img_dict[cls]), len(self.img_dict[cls]), False)
                    resample_selected_imgs_idx = np.random.choice(selected_imgs_idx[self.k_shot:],self.k_shot + self.k_query - len(selected_imgs_idx),False)
                    selected_imgs_idx = np.concatenate((selected_imgs_idx,resample_selected_imgs_idx),axis=0)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])
                support_x.append(
                    np.array(self.img_dict[cls])[indexDtrain].tolist())
                query_x.append(
                    np.array(self.img_dict[cls])[indexDtest].tolist())
                support_y.append([cls for _ in indexDtrain])
                
            random.shuffle(support_x)
            random.shuffle(query_x)                
            if self.num_distractor:
                for idx,cls in enumerate(unlabel_cls):
                    if len(self.img_dict[cls]) >= (self.spy_distractor_num + self.qry_distractor_num):
                        selected_imgs_idx = np.random.choice(len(self.img_dict[cls]), self.spy_distractor_num + self.qry_distractor_num, False)
                    else:
                        selected_imgs_idx = np.random.choice(len(self.img_dict[cls]), len(self.img_dict[cls]), False)
                        resample_selected_imgs_idx = np.random.choice(selected_imgs_idx[self.spy_distractor_num:],self.spy_distractor_num + self.qry_distractor_num - len(selected_imgs_idx),False)
                        selected_imgs_idx = np.concatenate((selected_imgs_idx,resample_selected_imgs_idx),axis=0)
                    indexDtrain = np.array(selected_imgs_idx[:self.spy_distractor_num])
                    indexDtest = np.array(selected_imgs_idx[self.spy_distractor_num:])
                    unlabel_support_x.append(
                        np.array(self.img_dict[cls])[indexDtrain].tolist())
                    unlabel_query_x.append(
                        np.array(self.img_dict[cls])[indexDtest].tolist())
                
            # shuffle the correponding relation between support set and query set
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
            if self.num_distractor:
                self.unlabel_support_x_batch.append(unlabel_support_x)  # append set to current sets
                self.unlabel_query_x_batch.append(unlabel_query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # print(self.root)
        #label
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int32)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int32)

        flatten_support_x = [item
                             for sublist in self.support_x_batch[index] for item in sublist]
        support_y = np.array(
            [self.img2label[item]
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [item
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        


        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx
            
        i = 0
        support_x = support_x.repeat(self.data_augmentation_num+1,1,1,1)

        for idx in range(self.data_augmentation_num+1):

            if idx == 0:
                for _, path in enumerate(flatten_support_x):
                    support_x[i] = self.origin_transform(path)

                    i = i + 1

            else:
                for _, path in enumerate(flatten_support_x):

                    support_x[i] = self.transform(path)
                    i = i + 1


        support_y_relative = np.tile(support_y_relative,self.data_augmentation_num+1)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.origin_transform(path)


        # unlabel
        # [setsz, 3, resize, resize]
        if self.num_distractor:
            unlabel_query_x = torch.FloatTensor(self.distractor_querysz, 3, self.resize, self.resize)
            # [querysz, 3, resize, resize]
            unlabel_support_x = torch.FloatTensor(self.distractor_setsz, 3, self.resize, self.resize)
            # [querysz]
            flatten_unlabel_support_x = [item
                                 for sublist in self.unlabel_support_x_batch[index] for item in sublist]

            flatten_unlabel_query_x = [item
                               for sublist in self.unlabel_query_x_batch[index] for item in sublist]
            i = 0
            unlabel_support_x = unlabel_support_x.repeat(self.data_augmentation_num+1,1,1,1)
            for idx in range(self.data_augmentation_num+1):

                if idx == 0:
                    for _, path in enumerate(flatten_unlabel_support_x):
                        unlabel_support_x[i] = self.origin_transform(path)

                        i = i + 1

                else:
                    for _, path in enumerate(flatten_unlabel_support_x):

                        unlabel_support_x[i] = self.transform(path)
                        i = i + 1

            for i, path in enumerate(flatten_unlabel_query_x):
                unlabel_query_x[i] = self.origin_transform(path)
                

        if self.num_distractor:
            return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative),unlabel_support_x, unlabel_query_x
        else:
            return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)
    def __len__(self):

        return self.batchsz
