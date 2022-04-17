import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np
import glob
import random
from torch.utils import data
import torch
import time
class datasetNShot(data.Dataset):

    # def __init__(self, root, batchsz, n_way, k_shot, k_query,img_c, img_sz,unsupsz ,train = False):
    def __init__(self, root, batchsz, n_way, k_shot, k_query,img_c, img_sz ,train = False):
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
        t0 = time.time()
        self.resize = img_sz
        self.img_c = img_c
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.querysz = self.n_way * self.k_query 
        self.setsz = self.n_way * self.k_shot
        # self.unsup_label = unsupsz
        # self.unsupsz = self.unsup_label * self.k_shot

        if train :
            self.dataset_type = "train"
        else:
            self.dataset_type = "test"
        if train == True:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        
        self.data = glob.glob(f"{root}/{self.dataset_type}/*/*.jpg")
        
        self.class_list = [x.split('/')[-2] for x in self.data]
        self.data = [x for x in self.data]
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
        self.support_x_batch = []  # support set batch

        self.query_x_batch = []  # query set batch

        for _ in range(batchsz):  # for each batch

            
            selected_cls = np.random.choice(list(self.img_dict.keys()), self.n_way, False)  # no duplicate

            np.random.shuffle(selected_cls)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            for idx,cls in enumerate(selected_cls):
                # 2. select k_shot + k_query for each class
                
                selected_imgs_idx = np.random.choice(len(self.img_dict[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])
                support_x.append(
                    np.array(self.img_dict[cls])[indexDtrain].tolist())
                query_x.append(
                    np.array(self.img_dict[cls])[indexDtest].tolist())
                support_y.append([cls for _ in indexDtrain])
            random.shuffle(support_x)
            random.shuffle(query_x)
            # shuffle the correponding relation between support set and query set
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
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

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx


        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        # for i, path in enumerate(flatten_query_x):
        #     query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y_relative)
        # return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)
    def __len__(self):

        return self.batchsz
