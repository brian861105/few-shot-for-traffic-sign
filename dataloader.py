from datasetNShot import datasetNShot
import torch

def train_data_generator(args):
    BelgiumTSC_train = datasetNShot('dataset/BelgiumTSC',
                       batchsz=10000,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=True)

    ArTS_train = datasetNShot('dataset/ArTS',
                       batchsz=10000,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=True)

    chinese_traffic_sign_train = datasetNShot('dataset/chinese_traffic_sign',
                       batchsz=1000,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=True)

    CVL_train = datasetNShot('dataset/CVL',
                       batchsz=1000,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=True)

    FullJCNN_train = datasetNShot('dataset/FullJCNN2013',
                       batchsz=1000,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=True)

    logo_train = datasetNShot('dataset/logo_2k',
                       batchsz=1000,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=True)

    train_data_generator = torch.utils.data.ConcatDataset([        
        BelgiumTSC_train,
        ArTS_train,
        chinese_traffic_sign_train,
        CVL_train,
        FullJCNN_train,
        logo_train
        ])
    
    return train_data_generator


def test_data_generator(args):
    GTSRB_test = datasetNShot('dataset/GTSRB',
                       batchsz=100,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=False)


    DFG_test = datasetNShot('dataset/DFG',
                       batchsz=100,
                       n_way=args["n_way"],
                       img_c=args["img_c"],
                       k_shot=args["k_spt"],
                       k_query=args["k_qry"],
                       img_sz=args["img_sz"],
                       train=False)


    test_data_generator = torch.utils.data.ConcatDataset([        
        GTSRB_test,
        DFG_test
        ])
    
    return test_data_generator