from .dataset_tools_2 import datasetNShot
import torch

def train_data_gen(args):
    BelgiumTSC_train = datasetNShot('datasets/BelgiumTSC',
                        batchsz=1000,
                        n_way=args["n_way"],
                        img_c=args["img_c"],
                        k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        img_sz=args["img_sz"],
                        num_distractor=args["num_distractor"],
                        spy_distractor_num=args["spy_distractor_num"],
                        qry_distractor_num=args["qry_distractor_num"],
                        data_augmentation_num=args["data_augmentation_num"],
                        train=True)

    ArTS_train = datasetNShot('datasets/ArTS',
                        batchsz=1000,
                        n_way=args["n_way"],
                        img_c=args["img_c"],
                        k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        img_sz=args["img_sz"],
                        num_distractor=args["num_distractor"],
                        spy_distractor_num=args["spy_distractor_num"],
                        qry_distractor_num=args["qry_distractor_num"],
                        data_augmentation_num=args["data_augmentation_num"],
                        train=True)

    chinese_traffic_sign_train = datasetNShot('datasets/chinese_traffic_sign',
                        batchsz=1000,
                        n_way=args["n_way"],
                        img_c=args["img_c"],
                        k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        img_sz=args["img_sz"],
                        num_distractor=args["num_distractor"],
                        spy_distractor_num=args["spy_distractor_num"],
                        qry_distractor_num=args["qry_distractor_num"],
                        data_augmentation_num=args["data_augmentation_num"],
                        train=True)

    CVL_train = datasetNShot('datasets/CVL',
                        batchsz=1000,
                        n_way=args["n_way"],
                        img_c=args["img_c"],
                        k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        img_sz=args["img_sz"],
                        num_distractor=args["num_distractor"],
                        spy_distractor_num=args["spy_distractor_num"],
                        qry_distractor_num=args["qry_distractor_num"],
                        data_augmentation_num=args["data_augmentation_num"],
                        train=True)

    FullJCNN_train = datasetNShot('datasets/FullJCNN2013',
                        batchsz=1000,
                        n_way=args["n_way"],
                        img_c=args["img_c"],
                        k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        img_sz=args["img_sz"],
                        num_distractor=args["num_distractor"],
                        spy_distractor_num=args["spy_distractor_num"],
                        qry_distractor_num=args["qry_distractor_num"],
                        data_augmentation_num=args["data_augmentation_num"],
                        train=True)

    logo_train = datasetNShot('datasets/logo_2k',
                        batchsz=1000,
                        n_way=args["n_way"],
                        img_c=args["img_c"],
                        k_shot=args["k_spt"],
                        k_query=args["k_qry"],
                        img_sz=args["img_sz"],
                        num_distractor=args["num_distractor"],
                        spy_distractor_num=args["spy_distractor_num"],
                        qry_distractor_num=args["qry_distractor_num"],
                        data_augmentation_num=args["data_augmentation_num"],
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


def test_data_gen(args):
    GTSRB_test = datasetNShot('datasets/GTSRB',
                            batchsz=40,
                            n_way=args["n_way"],
                            img_c=args["img_c"],
                            k_shot=args["k_spt"],
                            k_query=10,
                            img_sz=args["img_sz"],
                            num_distractor=2,
                            spy_distractor_num=1,
                            qry_distractor_num=5,
                            data_augmentation_num=args["data_augmentation_num"],
                            train=False)


    DFG_test = datasetNShot('datasets/DFG',
                            batchsz=40,
                            n_way=args["n_way"],
                            img_c=args["img_c"],
                            k_shot=args["k_spt"],
                            k_query=10,
                            img_sz=args["img_sz"],
                            num_distractor=2,
                            spy_distractor_num=1,
                            qry_distractor_num=5,
                            data_augmentation_num=args["data_augmentation_num"],                          
                            train=False)


    test_data_generator = torch.utils.data.ConcatDataset([        
        GTSRB_test,
        DFG_test
        ])
    
    return test_data_generator