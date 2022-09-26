from .dataset_tools import datasetNShot
import torch

def train_data_gen(args):
    train_data_generator = datasetNShot('datasets/mini_imagenet',
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


    return train_data_generator


def test_data_gen(args):
    test_data_generator = datasetNShot('datasets/mini_imagenet',
                            batchsz=4,
                            n_way=args["n_way"],
                            img_c=args["img_c"],
                            k_shot=args["k_spt"],
                            k_query=15,
                            img_sz=args["img_sz"],
                            num_distractor=args["num_distractor"],
                            spy_distractor_num=args["spy_distractor_num"],
                            qry_distractor_num=args["qry_distractor_num"],
                            data_augmentation_num=args["data_augmentation_num"],

                            train=False)



    return test_data_generator