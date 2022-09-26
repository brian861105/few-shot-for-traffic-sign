import  torch, os
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
from utils.dataloader import train_data_gen , test_data_gen
# from maml_meta import Meta
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from    torch import optim
from maml_learner import Learner
from    torch.nn import functional as F
from    copy import deepcopy
import json
import shutil
import torchvision.utils as vutils
import  argparse

def main(args):
    print(args)
    with open("maml_configs/" + args.args_path) as json_file:
        args = json.load(json_file)
    
    writer = SummaryWriter("maml_runs/" + args["save_path"])
    
    def mkdir_p(path):
        if not os.path.exists("maml/" + path):
            os.makedirs("maml/" + path)

    spt_size = args["k_spt"] * args["n_way"]
    qry_size = args["k_qry"] * args["n_way"]
    # BASE
    fm = args["fm"]
    config = [
        ("conv2d", [fm, 3, 3, 3, 1, 0]),
        ("leakyrelu", [0.2,True]),
        ("bn", [fm]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [fm, fm, 3, 3, 1, 0]),
        ("leakyrelu", [0.2,True]),
        ("bn", [fm]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [fm, fm, 3, 3, 1, 0]),
        ("leakyrelu", [0.2,True]),
        ("bn", [fm]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [fm, fm, 3, 3, 1, 0]),
        ("leakyrelu", [0.2,True]),
        ("bn", [fm]),
        ("max_pool2d", [2, 1, 0]),
        ("flatten", []),
        ("linear", [6, fm * 5 * 5])
    ]
    
    train_data_generator = train_data_gen(args)
    test_data_generator = test_data_gen(args)
    
    device = torch.device("cuda:0")
    nz = 100
    ngf = 64
    ndf = 64
    nc = 3

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 0, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)
    if args["gan"]:
        netG = Generator().to(device)
        netG.load_state_dict(torch.load("GAN_save_models/generator/fixed/model_step_15.pt")["model_state_dict"])
    class Meta(nn.Module):
        """
        Meta Learner
        """
        def __init__(self, args, config):
            """

            :param args:
            """
            super(Meta, self).__init__()

            self.update_lr = args["update_lr"]
            self.meta_lr = args["meta_lr"]
            self.n_way = args["n_way"]
            self.k_spt = args["k_spt"]
            self.k_qry = args["k_qry"]
            self.task_num = args["task_num"]
            self.update_step = args["update_step"]
            self.update_step_test = args["update_step_test"]
            self.distractor = args["num_distractor"]
            self.net = Learner(config, args["img_c"], args["img_sz"])
            self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
            if args["consine_schedule"]:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.meta_optim, T_max=args["consine_schedule"], eta_min=args["eta_min"])
            self.device = torch.device("cuda")
            self.gan = args["gan"]
            self.multi_step_loss_num_epochs = args["multi_step_loss_num_epochs"]
        def get_per_step_loss_importance_vector(self):
            """
            Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
            loss towards the optimization loss.
            :return: A tensor to be used to compute the weighted average of the loss, useful for
            the MSL (Multi Step Loss) mechanism.
            """
            loss_weights = np.ones(shape=(self.update_step)) * (
                    1.0 / self.update_step)
            decay_rate = 1.0 / self.update_step / self.multi_step_loss_num_epochs
            min_value_for_non_final_losses = 0.03 / self.update_step
            for i in range(len(loss_weights) - 1):
                curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
                loss_weights[i] = curr_value

            curr_value = np.minimum(
                loss_weights[-1] + (self.current_epoch * (self.update_step - 1) * decay_rate),
                1.0 - ((self.update_step - 1) * min_value_for_non_final_losses))
            loss_weights[-1] = curr_value
            loss_weights = torch.Tensor(loss_weights).to(device=self.device)
            return loss_weights

        def forward(self, x_spt, y_spt, x_qry, y_qry, current_epoch,unlabel_spt_image=None, unlabel_qry_image=None,gan_spt=None, gan_qry=None):
            """

            :param x_spt:   [b, setsz, c_, h, w]
            :param y_spt:   [b, setsz]
            :param x_qry:   [b, querysz, c_, h, w]
            :param y_qry:   [b, querysz]
            :return:
            """
            self.current_epoch = current_epoch
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            task_num, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)
            if self.gan :
                gan_sptsz = gan_spt.size(1)
                gan_qrysz = gan_qry.size(1)
            else:
                gan_sptsz = 0
                gan_qrysz = 0
            if self.distractor or self.gan:
                corrects = {}
                corrects["total_query_nway"] = np.zeros(self.update_step + 1)
                if self.distractor:
                    unlabel_querysz = unlabel_qry.size(1)
                    corrects["query_nway_recall"] = np.zeros(self.update_step + 1)
                    corrects["label_query_nway_recall"] = np.zeros(self.update_step + 1)
                    corrects["distractor_query_nway_recall"] = np.zeros(self.update_step + 1)
                if self.gan :
                    corrects["gan_query_nway"] = np.zeros(self.update_step + 1)
            else:
                corrects = {key: np.zeros(self.update_step + 1) for key in 
                    [
                    "query_nway_recall",
                    "label_query_nway_recall"
                    ]}
            losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

            for i in range(task_num):
                spt_image = x_spt[i]
                spt_label = y_spt[i]
                qry_image = x_qry[i]
                qry_label = y_qry[i]
                if self.distractor:
                    spt_image = torch.concat((spt_image,unlabel_spt[i]))
                    spt_unlabel_label = torch.full((unlabel_spt.size(1),), 5, dtype=torch.long,device=self.device)
                    spt_label = torch.cat((spt_label,spt_unlabel_label))
                    qry_image = torch.concat((qry_image,unlabel_qry[i]))
                    qry_unlabel_label = torch.full((unlabel_qry.size(1),), 5, dtype=torch.long,device=self.device)
                    qry_label = torch.cat((qry_label,qry_unlabel_label))
                if self.gan :
                    spt_image = torch.concat((spt_image,gan_spt[i]))
                    spt_gan_label = torch.full((gan_spt.size(1),), 5, dtype=torch.long,device=self.device)
                    spt_label = torch.cat((spt_label,spt_gan_label))
                    qry_image = torch.concat((qry_image,gan_qry[i]))
                    qry_gan_label = torch.full((gan_qry.size(1),), 5, dtype=torch.long,device=self.device)
                    qry_label = torch.cat((qry_label,qry_gan_label))

                # 1. run the i-th task and compute loss for k=0
                logits = self.net(spt_image, vars=None, bn_training=True)
                loss = F.cross_entropy(logits, spt_label)
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    if self.distractor or self.gan:
                        total_logits_q = self.net(qry_image, self.net.parameters(), bn_training=False)
                        total_pred_q = F.softmax(total_logits_q, dim=1).argmax(dim=1)
                        total_q_correct = torch.eq(total_pred_q, qry_label).sum().item()
                        corrects["total_query_nway"][0] += total_q_correct
                        loss_q = F.cross_entropy(total_logits_q, qry_label)
                        losses_q[0] += loss_q
                        if self.distractor:
                            label_logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=False)
                            label_pred_q = F.softmax(label_logits_q, dim=1).argmax(dim=1)
                            label_pred_q_correct = torch.eq(label_pred_q, y_qry[i]).sum().item()
                            corrects["label_query_nway_recall"][0] += label_pred_q_correct
                            label_pred_q = F.softmax(label_logits_q[:,:-1], dim=1).argmax(dim=1)
                            label_pred_q_correct = torch.eq(label_pred_q, y_qry[i]).sum().item()
                            corrects["query_nway_recall"][0] += label_pred_q_correct


                            unlabel_logits_q = self.net(unlabel_qry[i], self.net.parameters(), bn_training=False)
                            unlabel_pred_q = F.softmax(unlabel_logits_q, dim=1).argmax(dim=1)
                            other = torch.eq(unlabel_pred_q, qry_unlabel_label).sum().item()
                            corrects["distractor_query_nway_recall"][0] += other
                        if self.gan :
                            gan_logits_q = self.net(gan_qry[i], self.net.parameters(), bn_training=False)
                            gan_pred_q = F.softmax(gan_logits_q, dim=1).argmax(dim=1)
                            gan_counts = torch.eq(gan_pred_q, qry_gan_label).sum().item()
                            corrects["gan_query_nway"][0] += gan_counts
                    else:
                        logits_q = self.net(qry_image, self.net.parameters(), bn_training=True)
                        loss_q = F.cross_entropy(logits_q, qry_label)
                        pred_q = F.softmax(logits_q[:,:-1], dim=1).argmax(dim=1)
                        q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                        corrects["query_nway_recall"][0] += q_discrim_correct
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                        corrects["label_query_nway_recall"][0] += q_discrim_correct
                        losses_q[0] += loss_q
                # this is the loss and accuracy after the first update
                with torch.no_grad():
                    # [setsz, nway]
                    if self.distractor or self.gan:

                        total_logits_q = self.net(qry_image,fast_weights , bn_training=False)
                        total_pred_q = F.softmax(total_logits_q, dim=1).argmax(dim=1)
                        total_q_correct = torch.eq(total_pred_q, qry_label).sum().item()
                        corrects["total_query_nway"][1] += total_q_correct
                        loss_q = F.cross_entropy(total_logits_q, qry_label)
                        losses_q[1] += loss_q
                        if self.distractor:
                            label_logits_q = self.net(x_qry[i], fast_weights, bn_training=False)
                            label_pred_q = F.softmax(label_logits_q, dim=1).argmax(dim=1)
                            label_pred_q_correct = torch.eq(label_pred_q, y_qry[i]).sum().item()
                            corrects["label_query_nway_recall"][1] += label_pred_q_correct
                            label_pred_q = F.softmax(label_logits_q[:,:-1], dim=1).argmax(dim=1)
                            label_pred_q_correct = torch.eq(label_pred_q, y_qry[i]).sum().item()
                            corrects["query_nway_recall"][1] += label_pred_q_correct

                            unlabel_logits_q = self.net(unlabel_qry[i], fast_weights, bn_training=False)
                            unlabel_pred_q = F.softmax(unlabel_logits_q, dim=1).argmax(dim=1)
                            other = torch.eq(unlabel_pred_q, qry_unlabel_label).sum().item()
                            corrects["distractor_query_nway_recall"][1] += other
                        if self.gan :
                            gan_logits_q = self.net(gan_qry[i], fast_weights, bn_training=False)
                            gan_pred_q = F.softmax(gan_logits_q, dim=1).argmax(dim=1)
                            gan_counts = torch.eq(gan_pred_q, qry_gan_label).sum().item()
                            corrects["gan_query_nway"][1] += gan_counts
                    else:
                        logits_q = self.net(qry_image, fast_weights, bn_training=False)
                        loss_q = F.cross_entropy(logits_q, qry_label)
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                        corrects["query_nway_recall"][1] += q_discrim_correct
                        pred_q = F.softmax(logits_q[:,:-1], dim=1).argmax(dim=1)
                        q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                        corrects["label_query_nway_recall"][1] += q_discrim_correct
                        losses_q[1] += loss_q

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(spt_image, fast_weights, bn_training=True)
                    loss = F.cross_entropy(logits, spt_label)
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(qry_image, fast_weights, bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, qry_label)
                    losses_q[k + 1] += loss_q
                    with torch.no_grad():
                        if self.distractor or self.gan:
                            total_logits_q = self.net(qry_image, fast_weights, bn_training=False)
                            total_pred_q = F.softmax(total_logits_q, dim=1).argmax(dim=1)
                            total_q_correct = torch.eq(total_pred_q, qry_label).sum().item()
                            corrects["total_query_nway"][k+1] += total_q_correct
                            if self.distractor:
                                label_logits_q = self.net(x_qry[i], fast_weights, bn_training=False)
                                label_pred_q = F.softmax(label_logits_q, dim=1).argmax(dim=1)
                                label_pred_q_correct = torch.eq(label_pred_q, y_qry[i]).sum().item()
                                corrects["label_query_nway_recall"][k+1] += label_pred_q_correct
                                label_pred_q = F.softmax(label_logits_q[:,:-1], dim=1).argmax(dim=1)
                                label_pred_q_correct = torch.eq(label_pred_q, y_qry[i]).sum().item()
                                corrects["query_nway_recall"][k+1] += label_pred_q_correct

                                unlabel_logits_q = self.net(unlabel_qry[i], fast_weights, bn_training=False)
                                unlabel_pred_q = F.softmax(unlabel_logits_q, dim=1).argmax(dim=1)
                                other = torch.eq(unlabel_pred_q, qry_unlabel_label).sum().item()
                                corrects["distractor_query_nway_recall"][k+1] += other
                            if self.gan :
                                gan_logits_q = self.net(gan_qry[i], fast_weights, bn_training=False)
                                gan_pred_q = F.softmax(gan_logits_q, dim=1).argmax(dim=1)
                                gan_counts = torch.eq(gan_pred_q, qry_gan_label).sum().item()
                                corrects["gan_query_nway"][k+1] += gan_counts
                        else:
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                            corrects["query_nway_recall"][k+1] += q_discrim_correct
                            pred_q = F.softmax(logits_q[:,:-1], dim=1).argmax(dim=1)
                            q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                            corrects["label_query_nway_recall"][k+1] += q_discrim_correct
            # end of all tasks
            # sum over all losses on query set across all tasks
            loss_q = 0

            for num_step, loss in enumerate(losses_q[1:]):
                loss_q = loss_q + per_step_loss_importance_vectors[num_step] * losses_q[-1] / task_num
            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()

            self.meta_optim.step()

            accs = {}
            if (self.distractor or self.gan):
                accs["total_query_nway"] = corrects["total_query_nway"] / (task_num * (querysz + unlabel_querysz + gan_qrysz))
                if self.distractor:
                    accs["label_query_nway_recall"] = corrects["label_query_nway_recall"] / (task_num * querysz)
                    accs["query_nway_recall"] = corrects["query_nway_recall"] / (task_num * querysz)
                    accs["distractor_query_nway_recall"] = corrects["distractor_query_nway_recall"] / (task_num * unlabel_querysz)
                if gan_qrysz:
                    accs["gan_query_nway"] = corrects["gan_query_nway"] / (task_num * gan_qrysz)
            else:
                accs["query_nway_recall"] = corrects["query_nway_recall"] / (task_num * querysz)
                accs["label_query_nway_recall"] = corrects["label_query_nway_recall"] / (task_num * querysz)
            return accs,loss_q


        def finetunning(self, x_spt, y_spt, x_qry, y_qry, unlabel_spt=None, unlabel_qry=None, gan_spt=None, gan_qry=None):

            assert len(x_spt.shape) == 4
            querysz = x_qry.size(0)
            if self.gan :
                gan_sptsz = gan_spt.size(0)
                gan_qrysz = gan_qry.size(0)
            else:
                gan_sptsz = gan_qrysz = 0
            if self.distractor or self.gan:
                corrects = {}
                corrects["total_query_nway"] = np.zeros(self.update_step_test + 1)
                if self.distractor:
                    unlabel_querysz = unlabel_qry.size(0)
                    corrects["label_query_nway_recall"] = np.zeros(self.update_step_test + 1)
                    corrects["query_nway_recall"] = np.zeros(self.update_step_test + 1)
                    corrects["distractor_query_nway_recall"] = np.zeros(self.update_step_test + 1)
                if self.gan:
                    corrects["gan_query_nway"] = np.zeros(self.update_step_test + 1)
            else:
                corrects = {key: np.zeros(self.update_step_test + 1) for key in 
                                [
                                "query_nway_recall",
                                "label_query_nway_recall"
                                ]}
            # in order to not ruin the state of running_mean/variance and bn_weight/bias
            # we finetunning on the copied model instead of self.net
            net = deepcopy(self.net)
            spt_image = x_spt
            spt_label = y_spt
            qry_image = x_qry
            qry_label = y_qry
            if self.distractor:
                spt_image = torch.concat((spt_image,unlabel_spt))
                spt_unlabel_label = torch.full((unlabel_spt.size(0),), 5, dtype=torch.long,device=self.device)
                spt_label = torch.cat((spt_label,spt_unlabel_label))
                qry_image = torch.concat((qry_image,unlabel_qry))
                qry_unlabel_label = torch.full((unlabel_qry.size(0),), 5, dtype=torch.long,device=self.device)
                qry_label = torch.cat((qry_label,qry_unlabel_label))
            if self.gan :
                spt_image = torch.concat((spt_image,gan_spt))
                spt_gan_label = torch.full((gan_spt.size(0),), 5, dtype=torch.long,device=self.device)
                spt_label = torch.cat((spt_label,spt_gan_label))
                qry_image = torch.concat((qry_image,gan_qry))
                qry_gan_label = torch.full((gan_qry.size(0),), 5, dtype=torch.long,device=self.device)
                qry_label = torch.cat((qry_label,qry_gan_label))

            # 1. run the i-th task and compute loss for k=0
            logits = net(spt_image)
            loss = F.cross_entropy(logits, spt_label)

            grad = torch.autograd.grad(loss, net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                if self.distractor or self.gan:
                    total_logits_q = self.net(qry_image, self.net.parameters(), bn_training=False)
                    total_pred_q = F.softmax(total_logits_q, dim=1).argmax(dim=1)
                    total_q_correct = torch.eq(total_pred_q, qry_label).sum().item()
                    corrects["total_query_nway"][0] += total_q_correct
                    loss_q = F.cross_entropy(total_logits_q, qry_label)
                    if self.distractor:
                        label_logits_q = self.net(x_qry, self.net.parameters(), bn_training=False)
                        label_pred_q = F.softmax(label_logits_q, dim=1).argmax(dim=1)
                        label_pred_q_correct = torch.eq(label_pred_q, y_qry).sum().item()
                        corrects["label_query_nway_recall"][0] += label_pred_q_correct

                        label_logits_q = self.net(x_qry, self.net.parameters(), bn_training=False)
                        label_pred_q = F.softmax(label_logits_q[:,:-1], dim=1).argmax(dim=1)
                        label_pred_q_correct = torch.eq(label_pred_q, y_qry).sum().item()
                        corrects["query_nway_recall"][0] += label_pred_q_correct

                        unlabel_logits_q = self.net(unlabel_qry, self.net.parameters(), bn_training=False)
                        unlabel_pred_q = F.softmax(unlabel_logits_q, dim=1).argmax(dim=1)
                        other = torch.eq(unlabel_pred_q, qry_unlabel_label).sum().item()
                        corrects["distractor_query_nway_recall"][0] += other
                    if self.gan :
                        gan_logits_q = self.net(gan_qry, self.net.parameters(), bn_training=False)
                        gan_pred_q = F.softmax(gan_logits_q, dim=1).argmax(dim=1)
                        gan_counts = torch.eq(gan_pred_q, qry_gan_label).sum().item()
                        corrects["gan_query_nway"][0] += gan_counts
                else:
                    logits_q = self.net(qry_image, self.net.parameters(), bn_training=True)
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                    corrects["query_nway_recall"][0] += q_discrim_correct
                    pred_q = F.softmax(logits_q[:,:-1], dim=1).argmax(dim=1)
                    q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                    corrects["label_query_nway_recall"][0] += q_discrim_correct
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                if self.distractor or self.gan:

                    total_logits_q = self.net(qry_image,fast_weights , bn_training=False)
                    total_pred_q = F.softmax(total_logits_q, dim=1).argmax(dim=1)
                    total_q_correct = torch.eq(total_pred_q, qry_label).sum().item()
                    corrects["total_query_nway"][1] += total_q_correct
                    loss_q = F.cross_entropy(total_logits_q, qry_label)
                    if self.distractor:
                        label_logits_q = self.net(x_qry, fast_weights, bn_training=False)
                        label_pred_q = F.softmax(label_logits_q, dim=1).argmax(dim=1)
                        label_pred_q_correct = torch.eq(label_pred_q, y_qry).sum().item()
                        corrects["label_query_nway_recall"][1] += label_pred_q_correct

                        label_logits_q = self.net(x_qry, fast_weights, bn_training=False)
                        label_pred_q = F.softmax(label_logits_q[:,:-1], dim=1).argmax(dim=1)
                        label_pred_q_correct = torch.eq(label_pred_q, y_qry).sum().item()
                        corrects["query_nway_recall"][1] += label_pred_q_correct

                        unlabel_logits_q = self.net(unlabel_qry, fast_weights, bn_training=False)
                        unlabel_pred_q = F.softmax(unlabel_logits_q, dim=1).argmax(dim=1)
                        other = torch.eq(unlabel_pred_q, qry_unlabel_label).sum().item()
                        corrects["distractor_query_nway_recall"][1] += other
                    if self.gan :
                        gan_logits_q = self.net(gan_qry, fast_weights, bn_training=False)
                        gan_pred_q = F.softmax(gan_logits_q, dim=1).argmax(dim=1)
                        gan_counts = torch.eq(gan_pred_q, qry_gan_label).sum().item()
                        corrects["gan_query_nway"][1] += gan_counts
                else:
                    logits_q = self.net(qry_image, fast_weights, bn_training=True)

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                    corrects["query_nway_recall"][1] += q_discrim_correct
                    pred_q = F.softmax(logits_q[:,:-1], dim=1).argmax(dim=1)
                    q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                    corrects["label_query_nway_recall"][1] += q_discrim_correct
            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(spt_image, fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, spt_label)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = net(qry_image, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, qry_label)

                with torch.no_grad():
                    if self.distractor or self.gan:
                        total_logits_q = self.net(qry_image, fast_weights, bn_training=False)
                        total_pred_q = F.softmax(total_logits_q, dim=1).argmax(dim=1)
                        total_q_correct = torch.eq(total_pred_q, qry_label).sum().item()
                        corrects["total_query_nway"][k+1] += total_q_correct
                        if self.distractor:
                            label_logits_q = self.net(x_qry, fast_weights, bn_training=False)
                            label_pred_q = F.softmax(label_logits_q, dim=1).argmax(dim=1)
                            label_pred_q_correct = torch.eq(label_pred_q, y_qry).sum().item()
                            corrects["label_query_nway_recall"][k+1] += label_pred_q_correct

                            label_logits_q = self.net(x_qry, fast_weights, bn_training=False)
                            label_pred_q = F.softmax(label_logits_q[:,:-1], dim=1).argmax(dim=1)
                            label_pred_q_correct = torch.eq(label_pred_q, y_qry).sum().item()
                            corrects["query_nway_recall"][k+1] += label_pred_q_correct

                            unlabel_logits_q = self.net(unlabel_qry, fast_weights, bn_training=False)
                            unlabel_pred_q = F.softmax(unlabel_logits_q, dim=1).argmax(dim=1)
                            other = torch.eq(unlabel_pred_q, qry_unlabel_label).sum().item()
                            corrects["distractor_query_nway_recall"][k+1] += other
                        if self.gan :
                            gan_logits_q = self.net(gan_qry, fast_weights, bn_training=False)
                            gan_pred_q = F.softmax(gan_logits_q, dim=1).argmax(dim=1)
                            gan_counts = torch.eq(gan_pred_q, qry_gan_label).sum().item()
                            corrects["gan_query_nway"][k+1] += gan_counts
                    else:
                        logits_q = self.net(qry_image, fast_weights, bn_training=True)
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                        corrects["query_nway_recall"][k+1] += q_discrim_correct
                        pred_q = F.softmax(logits_q[:,:-1], dim=1).argmax(dim=1)
                        q_discrim_correct = torch.eq(pred_q, qry_label).sum().item()
                        corrects["label_query_nway_recall"][k+1] += q_discrim_correct
            del net
            accs = {}
            if (self.distractor or self.gan):
                accs["total_query_nway"] = corrects["total_query_nway"] / (querysz + unlabel_querysz + gan_qrysz)
                if self.distractor:
                    accs["label_query_nway_recall"] = corrects["label_query_nway_recall"] / querysz
                    accs["query_nway_recall"] = corrects["query_nway_recall"] / querysz
                    accs["distractor_query_nway_recall"] = corrects["distractor_query_nway_recall"] / (unlabel_querysz)
                if self.gan:
                    accs["gan_query_nway"] = corrects["gan_query_nway"] / (gan_qrysz)
            else:
                accs["query_nway_recall"] = corrects["query_nway_recall"] / querysz
                accs["label_query_nway_recall"] = corrects["label_query_nway_recall"] / querysz
            return accs
        
    device = torch.device("cuda")
    maml = Meta(args, config).to(device)
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    path = args["save_path"]
    step = 0
    mkdir_p(path)
    for epoch in range(args["epoch"]//6000):
            # fetch meta_batchsz num of episode each time

        train_dataloader = DataLoader(train_data_generator, args["task_num"], shuffle=True, num_workers=1, pin_memory=True)
        x_spt = y_spt = x_qry = y_qry = unlabel_spt = unlabel_qry = gan_qry = gan_spt = 0
        for idx,data  in enumerate(train_dataloader):
            if args["gan"]:
                with torch.no_grad():
                    gan_spt_noise = torch.randn(args["task_num"]*args["spy_gan_num"], 100, 1, 1, device=device)
                    gan_spt = netG(gan_spt_noise).to(device).view(args["task_num"],args["spy_gan_num"],args["img_c"],args["img_sz"],args["img_sz"])
                    gan_qry_noise = torch.randn(args["task_num"]*args["qry_gan_num"], 100, 1, 1, device=device)
                    gan_qry = netG(gan_qry_noise).to(device).view(args["task_num"],args["qry_gan_num"],args["img_c"],args["img_sz"],args["img_sz"])
            if len(data) == 4:
                (x_spt, y_spt, x_qry, y_qry) = data
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                if args["gan"]:
                    accs,loss_q = maml(x_spt, y_spt, x_qry, y_qry,step,gan_spt=gan_spt, gan_qry=gan_qry)
                else:
                    accs,loss_q = maml(x_spt, y_spt, x_qry, y_qry,step)
            else:
                (x_spt, y_spt, x_qry, y_qry, unlabel_spt, unlabel_qry) = data
                x_spt, y_spt, x_qry, y_qry, unlabel_spt, unlabel_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device), \
                unlabel_spt.to(device), unlabel_qry.to(device)
                if args["spy_gan_num"]:
                    accs,loss_q = maml(x_spt, y_spt, x_qry, y_qry,step,unlabel_spt_image=unlabel_spt, unlabel_qry_image=unlabel_qry,gan_spt=gan_spt, gan_qry=gan_qry)
                else:
                    accs,loss_q = maml(x_spt, y_spt, x_qry, y_qry,step,unlabel_spt_image=unlabel_spt, unlabel_qry_image=unlabel_qry)

            writer.add_scalar("Loss/train_loss", loss_q, step)
            if "total_query_nway" in accs:
                writer.add_scalar("Accuracy/train_total_query_nway", accs["total_query_nway"][-1], step)
            if "label_query_nway_recall" in accs:
                writer.add_scalar("Accuracy/train_label_query_nway_recall", accs["label_query_nway_recall"][-1], step)
            if "distractor_query_nway_recall" in accs:
                writer.add_scalar("Accuracy/train_distractor_query_nway_recall", accs["distractor_query_nway_recall"][-1], step)
            if "query_nway_recall" in accs:
                writer.add_scalar("Accuracy/train_query_nway_recall", accs["query_nway_recall"][-1], step)
            if "gan_query_nway_recall" in accs:
                writer.add_scalar("Accuracy/train_gan_query_nway_recall", accs["gan_query_nway_recall"][-1], step)
            if "query_nway" in accs:
                writer.add_scalar("Accuracy/train_query_nway_recall", accs["query_nway_recall"][-1], step)
            if step % 30 == 0:
                print("step:", step, "\ttraining acc:", accs)
            if step % 100 == 0:  # evaluation
                db_test = DataLoader(test_data_generator, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = {
                                "total_query_nway":[],
                                "distractor_query_nway_recall":[],
                                "query_nway_recall":[],
                                "label_query_nway_recall":[],
                                "gan_query_nway":[]
                }
                for test_data in db_test:
                    if args["gan"]:
                        gan_spt_noise = torch.randn(args["spy_gan_num"], 100, 1, 1, device=device)
                        gan_spt = netG(gan_spt_noise).to(device)
                        gan_qry_noise = torch.randn(args["qry_gan_num"], 100, 1, 1, device=device)
                        gan_qry = netG(gan_qry_noise).to(device)
                    if len(test_data) == 4:
                        x_spt, y_spt, x_qry, y_qry = test_data
                        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                        if args["gan"]:
                            accs,loss_q = maml.finetunning(x_spt, y_spt, x_qry, y_qry,gan_spt=gan_spt, gan_qry=gan_qry)
                        else:
                            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)

                    else:
                        x_spt, y_spt, x_qry, y_qry, unlabel_spt, unlabel_qry = test_data
                        x_spt, y_spt, x_qry, y_qry, unlabel_spt, unlabel_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device),\
                                                    unlabel_spt.squeeze(0).to(device), unlabel_qry.squeeze(0).to(device)
                        if args["gan"]:
                            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, unlabel_spt, unlabel_qry,gan_spt=gan_spt, gan_qry=gan_qry)
                        else:
                            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry, unlabel_spt, unlabel_qry)
                    if "total_query_nway" in accs:
                        accs_all_test["total_query_nway"].append(accs["total_query_nway"])
                    if "label_query_nway_recall" in accs:
                        accs_all_test["label_query_nway_recall"].append(accs["label_query_nway_recall"])
                    if "distractor_query_nway_recall" in accs:
                        accs_all_test["distractor_query_nway_recall"].append(accs["distractor_query_nway_recall"])
                    if "gan_query_nway" in accs:
                        accs_all_test["gan_query_nway"].append(accs["gan_query_nway"])
                    if "query_nway_recall" in accs:
                        accs_all_test["query_nway_recall"].append(accs["query_nway_recall"])
                # [b, update_step+1]
                if "total_query_nway" in accs:
                    accs["total_query_nway"] = np.array(accs_all_test["total_query_nway"]).mean(axis=0).astype(np.float16)
                    writer.add_scalar("Accuracy/test_total_query_nway_accuracy", accs["total_query_nway"][-1], step)
                if "label_query_nway_recall" in accs:
                    accs["label_query_nway_recall"] = np.array(accs_all_test["label_query_nway_recall"]).mean(axis=0).astype(np.float16)

                    writer.add_scalar("Accuracy/test_label_query_nway_accuracy", accs["label_query_nway_recall"][-1], step)
                if "distractor_query_nway_recall" in accs:
                    accs["distractor_query_nway_recall"] = np.array(accs_all_test["distractor_query_nway_recall"]).mean(axis=0).astype(np.float16)
                    writer.add_scalar("Accuracy/test_distractor_query_nway_recall_accuracy", accs["distractor_query_nway_recall"][-1], step)
                if "gan_query_nway" in accs:
                    accs["gan_query_nway"] = np.array(accs_all_test["gan_query_nway"]).mean(axis=0).astype(np.float16)
                    writer.add_scalar("Accuracy/test_gan_query_nway_accuracy", accs["gan_query_nway"][-1], step)
                if "query_nway_recall" in accs:
                    accs["query_nway_recall"] = np.array(accs_all_test["query_nway_recall"]).mean(axis=0).astype(np.float16)
                    writer.add_scalar("Accuracy/test_query_nway_accuracy", accs["query_nway_recall"][-1], step)
                print(step)
                print("Test acc:", accs)

                torch.save(maml.state_dict(), "maml/" + path + "/model_step" + str(step) + ".pt")
            step += 1
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--args_path', type=str)

    args = argparser.parse_args()

    main(args)