import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    generator_2 import Generator
from    copy import deepcopy
from torch.autograd import Variable
from conditioner import Conditioner



class MetaGAN(nn.Module):
    """
    Meta Learner with GAN incorporated
    """
    def __init__(self, args, shared_config, nway_config, discriminator_config, gen_config):
        """
        :param args:
        """
        super(MetaGAN, self).__init__()
        self.total_epochs = args["epoch"]
        self.update_lr = args["update_lr"]
        self.gan_update_lr = args["gan_update_lr"]
        self.meta_lr = args["meta_lr"]
        self.n_way = args["n_way"]
        self.k_spt = args["k_spt"]
        self.k_qry = args["k_qry"]
        self.tasks_per_batch = args["tasks_per_batch"]
        self.update_steps = args["update_steps"]
        self.update_steps_test = args["update_steps_test"]
        self.learn_inner_lr = args["learn_inner_lr"]
        self.condition_discrim = args["condition_discrim"]
        self.single_fast_test = args["single_fast_test"]
        self.consine_schedule = args["consine_schedule"]
        self.min_learning_rate = args["min_learning_rate"]
        self.conditioner = Conditioner()

        # links for understanding how to make generator config
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        # http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html
        # https://github.com/soumith/dcgan.torch/blob/master/main.lua
        self.generator = Generator(gen_config, args["img_c"], args["img_sz"], args["n_way"])
        # self.generator_reg = Generator(gen_config, args.img_c, args.img_sz, args.n_way)

        self.shared_net = Learner(shared_config, args["img_c"], args["img_sz"])
        self.nway_net = Learner(nway_config, args["img_c"], args["img_sz"])
        self.discrim_net = Learner(discriminator_config, args["img_c"], args["img_sz"])

        cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 
        params = list()
        if self.learn_inner_lr:
            self.learned_lrs = []
            for i in range(self.update_steps):
                gen_lrs =[Variable(self.FloatTensor(1).fill_(self.update_lr), requires_grad=True)]*len(self.generator.parameters())
                shared_lrs = [Variable(self.FloatTensor(1).fill_(self.update_lr), requires_grad=True)]*len(self.shared_net.parameters())
                nway_lrs = [Variable(self.FloatTensor(1).fill_(self.update_lr), requires_grad=True)]*len(self.nway_net.parameters())
                discrim_lrs = [Variable(self.FloatTensor(1).fill_(self.update_lr), requires_grad=True)]*len(self.discrim_net.parameters())

                self.learned_lrs.append((shared_lrs, nway_lrs, discrim_lrs, gen_lrs))
                for param_list in self.learned_lrs[i]:
                    params += param_list
        
        self.g_meta_optim = optim.Adam(self.generator.parameters(), lr=self.meta_lr)
        self.n_meta_optim = optim.Adam(self.nway_net.parameters(), lr=self.meta_lr)
        self.d_meta_optim = optim.Adam(self.discrim_net.parameters(), lr=self.meta_lr)
        # self.meta_optim = optim.Adam(params,lr=self.meta_lr)
        if self.consine_schedule:
            
            self.g_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.g_meta_optim, T_max=self.total_epochs,
                                                                  eta_min=self.min_learning_rate)
            self.n_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.n_meta_optim, T_max=self.total_epochs,
                                                                  eta_min=self.min_learning_rate)
            self.d_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.d_meta_optim, T_max=self.total_epochs,
                                                                  eta_min=self.min_learning_rate)
        self.real_val = 1.0 # requires that real_val > fake_val
        self.fake_val = 0.0
        self.create_graph = args["create_graph"]
        self.loss = args["loss"]


    # Returns predicted class logits and descriminator outputs according
    # to the input "x", using the shared/nway/discrim nets and weights provided
    # pass in 'conditions' if using conditioned discriminator
    def pred(self, x, weights=[None, None, None], nets=None, nway=True, discrim=True, conditions=None):
        if type(nets) == type(None):
            nets = [self.shared_net, self.nway_net, self.discrim_net]

        shared_weights, nway_weights, discrim_weights = weights
        shared_net, nway_net, discrim_net = nets

        # shared_layer = shared_net(x, vars=shared_weights, bn_training=True)
        discrim_logits = discrim_net(x, conditions=conditions, vars=discrim_weights, bn_training=True) if discrim else None
        class_logits = nway_net(x, vars=nway_weights, bn_training=True) if nway else None

        return class_logits, discrim_logits


    def get_num_corrects(self, real, y, x=None, weights=None, class_logits=None, discrim_logits=None, conditions=None):
        with torch.no_grad():
            if type(class_logits) == type(None):
                if self.condition_discrim:
                    class_logits, discrim_logits = self.pred(x, conditions=conditions[y], weights=weights)
                else:
                    class_logits, discrim_logits = self.pred(x, weights=weights)

            nway_correct = torch.eq(class_logits.argmax(dim=1), y).sum().item()

            if real:
                discrim_correct = (discrim_logits > 0.0).sum().item()
            else:
                discrim_correct = (discrim_logits < 0.0).sum().item()

        return nway_correct, discrim_correct

    # Returns the loss(es) of the y's according to the class and possibly also descriminator predictions
    def loss_cross_entropy(self, class_logits, y_class, discrim_logits=None, y_discrim=None):
        # https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py
        # should change to wasserstein loss with diff gen and discrim losses
        nway_loss = F.cross_entropy(class_logits, y_class)
        if type(discrim_logits) == type(None):
            return nway_loss

        discrim_loss = F.binary_cross_entropy_with_logits(discrim_logits, y_discrim)
        return nway_loss, discrim_loss
    # gives the generator and discriminator loss
    def loss_wasserstein_gp(self, gen_discrim_logits, real_discrim_logits, x_gen, x_real, weights, conditions=None):
        batch_size = real_discrim_logits.shape[0]
        lam = 10
        gen_discrim_loss = -torch.mean(gen_discrim_logits)

        alpha = self.FloatTensor(batch_size, 1, 1, 1).uniform_()
        x_interp = alpha * x_gen + (1-alpha) * x_real

        _, interp_discrim_logits = self.pred(x_interp, weights=weights, conditions=conditions, nway=False)

        grad = torch.autograd.grad([torch.mean(interp_discrim_logits)], [x_interp], create_graph=self.create_graph)
        derive_penalty = torch.pow(torch.norm(grad[0], p='fro') - 1, 2)

        discrim_loss = -gen_discrim_loss - torch.mean(real_discrim_logits) + lam * torch.mean(derive_penalty)
        return gen_discrim_loss, discrim_loss

    # Returns new weights by backpropping their affect on the losses.
    # Losses and weights should be (shared, nway, descrim)
    def update_weights(self, net_losses, net_weights, gen_loss, gen_weights):
        shared_loss, nway_loss, discrim_loss = net_losses
        shared_weights, nway_weights, discrim_weights = net_weights


        n_grad = torch.autograd.grad(nway_loss, nway_weights, retain_graph=True, create_graph=self.create_graph)
        n_weights = [w - self.update_lr * grad for grad, w in zip(n_grad, nway_weights)]

        d_grad = torch.autograd.grad(discrim_loss, discrim_weights, retain_graph=True, create_graph=self.create_graph)
        d_weights = [w - self.gan_update_lr * grad for grad, w in zip(d_grad, discrim_weights)]

        # s_grad = torch.autograd.grad(shared_loss, shared_weights, retain_graph=True, create_graph=self.create_graph)
        # s_weights = [w - self.update_lr * grad for grad, w in zip(s_grad, shared_weights)]
        s_weights = shared_weights

        g_grad = torch.autograd.grad(gen_loss, gen_weights, create_graph=self.create_graph)
        g_weights = [w - self.gan_update_lr * grad for grad, w in zip(g_grad, gen_weights)]

        return (s_weights, n_weights, d_weights), g_weights

    # Returns new weights by backpropping their affect on the losses.
    # Losses and weights should be (shared, nway, descrim)
    def update_weights_learned_lr(self, net_losses, net_weights, gen_loss, gen_weights, learned_lrs):
        shared_loss, nway_loss, discrim_loss = net_losses
        shared_weights, nway_weights, discrim_weights = net_weights
        shared_lrs, nway_lrs, discrim_lrs, gen_lrs = learned_lrs

        n_grad = torch.autograd.grad(nway_loss, nway_weights, retain_graph=True, create_graph=self.create_graph)
        n_weights = [w - lr * grad for grad, w, lr in zip(n_grad, nway_weights, nway_lrs)]

        d_grad = torch.autograd.grad(discrim_loss, discrim_weights, retain_graph=True, create_graph=self.create_graph)
        d_weights = [w - lr * grad for grad, w, lr in zip(d_grad, discrim_weights, discrim_lrs)]

        # s_grad = torch.autograd.grad(shared_loss, shared_weights, retain_graph=True, create_graph=self.create_graph)
        # s_weights = [w - lr * grad for grad, w, lr in zip(s_grad, shared_weights, shared_lrs)]
        s_weights = shared_weights

        g_grad = torch.autograd.grad(gen_loss, gen_weights, create_graph=self.create_graph)
        g_weights = [w - lr * grad for grad, w, lr in zip(g_grad, gen_weights, gen_lrs)]

        return (s_weights, n_weights, d_weights), g_weights

    def single_task_forward(self, x_spt, y_spt, x_qry, y_qry, nets=None, images=False):
        support_sz, c_, h, w = x_spt.size()
        qry_sz = x_spt.size(0)
        corrects = {key: np.zeros(self.update_steps + 1) for key in 
                        ["discrim_loss", # number of meta-test (query) images correctly discriminated
                        "q_nway", # number of meta-test (query) images correctly classified
                        "gen_loss", # number of generated images correctly discriminated
                        "gen_nway"]} # number of generated images correctly classified
        

        if type(nets) == type(None):
            nets = (self.shared_net, self.nway_net, self.discrim_net)

        # net_weights = []
        # for net in nets:
        #     net_weights.append([w.clone() for w in net.parameters()])

        # gen_weights = [w.clone() for w in self.generator.parameters()]

        net_weights = [net.parameters() for net in nets]
        gen_weights = self.generator.parameters()
       

        # inner_g_optim = optim.Adam(gen_weights, 1e-3, betas=(0.5, 0.999))
        # inner_d_optim = optim.Adam(net_weights[2], 1e-3, betas=(0.5, 0.999))


        
        with torch.no_grad():
            image_embeddings = self.conditioner(x_spt)
            image_embeddings = image_embeddings.squeeze()
            class_image_embeddings = torch.zeros((self.n_way, image_embeddings.shape[-1]))
            for i in range(self.n_way):
                y_spt_idxs = [j for j, x in enumerate(y_spt) if x == i]
                class_image_embeddings[i] = image_embeddings[y_spt_idxs].mean(axis=0)

        # this is the meta-test loss and accuracy before first update
        q_nway, q_discrim = self.get_num_corrects(real=True, y=y_qry, weights=[None, None, None], x=x_qry, conditions=class_image_embeddings)
        with torch.no_grad(): 
            q_class_logits, _ = self.pred(x_qry, weights=net_weights, discrim=False)
            loss_q = self.loss_cross_entropy(q_class_logits, y_qry) # doesn't use discrim loss
        corrects['q_nway'][0] += q_nway
        corrects['discrim_loss'][0] += loss_q


        real = Variable(self.FloatTensor(support_sz, 1).fill_(self.real_val), requires_grad=False)
        fake = Variable(self.FloatTensor(support_sz, 1).fill_(self.fake_val), requires_grad=False)
        # run the i-th task and compute loss for k-th inner update
        for k in range(1, self.update_steps + 1):
            x_gen, y_gen = self.generator(x_spt, y_spt, vars=gen_weights, bn_training=True) 

            y_spt_idxs = [int((y_spt == i).nonzero()[0]) for i in y_spt]
            if self.condition_discrim:
                real_class_logits, real_discrim_logits = self.pred(x_spt, conditions=class_image_embeddings[y_spt], weights=net_weights)
                gen_class_logits, gen_discrim_logits = self.pred(x_gen, conditions=class_image_embeddings[y_spt], weights=net_weights)
            else:
                real_class_logits, real_discrim_logits = self.pred(x_spt, weights=net_weights)
                gen_class_logits, gen_discrim_logits = self.pred(x_gen, weights=net_weights)

            if self.loss == "wasserstein":
                # assert y_gen == y_spt
                # we fucked if that aint true ^
                gen_discrim_loss, discrim_loss = self.loss_wasserstein_gp(gen_discrim_logits, real_discrim_logits, x_gen, x_spt, net_weights, conditions=class_image_embeddings[y_spt])
                real_nway_loss = self.loss_cross_entropy(real_class_logits, y_spt)
                gen_nway_loss = self.loss_cross_entropy(gen_class_logits, y_gen)
                nway_loss = (gen_nway_loss + real_nway_loss) / 2

                shared_loss =  nway_loss + discrim_loss 
                gen_loss = gen_discrim_loss + gen_nway_loss

            else:
                real_nway_loss, real_discrim_loss = self.loss_cross_entropy(real_class_logits, y_spt, real_discrim_logits, real)
                gen_nway_loss, gen_discrim_loss = self.loss_cross_entropy(gen_class_logits, y_gen, gen_discrim_logits, fake)

                nway_loss = (gen_nway_loss + real_nway_loss) / 2
                discrim_loss = (gen_discrim_loss + real_discrim_loss) / 2
                shared_loss = nway_loss + discrim_loss  #

                gen_loss = -1 * torch.nn.functional.logsigmoid(gen_discrim_logits).mean() #- gen_discrim_loss
                # if self.condition_discrim:
                #     gen_loss = -1 * torch.nn.functional.logsigmoid(gen_discrim_logits).mean()#- gen_discrim_loss #
                # else:
                #     gen_reg_weights = self.generator_reg.parameters()
                #     reg = 0
                #     for t1, t2 in zip(gen_reg_weights, gen_weights):
                #         reg += torch.norm(t1-t2)
                #     gen_loss = - gen_discrim_loss + self.gen_reg_w*reg #gen_nway_loss 

            # 2. compute grad on theta_pi
            net_losses = (shared_loss, nway_loss, discrim_loss)
            if self.learn_inner_lr:
                net_weights, gen_weights = self.update_weights_learned_lr(net_losses, net_weights, gen_loss, gen_weights, self.learned_lrs[k-1])
            else:
                net_weights, gen_weights = self.update_weights(net_losses, net_weights, gen_loss, gen_weights)


            # gen-nway and gen-discrim accuracy
            # using gen from before the update to save computation
            gen_nway_correct, gen_discrim_correct = self.get_num_corrects(real=False, y=y_gen, class_logits=gen_class_logits, discrim_logits=gen_discrim_logits, conditions=class_image_embeddings)

            corrects["gen_nway"][k-1] += gen_nway_correct
            corrects["gen_loss"][k-1] += gen_loss.item()

            # meta-test nway and discrim accuracy
            # [query_sz]
            q_nway_correct, q_discrim_correct = self.get_num_corrects(real=True, y=y_qry, x=x_qry, weights=net_weights, conditions=class_image_embeddings)
            
            corrects['q_nway'][k] += q_nway_correct
            corrects['discrim_loss'][k] += discrim_loss.item()


        # final gen-discrim and gen-nway accuracy
        with torch.no_grad():
            x_gen, y_gen = self.generator(x_spt, y_spt, vars=gen_weights, bn_training=False)
            gen_class_logits, gen_discrim_logits = self.pred(x_gen, weights=net_weights)
            gen_nway_loss, gen_discrim_loss = self.loss_cross_entropy(gen_class_logits, y_gen, gen_discrim_logits, fake)
            gen_loss = -1 * torch.nn.functional.logsigmoid(gen_discrim_logits).mean() #- gen_discrim_loss
            gen_nway_correct, gen_discrim_correct = self.get_num_corrects(real=False, y=y_gen, x=x_gen, weights=net_weights, conditions=class_image_embeddings)

            corrects['gen_nway'][-1] += gen_nway_correct
            corrects['gen_loss'][-1] += gen_loss.item()

        # meta-test loss
        real_q_class_logits, real_q_discim_logits = self.pred(x_qry, weights=net_weights)
        x_gen, y_gen = self.generator(x_spt, y_spt, vars=gen_weights, bn_training=False)
        gen_q_class_logits, gen_q_discim_logits = self.pred(x_gen, weights=net_weights)
        
        real = Variable(self.FloatTensor(qry_sz, 1).fill_(self.real_val), requires_grad=False)
        fake = Variable(self.FloatTensor(qry_sz, 1).fill_(self.fake_val), requires_grad=False)
        
        real_nway_loss, real_discrim_loss = self.loss_cross_entropy(real_q_class_logits, y_qry, real_discrim_logits, real)
        gen_nway_loss, gen_discrim_loss = self.loss_cross_entropy(gen_class_logits, y_gen, gen_discrim_logits, fake)

        n_loss_q = (gen_nway_loss + real_nway_loss) / 2
        d_loss_q = (gen_discrim_loss + real_discrim_loss) / 2
        g_loss_q = -1 * torch.nn.functional.logsigmoid(gen_discrim_logits).mean() #- gen_discrim_loss
        
        if images:
            return g_loss_q, n_loss_q, d_loss_q, corrects, x_gen
        else:
            return g_loss_q, n_loss_q, d_loss_q, corrects


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, support_sz, c_, h, w]
        :param y_spt:   [b, support_sz]
        :param x_qry:   [b, query_sz, c_, h, w]
        :param y_qry:   [b, query_sz]
        :return:
        """
        tasks_per_batch, support_sz, c_, h, w = x_spt.size()
        query_sz = x_qry.size(1)

        if self.single_fast_test:
            tasks_per_batch = 1
        g_loss_q = 0
        n_loss_q = 0
        d_loss_q = 0
        corrects = {key: np.zeros(self.update_steps + 1) for key in 
                        ["discrim_loss", # number of meta-test (query) images correctly discriminated
                        "q_nway", # number of meta-test (query) images correctly classified
                        "gen_loss", # number of generated images correctly discriminated
                        "gen_nway"]} # number of generated images correctly classified
        
        for i in range(tasks_per_batch):
            g_loss_q_tmp,n_loss_q_tmp,d_loss_q_tmp, corrects_tmp = self.single_task_forward(x_spt[i], y_spt[i], x_qry[i], y_qry[i], images=False)
            g_loss_q += g_loss_q_tmp
            n_loss_q += n_loss_q_tmp
            d_loss_q += d_loss_q_tmp

            assert len(corrects_tmp.keys()) == len(corrects.keys())
            for key in corrects.keys():
                corrects[key] += corrects_tmp[key]

        # end of all tasks
        # sum over final losses on query set across all tasks
        g_loss_q /= tasks_per_batch
        n_loss_q /= tasks_per_batch
        d_loss_q /= tasks_per_batch

        # optimize theta parameters
        self.g_meta_optim.zero_grad()
        g_loss_q.backward()
        self.g_meta_optim.step()
        
        self.n_meta_optim.zero_grad()
        n_loss_q.backward()
        self.n_meta_optim.step()
        
        self.d_meta_optim.zero_grad()
        d_loss_q.backward()
        self.d_meta_optim.step()
        self.meta_optim.zero_grad()
        self.meta_optim.step()
        accs = {}
        accs["q_nway"] = corrects["q_nway"] / (query_sz * tasks_per_batch)
        accs["discrim_loss"] = corrects["discrim_loss"]
        # right now i assume that we are generating the support numer of samples (which we do)
        # if we ever change this then this code will need to change as well
        accs["gen_loss"] = corrects["gen_loss"]
        accs["gen_nway"] = corrects["gen_nway"] / (support_sz * tasks_per_batch)


        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [support_sz, c_, h, w]
        :param y_spt:   [support_sz]
        :param x_qry:   [query_sz, c_, h, w]
        :param y_qry:   [query_sz]
        :return:
        """

        support_sz, c_, h, w = x_spt.size()

        assert len(x_spt.shape) == 4

        query_sz = x_qry.size(0)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        shared_net = deepcopy(self.shared_net)
        nway_net = deepcopy(self.nway_net)
        discrim_net = deepcopy(self.discrim_net)
        nets = (shared_net, nway_net, discrim_net)

        loss_q, corrects, imgs = self.single_task_forward(x_spt, y_spt, x_qry, y_qry, nets=nets, images=True)

        del shared_net
        del nway_net
        del discrim_net
        del nets # this may not be necessary

        accs = corrects['q_nway'] / query_sz


        return accs, imgs

def main():
    pass


if __name__ == '__main__':
    main()
