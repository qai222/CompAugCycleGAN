import functools
import itertools
import logging
from collections import OrderedDict

import torch.nn
import torch.nn
from torch.utils.data import DataLoader

from cacgan.data.dataset import FormulaDataset
from cacgan.gans.general import F, criterion_GAN
from cacgan.gans.logger import Logger
from cacgan.gans.networks import Discriminator, print_network, weights_init, ResnetGenerator
from cacgan.utils import *


class CycleGan:
    """
    model for cyclegan
    """

    def __init__(
            self,
            input_nc=65,
            output_nc=65,
            possible_elements=None,

            lr=0.0002,
            lr_divider=5,
            lr_slowdown_freq=50,
            lr_slowdown_start=50,
            lr_slowdown_param=0.95,

            g_blocks=9,
            lambda_A=1.0,
            lambda_B=1.0,

            cyc_weight=0.02,

            work_dir="./",
            emd=None,

    ):
        self.work_dir = os.path.abspath(work_dir)
        self.save_dir = os.path.join(self.work_dir, "chk")
        self.load_dir = os.path.join(self.work_dir, "chk")
        create_dir(self.work_dir)
        create_dir(self.save_dir)
        create_dir(self.load_dir)

        self.possible_elements = possible_elements
        self.lr = lr
        self.old_lr = lr
        self.lr_divider = lr_divider
        self.lr_slowdown_param = lr_slowdown_param
        self.lr_slowdown_freq = lr_slowdown_freq
        self.lr_slowdown_start = lr_slowdown_start

        self.cyc_weight = cyc_weight

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        use_sigmoid = False
        use_dropout = False

        self.input_nc = input_nc
        self.output_nc = output_nc
        ngf = input_nc + output_nc
        self.netG_A_B = ResnetGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, use_dropout=use_dropout,
                                        n_blocks=g_blocks, use_bias=True)
        self.netG_B_A = ResnetGenerator(input_nc=output_nc, output_nc=input_nc, ngf=ngf, use_dropout=use_dropout,
                                        n_blocks=g_blocks, use_bias=True)

        ndf = 2 * input_nc
        # self.netD_A = Discriminator(input_nc=input_nc, ndf=ndf, use_sigmoid=use_sigmoid)
        self.netD_A = Discriminator(input_nc=input_nc, ndf=ndf, use_sigmoid=use_sigmoid, use_norm=False)
        ndf = 2 * output_nc
        # self.netD_B = Discriminator(input_nc=output_nc, ndf=ndf, use_sigmoid=use_sigmoid)
        self.netD_B = Discriminator(input_nc=output_nc, ndf=ndf, use_sigmoid=use_sigmoid, use_norm=False)

        if DEVICE == "cuda":
            self.netG_A_B.cuda()
            self.netG_B_A.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()

        self.netG_A_B.apply(weights_init)
        self.netG_B_A.apply(weights_init)
        self.netD_A.apply(weights_init)
        self.netD_B.apply(weights_init)

        ###############################
        # optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A_B.parameters(), self.netG_B_A.parameters()),
                                            lr=self.lr, betas=(0.5, 0.999))
        # init lr for D
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=self.lr / self.lr_divider, betas=(0.5, 0.999))

        self.criterionGAN = functools.partial(criterion_GAN, use_sigmoid=use_sigmoid)

        self.emd = emd

        if self.emd == "mp":
            from cacgan.sinkhornOT import element_distance_matrix, ModifiedPettiforScale, emdloss
            logging.warning("using emd loss")
            with torch.no_grad():
                dmatrix = element_distance_matrix(possible_elements=self.possible_elements,
                                                  sequence_used=ModifiedPettiforScale)
            if len(GPU_IDs) > 0:
                dmatrix = dmatrix.cuda()
            self.criterionCycle = emdloss
            self.dist = dmatrix
        else:
            self.criterionCycle = F.l1_loss

    def write_structure(self):
        with open("%s/nets.txt" % self.save_dir, 'w') as nets_f:
            print_network(self.netG_A_B, nets_f)
            print_network(self.netG_B_A, nets_f)
            print_network(self.netD_A, nets_f)
            print_network(self.netD_B, nets_f)

    def train_instance(self, real_A, real_B):

        fake_B = self.netG_A_B.forward(real_A)
        fake_A = self.netG_B_A.forward(real_B)

        # NOTE: ".detach()" makes sure no gradient flows to the generator or encoder
        pred_fake_A = self.netD_A.forward(fake_A.detach())
        loss_D_fake_A = self.criterionGAN(pred_fake_A, False)

        pred_true_A = self.netD_A.forward(real_A)
        loss_D_true_A = self.criterionGAN(pred_true_A, True)

        pred_fake_B = self.netD_B.forward(fake_B.detach())
        loss_D_fake_B = self.criterionGAN(pred_fake_B, False)

        pred_true_B = self.netD_B.forward(real_B)
        loss_D_true_B = self.criterionGAN(pred_true_B, True)

        loss_D_A = 0.5 * (loss_D_fake_A + loss_D_true_A)
        loss_D_B = 0.5 * (loss_D_fake_B + loss_D_true_B)
        loss_D = loss_D_A + loss_D_B

        # NOTE: after the following snippet, the discriminator parameters will change
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        # NOTE: The generator and encoder ALI loss is computed using the new (updated)
        # discriminator parameters.
        pred_fake_A = self.netD_A.forward(fake_A)
        loss_G_A = self.criterionGAN(pred_fake_A, True)

        pred_fake_B = self.netD_B.forward(fake_B)
        loss_G_B = self.criterionGAN(pred_fake_B, True)

        rec_A = self.netG_B_A.forward(fake_B)
        if self.emd:
            loss_cycle_A = self.criterionCycle(rec_A, real_A, self.dist)
        else:
            loss_cycle_A = self.criterionCycle(rec_A, real_A)

        rec_B = self.netG_A_B.forward(fake_A)
        if self.emd:
            loss_cycle_B = self.criterionCycle(rec_B, real_B, self.dist)
        else:
            loss_cycle_B = self.criterionCycle(rec_B, real_B)

        ##### Generation and Encoder optimization
        # lambda_A = 10.0
        # lambda_B = 10.0
        loss_cycle = loss_cycle_A * self.lambda_A + loss_cycle_B * self.lambda_B
        loss_G = loss_G_A + loss_G_B + loss_cycle * self.cyc_weight

        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        ##### Return dicts
        losses = OrderedDict(
            [
                ('D_A', loss_D_A.data.item()),
                ('G_A', loss_G_A.data.item()),
                ('Cyc_A', loss_cycle_A.data.item()),
                ('D_B', loss_D_B.data.item()),
                ('G_B', loss_G_B.data.item()),
                ('Cyc_B', loss_cycle_B.data.item()),
                ('D', loss_D.data.item(),),
                ('G', loss_G.data.item(),),
                ('C', loss_cycle.data.item(),),
            ]
        )
        return losses

    def update_lr(self):
        new_lr = self.lr * self.lr_slowdown_param
        self.old_lr = self.lr
        self.lr = new_lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = self.lr
        logging.warning('update learning rate: %f -> %f' % (self.old_lr, self.lr))

    def save(self):
        checkpoint = {
            'netG_A_B': self.netG_A_B.state_dict(),
            'netG_B_A': self.netG_B_A.state_dict(),
            'netD_A': self.netD_A.state_dict(),
            'netD_B': self.netD_B.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
        }

        for k in checkpoint:
            torch.save(checkpoint[k], os.path.join(self.save_dir, "{}.pth".format(k)))

    def save_epoch(self, nepoch):
        checkpoint = {
            'netG_A_B': self.netG_A_B.state_dict(),
            'netG_B_A': self.netG_B_A.state_dict(),
            'netD_A': self.netD_A.state_dict(),
            'netD_B': self.netD_B.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
        }
        save_dir = os.path.join(self.save_dir, str(nepoch))
        create_dir(save_dir)
        for k in checkpoint:
            torch.save(checkpoint[k], os.path.join(save_dir, "{}.pth".format(k)))
        with open(os.path.join(save_dir, "done.dummy"), "w") as f:
            f.write("done")

    def load(self, epoch=None):
        checkpoint = {
            'netG_A_B': self.netG_A_B,
            'netG_B_A': self.netG_B_A,
            'netD_A': self.netD_A,
            'netD_B': self.netD_B,
            'optimizer_D': self.optimizer_D,
            'optimizer_G': self.optimizer_G,
        }
        if epoch is None:
            for model_name in checkpoint:
                pth_file = os.path.join(self.load_dir, '{}.pth'.format(model_name))
                state_dict = torch.load(pth_file)
                checkpoint[model_name].load_state_dict(state_dict)
        elif isinstance(epoch, int):
            load_dir = os.path.join(self.load_dir, str(epoch))
            for model_name in checkpoint:
                pth_file = os.path.join(load_dir, '{}.pth'.format(model_name))
                state_dict = torch.load(pth_file)
                checkpoint[model_name].load_state_dict(state_dict)

    def eval(self):
        self.netG_A_B.eval()
        self.netG_B_A.eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def train(
            self,
            train_set: FormulaDataset,
            nepochs=1000,
            batchsize=100,
            pin_mem=len(GPU_IDs) > 0,
            save_freq=200,
            load_pre_trained=True,
            starting_epoch=0,
            load_epoch=None
    ):
        seed_rng(SEED)
        assert train_set.mode == "train"

        input_A = torch.empty(batchsize, self.input_nc, device=DEVICE)
        input_B = torch.empty(batchsize, self.output_nc, device=DEVICE)
        train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=True,
                                      pin_memory=pin_mem)

        if load_pre_trained:
            try:
                self.load(epoch=load_epoch)
                logging.warning("*** model loaded epoch: {}".format(load_epoch))
            except:
                pass

        logger = Logger(nepochs, len(train_dataloader))
        loss_outfile = os.path.join(self.save_dir, "loss.csv")

        for epoch in range(starting_epoch, nepochs + starting_epoch):
            if epoch >= self.lr_slowdown_start + starting_epoch and epoch % self.lr_slowdown_freq == 0:
                self.update_lr()

            for i, batch in enumerate(train_dataloader):
                real_A = batch["A"]
                real_A = input_A.copy_(real_A)
                real_B = batch["B"]
                real_B = input_B.copy_(real_B)

                losses = self.train_instance(real_A, real_B)
                logger.log(losses)

            if epoch == starting_epoch:
                loss_string = ",".join(["i", ] + [k for k in logger.losses_at_this_epoch])
                loss_string += "\n"
                loss_string += "%06d" % epoch
            else:
                loss_string = "%06d" % epoch

            for k, v in logger.losses_at_this_epoch.items():
                loss_string += " ,{:.4f}".format(v)
            loss_string += "\n"
            f = open(loss_outfile, "a")
            f.write(loss_string)
            f.close()

            if epoch % save_freq == 0 and epoch > 0:
                logging.warning("save epoch at: {}".format(epoch))
                self.save_epoch(epoch)
                self.save()

    def predict_cyc(self, test_set: FormulaDataset, unique=True, back_to_train=True, return_numpy=True):
        self.netG_A_B.eval()
        self.netG_B_A.eval()

        test_set.test()
        batchsize = len(test_set)

        input_A = torch.empty(batchsize, self.input_nc, device=DEVICE)
        input_B = torch.empty(batchsize, self.output_nc, device=DEVICE)
        test_dataloader = DataLoader(test_set, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=True,
                                     pin_memory=True)

        for i, batch in enumerate(test_dataloader):
            break

        real_A = batch["A"]
        real_B = batch["B"]

        with torch.no_grad():
            real_A = input_A.copy_(real_A)
            real_B = input_B.copy_(real_B)

        if unique:
            real_A = torch.unique(real_A, dim=0)
            real_B = torch.unique(real_B, dim=0)

        fake_A_encoded = self.netG_B_A(real_B).detach()
        fake_B_encoded = self.netG_A_B(real_A).detach()

        real_A_encoded = real_A.detach()
        real_B_encoded = real_B.detach()

        if return_numpy:
            fake_B_encoded = fake_B_encoded.cpu().detach().numpy()
            fake_A_encoded = fake_A_encoded.cpu().detach().numpy()
            real_A_encoded = real_A.cpu().detach().numpy()
            real_B_encoded = real_B.cpu().detach().numpy()

        if back_to_train:
            self.netG_A_B.train()
            self.netG_B_A.train()
        return real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded
