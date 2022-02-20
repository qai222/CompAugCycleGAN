import functools
import itertools
import logging
from collections import OrderedDict

import torch.nn
import torch.nn
import torch.nn
import torch.nn
from torch.utils.data import DataLoader

from cacgan.data.dataset import FormulaDataset
from cacgan.gans.general import F, criterion_GAN, discriminate
from cacgan.gans.logger import Logger
from cacgan.gans.networks import Discriminator, print_network, weights_init, ConditionalResnetGenerator, \
    LatentEncoder, DiscriminatorLatent
from cacgan.utils import *


class AugCycleGan:
    """
    model for augmented cyclegan
    adapted from Augmeted Cyclegan https://github.com/aalmah/augmented_cyclegan by Amjad Almahairi
    """

    def __init__(
            self,
            # should not need to change these
            input_nc=65,
            output_nc=65,
            possible_elements=None,
            g_blocks=9,
            nlatent=1,
            use_bias=True,
            work_dir="./",

            # lr related, should tune
            lr=0.0002,
            lr_divider=5,
            lr_slowdown_freq=50,
            lr_slowdown_start=50,
            lr_slowdown_param=0.95,

            # cycle loss composition, should tune
            lambda_A=1.0,
            lambda_B=1.0,
            lambda_z_B=1.0,
            lambda_z_A=1.0,

            # cycle loss weight in G loss, important, should tune
            cyc_weight=0.02,

            # if use emd, currently only support "mp" (Modified Pettifor Scale)
            emd=None,

    ):

        self.work_dir = os.path.abspath(work_dir)
        self.save_dir = os.path.join(self.work_dir, "chk")
        self.load_dir = os.path.join(self.work_dir, "chk")
        create_dir(self.work_dir)
        create_dir(self.save_dir)
        create_dir(self.load_dir)

        self.possible_elements = possible_elements
        self.nlatent = nlatent

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_z_B = lambda_z_B
        self.lambda_z_A = lambda_z_A
        self.cyc_weight = cyc_weight

        self.lr = lr
        self.old_lr = lr
        self.lr_divider = lr_divider
        self.lr_slowdown_param = lr_slowdown_param
        self.lr_slowdown_freq = lr_slowdown_freq
        self.lr_slowdown_start = lr_slowdown_start

        use_sigmoid = False  # let's stick to ls-gan...

        self.current_epoch = None

        self.input_nc = input_nc
        self.output_nc = output_nc
        ngf = input_nc + output_nc

        nef = input_nc + output_nc
        enc_input_nc = output_nc
        enc_output_nc = input_nc

        enc_input_nc += input_nc
        enc_output_nc += output_nc

        self.netE_B = LatentEncoder(nlatent=nlatent, input_nc=enc_input_nc, nef=nef, use_bias=use_bias)
        self.netE_A = LatentEncoder(nlatent=nlatent, input_nc=enc_output_nc, nef=nef, use_bias=use_bias)

        self.netG_A_B = ConditionalResnetGenerator(nlatent=nlatent, input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                                                   use_dropout=False,
                                                   n_blocks=g_blocks, use_bias=use_bias)
        self.netG_B_A = ConditionalResnetGenerator(nlatent=nlatent, input_nc=output_nc, output_nc=input_nc, ngf=ngf,
                                                   use_dropout=False,
                                                   n_blocks=g_blocks, use_bias=use_bias)

        ndf = 2 * input_nc
        self.netD_A = Discriminator(input_nc=input_nc, ndf=ndf, use_sigmoid=use_sigmoid, use_norm=False)

        ndf = 2 * output_nc
        self.netD_B = Discriminator(input_nc=output_nc, ndf=ndf, use_sigmoid=use_sigmoid, use_norm=False)

        self.netD_z_A = DiscriminatorLatent(nlatent=nlatent, ndf=ndf, use_sigmoid=use_sigmoid, use_norm=True)
        self.netD_z_B = DiscriminatorLatent(nlatent=nlatent, ndf=ndf, use_sigmoid=use_sigmoid, use_norm=True)

        if len(GPU_IDs) > 0:
            self.netG_A_B.cuda()
            self.netG_B_A.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()
            self.netE_B.cuda()
            self.netE_A.cuda()
            self.netD_z_B.cuda()
            self.netD_z_A.cuda()

        self.netG_A_B.apply(weights_init)
        self.netG_B_A.apply(weights_init)
        self.netD_A.apply(weights_init)
        self.netD_B.apply(weights_init)
        self.netE_A.apply(weights_init)
        self.netE_B.apply(weights_init)
        self.netD_z_A.apply(weights_init)
        self.netD_z_B.apply(weights_init)

        ###############################
        # optimizers
        self.optimizer_G_B = torch.optim.Adam(itertools.chain(self.netG_A_B.parameters(), self.netE_B.parameters()),
                                              lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_G_A = torch.optim.Adam(itertools.chain(self.netG_B_A.parameters(), self.netE_A.parameters()),
                                              lr=self.lr, betas=(0.5, 0.999))

        self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(), self.netD_z_B.parameters(), ),
                                              lr=self.lr / self.lr_divider, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_z_A.parameters(), ),
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

    def write_structure(self, outpath=None):
        if outpath is None:
            outpath = "%s/nets.txt" % self.save_dir
        with open(outpath, 'w') as nets_f:
            print_network(self.netG_A_B, nets_f)
            print_network(self.netG_B_A, nets_f)
            print_network(self.netD_A, nets_f)
            print_network(self.netD_B, nets_f)
            print_network(self.netE_A, nets_f)
            print_network(self.netE_B, nets_f)
            print_network(self.netD_z_A, nets_f)
            print_network(self.netD_z_B, nets_f)

    def train_instance(self, real_A, real_B, prior_z_A, prior_z_B):

        fake_B = self.netG_A_B.forward(real_A, prior_z_B)
        fake_A = self.netG_B_A.forward(real_B, prior_z_A)

        concat_B_A = torch.cat((fake_A, real_B), dim=1)
        mu_z_realB, logvar_z_realB = self.netE_B.forward(concat_B_A)

        concat_A_B = torch.cat((fake_B, real_A), dim=1)
        mu_z_realA, logvar_z_realA = self.netE_A.forward(concat_A_B)

        post_z_realB = mu_z_realB.view(mu_z_realB.size(0), mu_z_realB.size(1))
        logvar_z_realB = logvar_z_realB * 0.0
        post_z_realA = mu_z_realA.view(mu_z_realA.size(0), mu_z_realA.size(1))
        logvar_z_realA = logvar_z_realA * 0.0

        loss_D_fake_A, loss_D_true_A, pred_fake_A, pred_true_A = discriminate(self.netD_A, self.criterionGAN,
                                                                              fake_A.detach(), real_A)
        loss_D_fake_B, loss_D_true_B, pred_fake_B, pred_true_B = discriminate(self.netD_B, self.criterionGAN,
                                                                              fake_B.detach(), real_B)

        loss_D_post_z_B, loss_D_prior_z_B, pred_post_z_B, pred_prior_z_B = discriminate(self.netD_z_B,
                                                                                        self.criterionGAN,
                                                                                        post_z_realB.detach(),
                                                                                        prior_z_B)
        loss_D_post_z_A, loss_D_prior_z_A, pred_post_z_A, pred_prior_z_A = discriminate(self.netD_z_A,
                                                                                        self.criterionGAN,
                                                                                        post_z_realA.detach(),
                                                                                        prior_z_A)

        loss_D_A = 0.5 * (loss_D_fake_A + loss_D_true_A)
        loss_D_B = 0.5 * (loss_D_fake_B + loss_D_true_B)
        loss_D_z_B = 0.5 * (loss_D_post_z_B + loss_D_prior_z_B)
        loss_D_z_A = 0.5 * (loss_D_post_z_A + loss_D_prior_z_A)

        loss_D = loss_D_A + loss_D_B

        loss_D += loss_D_z_B + loss_D_z_A

        # NOTE: after the following snippet, the discriminator parameters will change
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_B.zero_grad()
        loss_D.sum().backward()
        self.optimizer_D_A.step()
        self.optimizer_D_B.step()

        # NOTE: The generator and encoder ALI loss is computed using the new (updated)
        # discriminator parameters.
        pred_fake_A = self.netD_A.forward(fake_A)
        loss_G_A = self.criterionGAN(pred_fake_A, True)

        pred_fake_B = self.netD_B.forward(fake_B)
        loss_G_B = self.criterionGAN(pred_fake_B, True)

        pred_post_z_B = self.netD_z_B.forward(post_z_realB)
        loss_G_z_B = self.criterionGAN(pred_post_z_B, True)

        pred_post_z_A = self.netD_z_A.forward(post_z_realA)
        loss_G_z_A = self.criterionGAN(pred_post_z_A, True)

        # reconstruct z_B from A and fake_B : A ==> z_B <== fake_B
        concat_A_B = torch.cat((real_A, fake_B), 1)
        mu_z_fakeB, logvar_z_fakeB = self.netE_B.forward(concat_A_B)

        # reconstruct z_A from B and fake_A : B ==> z_A <== fake_A
        concat_B_A = torch.cat((real_B, fake_A), 1)
        mu_z_fakeA, logvar_z_fakeA = self.netE_A.forward(concat_B_A)

        # minimize the NLL of original z_B sample
        bs = prior_z_B.size(0)
        loss_cycle_z_B = F.l1_loss(mu_z_fakeB.view(bs, self.nlatent),
                                   prior_z_B.view(bs, self.nlatent))
        loss_cycle_z_A = F.l1_loss(mu_z_fakeA.view(bs, self.nlatent),
                                   prior_z_A.view(bs, self.nlatent))

        ##### B -> A,z_B -> B cycle loss
        rec_B = self.netG_A_B.forward(fake_A, post_z_realB)
        if self.emd:
            # loss_cycle_B = self.criterionCycle(rec_B, real_B, self.dist)
            loss_cycle_B = self.criterionCycle(rec_B, real_B, self.dist)
        else:
            loss_cycle_B = self.criterionCycle(rec_B, real_B)

        rec_A = self.netG_B_A.forward(fake_B, post_z_realA)
        if self.emd:
            # loss_cycle_A = self.criterionCycle(rec_A, real_A, self.dist)
            loss_cycle_A = self.criterionCycle(rec_A, real_A, self.dist)
        else:
            loss_cycle_A = self.criterionCycle(rec_A, real_A)

        ##### Generation and Encoder optimization

        loss_cycle_A = loss_cycle_A.sum()
        loss_cycle_B = loss_cycle_B.sum()
        loss_cycle_z_A = loss_cycle_z_A.sum()
        loss_cycle_z_B = loss_cycle_z_B.sum()

        loss_cycle = loss_cycle_A * self.lambda_A + loss_cycle_B * self.lambda_B
        loss_cycle += loss_cycle_z_B * self.lambda_z_B + loss_cycle_z_A + self.lambda_z_A
        loss_G = loss_G_A + loss_G_B + self.cyc_weight * loss_cycle

        loss_G += loss_G_z_B + loss_G_z_A

        self.optimizer_G_A.zero_grad()
        self.optimizer_G_B.zero_grad()
        loss_G.sum().backward()
        self.optimizer_G_A.step()
        self.optimizer_G_B.step()

        losses = OrderedDict(
            [
                ('D_A', loss_D_A.data.item()),
                ('G_A', loss_G_A.data.item()),
                ('Cyc_A', loss_cycle_A.data.item()),
                ('D_B', loss_D_B.data.item()),
                ('G_B', loss_G_B.data.item()),
                ('Cyc_B', loss_cycle_B.data.item()),
                ('G_z_A', loss_G_z_A.data.item()),
                ('G_z_B', loss_G_z_B.data.item()),
                ('Cyc_z_B', loss_cycle_z_B.data.item()),
                ('Cyc_z_A', loss_cycle_z_A.data.item()),
                ('D', loss_D.data.item(),),
                ('G', loss_G.data.item(),),
                ('C', loss_cycle.data.item(),),
            ]
        )
        return losses

    def predict_B(self, real_A, z_B):
        return self.netG_A_B.forward(real_A, z_B)

    def predict_enc_params_zB(self, real_A, real_B):
        concat_B_A = torch.cat((real_A, real_B), 1).to(device=DEVICE)
        mu, logvar = self.netE_B.forward(concat_B_A)
        return mu

    def save(self):
        checkpoint = {
            'netG_A_B': self.netG_A_B.state_dict(),
            'netG_B_A': self.netG_B_A.state_dict(),
            'netD_A': self.netD_A.state_dict(),
            'netD_B': self.netD_B.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
            'optimizer_G_A': self.optimizer_G_A.state_dict(),
            'optimizer_G_B': self.optimizer_G_B.state_dict(),
        }
        for k in checkpoint:
            torch.save(checkpoint[k], os.path.join(self.save_dir, "{}.pth".format(k)))

    def save_epoch(self, nepoch):
        checkpoint = {
            'netG_A_B': self.netG_A_B.state_dict(),
            'netG_B_A': self.netG_B_A.state_dict(),
            'netD_A': self.netD_A.state_dict(),
            'netD_B': self.netD_B.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
            'optimizer_G_A': self.optimizer_G_A.state_dict(),
            'optimizer_G_B': self.optimizer_G_B.state_dict(),
        }
        save_dir = os.path.join(self.save_dir, str(nepoch))
        create_dir(save_dir)
        for k in checkpoint:
            torch.save(checkpoint[k], os.path.join(save_dir, "{}.pth".format(k)))
        with open(os.path.join(save_dir, "done.dummy"), "w") as f:
            f.write("done")

    def load(self, epoch=None):
        logging.warning("loading {} from: {}".format(self.__class__.__name__, self.load_dir))
        checkpoint = {
            'netG_A_B': self.netG_A_B,
            'netG_B_A': self.netG_B_A,
            'netD_A': self.netD_A,
            'netD_B': self.netD_B,
            'optimizer_D_A': self.optimizer_D_A,
            'optimizer_D_B': self.optimizer_D_B,
            'optimizer_G_A': self.optimizer_G_A,
            'optimizer_G_B': self.optimizer_G_B,
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
        self.netE_A.eval()
        self.netE_B.eval()

    def update_lr(self):
        new_lr = self.lr * self.lr_slowdown_param
        self.old_lr = self.lr
        self.lr = new_lr
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.optimizer_G_A.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = self.lr / self.lr_divider
        for param_group in self.optimizer_G_B.param_groups:
            param_group['lr'] = self.lr / self.lr_divider
        logging.warning('update learning rate: %f -> %f' % (self.old_lr, self.lr))

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
        logging.warning("training with batchsize: {}".format(batchsize))

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

                prior_z_A = real_B.data.new(real_B.size(0), self.nlatent).normal_(0, 1)
                prior_z_B = real_A.data.new(real_A.size(0), self.nlatent).normal_(0, 1)

                losses = self.train_instance(real_A, real_B, prior_z_A, prior_z_B)
                logger.log(losses)

            # save_pkl(logger.losses_by_epoch, os.path.join(self.save_dir, "loss.pkl"))
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

            self.current_epoch = epoch

            if epoch % save_freq == 0 and epoch > 0:
                logging.warning("save epoch at: {}".format(epoch))
                self.save_epoch(epoch)
                self.save()

    def predict_from_prior(self, test_set: FormulaDataset, n_trials=50, std=5, back_to_train=True, unique=True, ):
        """
        returns (real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded)
        here fake A and fake B are 3d arrays, the first dimension is num_trials
        """
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

        real_A_comps = batch["Acomp"]
        real_B_comps = batch["Bcomp"]

        # # print(len(real_A_comps), len(real_B_comps))
        # from pymatgen.core.composition import Composition
        # real_A_comps = set(Composition(f) for f in real_A_comps)
        # real_B_comps = set(Composition(f) for f in real_B_comps)
        #
        # # print(len(real_A_comps), len(real_B_comps))

        with torch.no_grad():
            real_A = input_A.copy_(real_A)
            real_B = input_B.copy_(real_B)

        # unique removes the pairing between real_A[i] and real_B[i], that is, they can be of different chem sys
        if unique:
            real_A = torch.unique(real_A, dim=0, )
            real_B = torch.unique(real_B, dim=0, )

        fake_A_encoded = np.zeros((n_trials, real_B.shape[0], self.input_nc))
        fake_B_encoded = np.zeros((n_trials, real_A.shape[0], self.output_nc))

        for itry in range(n_trials):
            with torch.no_grad():
                z_A = real_B.data.new(real_B.shape[0], self.nlatent).normal_(0, std)
                z_B = real_A.data.new(real_A.shape[0], self.nlatent).normal_(0, std)

            fake_B = self.netG_A_B(real_A, z_B).data
            fake_A = self.netG_B_A(real_B, z_A).data

            fake_B = fake_B.cpu().detach().numpy()
            fake_A = fake_A.cpu().detach().numpy()
            fake_A_encoded[itry] = fake_A
            fake_B_encoded[itry] = fake_B
        real_A_encoded = real_A.cpu().detach().numpy()
        real_B_encoded = real_B.cpu().detach().numpy()

        if back_to_train:
            self.netG_A_B.train()
            self.netG_B_A.train()

        return real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded

    def predict_from_opt(self, test_set: FormulaDataset, zlim=10.0, steps=2000, unique=True, back_to_train=True, ):
        """
        returns (real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded)
        here fake A and fake B are 3d arrays, the first dimension is num_z_sampled (steps argument here)
        """
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

        fake_A_encoded = np.zeros((steps, real_B.shape[0], self.input_nc))
        fake_B_encoded = np.zeros((steps, real_A.shape[0], self.output_nc))

        assert self.nlatent == 1

        for iz, z in enumerate(np.linspace(-zlim, zlim, num=steps)):
            with torch.no_grad():
                z_A = torch.tensor(z, dtype=torch.float32, device=DEVICE).repeat(real_B.shape[0], 1)
                z_B = torch.tensor(z, dtype=torch.float32, device=DEVICE).repeat(real_A.shape[0], 1)

            fake_B = self.netG_A_B(real_A, z_B).data
            fake_A = self.netG_B_A(real_B, z_A).data

            fake_B = fake_B.cpu().detach().numpy()
            fake_A = fake_A.cpu().detach().numpy()
            fake_A_encoded[iz] = fake_A
            fake_B_encoded[iz] = fake_B

        real_A_encoded = real_A.cpu().detach().numpy()
        real_B_encoded = real_B.cpu().detach().numpy()

        if back_to_train:
            self.netG_A_B.train()
            self.netG_B_A.train()
        return real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded
