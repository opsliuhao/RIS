import torch
from collections import OrderedDict
from torch.autograd import Variable
from image_pool import ImagePool
from base_model import BaseModel
import networks
from torchvision import models

class ResViT_model(BaseModel):
    def name(self):
        return 'ResViT_model'

    def __init__(self,istrain,ispretrain):
        BaseModel.__init__(self, istrain,ispretrain)
        self.isTrain = istrain
        self.ispretrain = ispretrain
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(1, 1, 64,
                                     'resvit','Res-ViT-B_16',256,'instance',False, 'normal', self.gpu_ids,
                                      pre_trained_trans=1,pre_trained_resnet = 0)


        if self.isTrain:
            self.lambda_f = 0.9
            use_sigmoid = False
            self.netD = networks.define_D(2, 64,
                                          'basic','Res-ViT-B_16',256,
                                          3, 'instance', use_sigmoid, 'normal', self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(50)
            # define loss functions
            self.criterionGAN = networks.GANLoss('lsgan').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.schedulers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=self.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=self.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = True
        # torch.set_printoptions(profile="full")
        self.real_A = input['A' if AtoB else 'B'].to(self.device).type(torch.cuda.FloatTensor)
        self.real_B = input['B' if AtoB else 'A'].to(self.device).type(torch.cuda.FloatTensor)
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B= self.netG(self.real_A)
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*1

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*1
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*1
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
