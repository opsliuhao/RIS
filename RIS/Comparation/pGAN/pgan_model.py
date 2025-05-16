import torch
from collections import OrderedDict
from torch.autograd import Variable
import image_pool
from base_model import BaseModel
import networks

from torchvision import models

class pGAN(BaseModel):
    def name(self):
        return 'pGAN'

    def __init__(self,isTrain,ispretrain,continue_train,slice_scope):
        BaseModel.__init__(self, isTrain, ispretrain)
        self.isTrain = isTrain
        self.continue_train = continue_train
        self.ispretrain = ispretrain
        self.slice_scope = slice_scope

        # load/define networks
        self.netG = networks.define_G(1, 1, 64,
                                      "instance", not True, "normal", self.gpu_ids)
        self.vgg=VGG16().cuda()
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            use_sigmoid = True
            self.netD = networks.define_D(2, 64,
                                          3, "instance", use_sigmoid, "normal", self.gpu_ids)
        if not self.isTrain or self.continue_train:
            self.load_network(self.netG, 'G', self.continue_epoch,slice_scope=self.slice_scope)
            if self.isTrain:
                self.load_network(self.netD, 'D', self.continue_epoch,slice_scope=self.slice_scope)

        if self.isTrain:

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not True, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = True
        input_A = input['A' if AtoB else 'B'].to(self.device).type(torch.cuda.FloatTensor)
        input_B = input['B' if AtoB else 'A'].to(self.device).type(torch.cuda.FloatTensor)
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    
    def test(self):
        # no backprop gradients
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
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
        #Perceptual loss
        self.VGG_real=self.vgg(self.real_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG_fake=self.vgg(self.fake_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG_loss=self.criterionL1(self.VGG_fake,self.VGG_real)* 1
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.VGG_loss
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_losses(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_VGG', self.VGG_loss.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def save(self, label,slice_scope):
        self.save_network(self.netG, 'G', label, self.gpu_ids,slice_scope=slice_scope)
        self.save_network(self.netD, 'D', label, self.gpu_ids,slice_scope=slice_scope)

      

#Extracting VGG feature maps before the 2nd maxpooling layer  
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)       
        return h_relu2

