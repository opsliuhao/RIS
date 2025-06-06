import os
import torch
from collections import OrderedDict
# 抽象类
from abc import ABC, abstractmethod
import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self,istrain,ispretrain):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.gpu_ids = [0]
        # 默认是训练状态
        self.isTrain = istrain
        self.ispretrain = ispretrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        if self.ispretrain:
            self.save_dir = os.path.join(os.getcwd(),"monkey_brain","pretrain_results")  # save all the checkpoints to save_dir
        else:
            self.save_dir = os.path.join(os.getcwd(), "monkey_brain", "results")
            # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        # 如果输入图像size不同的话，可以设为false，提高效率

        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.continue_train = False
        self.load_model_epoch = '80'

        self.now_epoch = 80
        self.lr = 0.0002
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self,slice_scope):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            # 得到学习速率变化的办法，默认是学习率不变。
            self.schedulers = [networks.get_scheduler(optimizer) for optimizer in self.optimizers]
        # 先不写保存模型了。
        # xian zai xie shang le now i write it
        if not self.isTrain or self.continue_train:
            load_suffix = self.load_model_epoch
            self.load_networks(load_suffix,slice_scope)
        #shi fou shu chu networks jia gou ,mo ren shi shuchu.
        # whether to output the structure of networks,defaulting is to output
        self.print_networks(True)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self,slices):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        self.lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, self.lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, slice_scope):
        # 保存当前模型。
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # 保存模型里存放的key和value的值。
                    # state_dict() 存放的是键值对。 如:conv1.weight torch.Size([6,3,5,5])
                    # 存放的是各种层的键值对。
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        # keys是一个列表
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        # 如果是最后一个
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            # 如：self.__class__ 获取类 <class '__main__.Basemodel'>
            # 如：self.__class__.__name__ 是获取类名 就是Basemodel
            # startswith是用来检查字符串是否以指定字符串开头。
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    # 用 . 去连接keys，并出栈。
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            # 第二个变量，从module里获取key的值。
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, slice_scope):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        # 在CyleGAN里面model_names就是下面这个列表。
        # self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        for name in self.model_names:
            if isinstance(name, str):
                # 这是为要加载模型做准备，这步是给加载的模型起个名字，保存的模型后缀是.pth
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)

                # getattr是返回一个对象属性值，类的一个对象的属性值，其实就是返回该对象的值，是几就是几。
                # net其实就是 netG_A,netG_B,等等。这里getattr其实就是获取netG_A等本身。
                # 而cyclegan里面的netG_A是这个，得到的就是这个对象。
                # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                net = getattr(self, 'net' + name)

                # 如果net是DataParallel的类型，那就并行。
                if isinstance(net, torch.nn.DataParallel):
                    # module表示要并行化的模块
                    net = net.module

                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                # 加载路径和设备
                state_dict = torch.load(load_path, map_location=str(self.device))
                # hasattr判断对象是否包含对应的属性。前面是否包含后面
                # 如果state_dict包含_metadata删掉，虽然不知道为啥，但感觉和CycleGAN没啥关系。
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    # 第一个参数是torch.load的返回值，第二个参数就是网络，第三个参数是torch.load返回字典的所有键，以.为分隔符，分开成列表。
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_continue_train(self,bool):
        """
        :param bool: you decide if you continue to train,inputting bool
        :return: None
        """
        self.continue_train = bool
    def set_load_model_epoch(self,epoch):
        """
        :param epoch:  you want the epoch what time to begin,inputting epoch,defult is 'latest',jiushi zui jinde nage.
        :return: None
        """
        self.load_model_epoch = epoch
