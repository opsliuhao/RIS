import os
import torch
from collections import OrderedDict

class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self,istrain,ispretrain):
        self.istrain = istrain
        self.ispretrain = ispretrain
        self.continue_epoch = "80"
        self.gpu_ids = [0]
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        if self.ispretrain:
            self.save_dir = os.path.join(os.getcwd(),"monkey_brain","pretrain_results")  # save all the checkpoints to save_dir
        else:
            self.save_dir = os.path.join(os.getcwd(), "monkey_brain", "results")
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.visual_names = []

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass


    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids,slice_scope):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label,slice_scope):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))


    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
