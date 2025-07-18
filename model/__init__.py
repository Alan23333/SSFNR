import os
import scipy.io as sio
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
import model.DCT as DCT
import model.DHIF as DHIF
import model.DSPNet as DSPNet
import model.KNLNet as KNLNet
import model.MSST as MSST
import model.DTNet as DTNet
import model.MOG as MOG
import model.MIMO as MIMO
import model.SSFNet as SSFNet
import model.DTNetChikusei as DTNetChikusei
from utility import *


MODEL_DICT = {'SSFNet': SSFNet}

class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        ckp.write_log(f"Making model {opt.model}")
        # self.device = torch.device('cpu' if opt.cpu else 'cuda')

        self.device = torch.device("cuda:%s" % opt.gpu_ids[0]
                                   if torch.cuda.is_available() and len(opt.gpu_ids) > 0
                                   else "cpu")

        try:
            Net = MODEL_DICT[opt.model]
        except KeyError:
            raise ValueError(f"Unknown model `{opt.model}`; "
                             f"available: {list(MODEL_DICT)}")

        self.model = Net.make_model(opt).to(self.device)


        if opt.test_only:
            self.load(opt.pre_train, cpu=opt.cpu)

        # compute parameter
        num_parameter = self.count_parameters(self.model)
        ckp.write_log(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M")

    def forward(self, x, y='subspace'):
        if y == 'subspace' :
            return self.model(x)
        else:
            return self.model(x, y)

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self, path, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(path, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', 'model_best.pt')
            )

    def load(self, pre_train='.', cpu=False):
        """
        Load checkpoint safely:
        1. always deserialize on CPU to avoid GPU OOM
        2. move model to target device only after state_dict is loaded
        """
        if pre_train == '.':
            return

        print(f'Loading model from {pre_train}')

        ckpt = torch.load(pre_train, map_location='cpu')

        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        self.get_model().cpu()
        missing, unexpected = self.get_model().load_state_dict(ckpt, strict=False)

        if missing:
            print(f'=> Missing keys     : {len(missing)}')
        if unexpected:
            print(f'=> Unexpected keys  : {len(unexpected)}')

        if not cpu:
            self.get_model().to(self.device)
            torch.cuda.empty_cache()
