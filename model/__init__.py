import os
import scipy.io as sio
import torch
import torch.nn as nn
import model.SSFNR as SSFNR
from utility import *


MODEL_DICT = {
    'SSFNR': SSFNR,
}

class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        ckp.write_log(f"Making model {opt.model}")

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
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )