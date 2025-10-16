from utility import *
import data
import model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer


if __name__=='__main__':
    set_seed(args.seed)
    args = optset(args)
    args.subspace = 'Yes'
    args.save = f'./experiment/{args.dataset}/f{args.scale}/{args.model}/'
    checkpoint = Checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args)
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()