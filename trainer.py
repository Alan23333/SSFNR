import numpy as np
import torch
import utility
from decimal import Decimal
from tqdm import tqdm
import datetime
import cv2
import time
from thop import profile
from thop import clever_format


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        if not self.opt.test_only:
            self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else "cpu")
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}\t{}'.format(epoch, Decimal(lr), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (C, CGT) in enumerate(self.loader_train):
            C   = C.to(self.device, non_blocking=True)
            CGT = CGT.to(self.device, non_blocking=True)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            C = self.model(C)

            loss = self.loss(C, CGT)

            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s\t{}'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            eval_psnr = 0
            eval_ssim = 0
            eval_sam = 0
            eval_ergas = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            timer_model = utility.timer()
            for _, (
                    X,
                    Y, Z, D, A,
                    filename
            ) in enumerate(tqdm_test):
                filename = filename[0]
                A = A.to(self.device, non_blocking=True)
                timer_model.tic()

                D = D[0].numpy()

                C = self.model(A)

                X = X[0].numpy()
                Y = Y[0].numpy()
                Z = Z[0].numpy()

                C = C[0].permute(1, 2, 0).cpu().numpy()
                sr1 = np.dot(D, utility.hyperConvert2D(C))
                sr2 = utility.reshape(sr1.T, (self.opt.testset_sizeI, self.opt.testset_sizeI, self.opt.channel))
                sr = utility.Upsample(utility.hyperConvert2D(sr2), Y, Z, self.opt.scale)

                psnr = utility.PSNR(X, sr, self.opt.data_range)
                eval_psnr += psnr
                ssim = utility.SSIM(X, sr, self.opt.data_range)
                eval_ssim += ssim
                sam = utility.SAM(X, sr)
                eval_sam += sam
                ergas = utility.ERGAS(X, sr, self.opt.scale)
                eval_ergas += ergas


            self.ckp.log[-1, 0] = eval_psnr / len(self.loader_test)
            eval_ssim = eval_ssim / len(self.loader_test)
            eval_sam = eval_sam / len(self.loader_test)
            eval_ergas = eval_ergas / len(self.loader_test)
            best = self.ckp.log.max(0)

            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.2f}\tSSIM: {:.4f}\tSAM: {:.2f}\tERGAS: {:.2f} (Best: {:.2f} @epoch {})'.format(
                    self.opt.dataset, self.scale,
                    self.ckp.log[-1, 0],
                    eval_ssim,
                    eval_sam,
                    eval_ergas,
                    best[0][0],
                    best[1][0] + 1
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs