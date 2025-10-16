import os
from data import common
import numpy as np
import scipy.io as sio
import random
from torch.utils.data import Dataset
import time

class TrainSet(Dataset):
    def __init__(self, args):
        self.args = args
        data = sio.loadmat('data/CAVE/response coefficient')
        self.R = data['R']
        self.B = data['B96']
        
        self.PrepareDataAndiniValue(self.args, self.R)
        self._set_dataset_length()
        self.X, self.Y = self.all_train_data_in(self.args)

    def __getitem__(self, idx):
        idx = self._get_index(idx)

        X = self.X[idx]
        Y = self.Y[idx]

        X, Y, Z = self.train_data_in(X, Y, self.B, self.args.patch_size, self.args.scale, self.args.channel)

        D, A = common.Upsample(self.args, Y, Z, self.R.T, self.B, self.args.scale)
        CGT = np.dot(D.T, common.hyperConvert2D(X))

        CGT = common.reshape(CGT.T, (self.args.patch_size, self.args.patch_size, -1))
        A = common.np2Tensor(A, data_range=self.args.data_range)
        CGT = common.np2Tensor(CGT, data_range=self.args.data_range)

        return A, CGT

    def __len__(self):
        return self.dataset_length

    def _set_dataset_length(self):
        self.dataset_length = self.args.train_num

    def _get_index(self, idx):
        return idx % self.args.trainset_file_num

    # Prepare dataset for training
    def PrepareDataAndiniValue(self, args, R):
        prepare = args.prepare
        DataRoad = f'data/CAVE/f{args.scale}/train/'
        if prepare != 'No':
            print('Generating the training dataset in folder data/CAVE/train')
            # the index will become traning dataset
            Ind = [2, 31, 25, 6, 27, 15, 19, 14, 12, 28, 26, 29, 8, 13, 22, 7, 24, 30, 10, 23]

            common.mkdir(DataRoad + 'X/')
            common.mkdir(DataRoad + 'Y/')

            for root, dirs, files in os.walk('data/CAVE/complete_ms_data/'):
                dirs.sort()
                for i in range(20):
                    print('processing ' + dirs[Ind[i] - 1])
                    X = common.readImofDir(
                        'data/CAVE/complete_ms_data/' + dirs[Ind[i] - 1] + '/' + dirs[Ind[i] - 1]
                    )
                    Y = np.tensordot(X, R, (2, 0))
                    sio.savemat(DataRoad + 'X/' + dirs[Ind[i] - 1] + '.mat', {'X': X})
                    sio.savemat(DataRoad + 'Y/' + dirs[Ind[i] - 1] + '.mat', {'Y': Y})
                break

        else:
            print('Using the prepared trainset and initial values in folder data/CAVE/train')

    def all_train_data_in(self, args):
        dataX_list = []
        dataY_list = []

        for root, dirs, files in os.walk(f'data/CAVE/f{args.scale}/train/X/'):
            files.sort()
            for i in range(self.args.trainset_file_num):
                dataX = sio.loadmat(f"data/CAVE/f{args.scale}/train/X/" + files[i])
                dataY = sio.loadmat(f"data/CAVE/f{args.scale}/train/Y/" + files[i])
                dataX_list.append(dataX['X'])
                dataY_list.append(dataY['Y'])

        return dataX_list, dataY_list

    def train_data_in(self, X, Y, B, sizeI, sf, channel=31):
        batch_Z = np.zeros((sizeI // sf, sizeI // sf, channel), 'f')
        px = random.randint(0, self.args.trainset_sizeIX - sizeI)
        py = random.randint(0, self.args.trainset_sizeIY - sizeI)
        subX = X[px:px + sizeI:1, py:py + sizeI:1, :]
        subY = Y[px:px + sizeI:1, py:py + sizeI:1, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # Random rotation
        for j in range(rotTimes):
            subX = np.rot90(subX)
            subY = np.rot90(subY)

        # Random vertical Flip
        for j in range(vFlip):
            subX = subX[:, ::-1, :]
            subY = subY[:, ::-1, :]

        # Random Horizontal Flip
        for j in range(hFlip):
            subX = subX[::-1, :, :]
            subY = subY[::-1, :, :]

        batch_X = subX
        batch_Y = subY

        for i in range(channel):
            subZ = np.real(np.fft.ifft2(np.fft.fft2(batch_X[:, :, i]) * B))
            subZ = subZ[0:sizeI:sf, 0:sizeI:sf]
            batch_Z[:, :, i] = subZ

        return batch_X, batch_Y, batch_Z


class TestSet(Dataset):
    def __init__(self, args):
        self.args = args
        data = sio.loadmat('data/CAVE/response coefficient')
        self.R = data['R']
        self.B = data['B512']

        self.PrepareDataAndiniValue(self.R, self.B, self.args.scale, self.args.prepare, self.args.channel)
        self._set_dataset_length()

    def __getitem__(self, idx):
        X, Y, Z, filename = self.all_test_data_in(self.args, idx)

        D, A = common.Upsample(self.args, Y, Z, self.R.T, self.B, self.args.scale)
        A = common.np2Tensor(
            A, data_range=self.args.data_range
        )
        return X, Y, Z, D, A, filename

    def __len__(self):
        return self.dataset_length

    def _set_dataset_length(self):
        self.dataset_length = self.args.testset_file_num

    # Prepare dataset for testing
    def PrepareDataAndiniValue(self, R, B, sf, prepare='Yes', channel=31):
        DataRoad = f'data/CAVE/f{sf}/test/'
        if prepare != 'No':
            print('Generating the testing dataset in folder data/CAVE/test')
            # random index will become testing dataset
            Ind = [18, 17, 21, 3, 9, 4, 20, 5, 16, 32, 11, 1]

            common.mkdir(DataRoad + 'X/')
            common.mkdir(DataRoad + 'Y/')
            common.mkdir(DataRoad + 'Z/')

            for root, dirs, files in os.walk('data/CAVE/complete_ms_data/'):
                dirs.sort()
                for i in range(self.args.testset_file_num):
                    Z = np.zeros([self.args.testset_sizeI // sf, self.args.testset_sizeI // sf, channel])
                    print('processing ' + dirs[Ind[i] - 1])
                    X = common.readImofDir(
                        'data/CAVE/complete_ms_data/' + dirs[Ind[i] - 1] + '/' + dirs[Ind[i] - 1]
                    )
                    Y = np.tensordot(X, R, (2, 0))
                    for j in range(channel):
                        subZ = np.real(np.fft.ifft2(np.fft.fft2(X[:, :, j]) * B))
                        subZ = subZ[0::sf, 0::sf]
                        Z[:, :, j] = subZ
                    sio.savemat(DataRoad + 'X/' + dirs[Ind[i] - 1] + '.mat', {'X': X})
                    sio.savemat(DataRoad + 'Y/' + dirs[Ind[i] - 1] + '.mat', {'Y': Y})
                    sio.savemat(DataRoad + 'Z/' + dirs[Ind[i] - 1] + '.mat', {'Z': Z})
                break

        else:
            print('Using the prepared testset and initial values in folder data/CAVE/test')

    def all_test_data_in(self, args, idx):
        for root, dirs, files in os.walk(f'data/CAVE/f{args.scale}/test/X/'):
            filename = files[idx]
            data = sio.loadmat(f'data/CAVE/f{args.scale}/test/X/' + filename)
            X = data['X']
            data = sio.loadmat(f'data/CAVE/f{args.scale}/test/Y/' + filename)
            Y = data['Y']
            data = sio.loadmat(f'data/CAVE/f{args.scale}/test/Z/' + filename)
            Z = data['Z']

        return X, Y, Z, filename