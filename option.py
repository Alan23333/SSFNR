import argparse

parser = argparse.ArgumentParser(description='XXX')

parser.add_argument('--prepare', type=str, default='No',
                    help='if reprepare dataset samples for training and testing, Yes|No')
parser.add_argument('--n_threads', type=int, default=8,
                    help='number of threads for dataset loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='dataset directory')

parser.add_argument('--model', type=str, default='SSFNR')
parser.add_argument('--dataset', type=str, default='CAVE',
                    help='Which dataset to train and test')
parser.add_argument('--scale', type=int, default=8,
                    help='super resolution scale')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--train_num', type=int, default=16000)
parser.add_argument('--gpu_ids', default='0')

parser.add_argument('--pre_train', type=str, default='experiment/CAVE/f8/SSFNR/model/SSFNR.pt',
                    help='pre-trained model directory')
parser.add_argument('--patch_size', type=int, default=96,
                    help='output patch size')
parser.add_argument('--data_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--negval', type=float, default=0.2,
                    help='Negative value parameter for Leaky ReLU')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model, store_true|store_false')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration, L1|MSE')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
parser.add_argument('--save', type=str, help='file name to save')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_false',
                    help='save output results')

args = parser.parse_args(args=[])