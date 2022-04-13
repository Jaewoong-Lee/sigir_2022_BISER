import argparse
import warnings
import os

from datetime import datetime
from time import time

import tensorflow as tf
from trainer import Trainer
from util.preprocessor import preprocess_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-m',  default='mf', type=str, required=False, \
                choices=['mf', 'relmf', 'uae', 'iae', 'cjmf', 'macr', 'proposed'])
parser.add_argument('--preprocess_data', default=False, required=False, action='store_true')
parser.add_argument('--unbiased_eval', default=False, required=False, action='store_true')
parser.add_argument('--dataset', default='coat', type=str, required=False, choices=['coat', 'yahoo'])
parser.add_argument('--learning_rate', '-lr', default=0.005, type=float, required=False)
parser.add_argument('--regularization', '-reg', default=0.00001, type=float, required=False)
parser.add_argument('--random_state', '-ran', type=int, default=1, required=False)
parser.add_argument('--hidden', '-hidden',  type=int, default=50, required=False)
# CJMF
parser.add_argument('--alpha_cjmf', type=float, default=220000, help='exp model regularization in cjmf')
parser.add_argument('--beta_cjmf', type=float, default=0.5, help='res regularization in cjmf')
parser.add_argument('--C', type=int, default=8, help='C in cjmf')
# MACR
parser.add_argument('--macr_c', '-macr_c', default=40, type=float, required=False)
parser.add_argument('--macr_alpha', '-macr_alpha', default=1e-2, type=float, required=False)
parser.add_argument('--macr_beta', '-macr_beta', default=1e-3, type=float, required=False)

parser.add_argument('--alpha', '-alpha', default=0.5, type=float, required=False)
parser.add_argument('--clip', '-clip', default=0.1, type=float, required=False)
parser.add_argument('--neg_sample', '-neg_sample', default=10, type=int, required=False)
parser.add_argument('--max_epoch', type=int, default=500, help='number of max epochs to train')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--threshold', type=int, default=4, help='binalized threshold')
parser.add_argument('--wu', type=float, default=0.1, help='regularization for uAE loss in proposed')
parser.add_argument('--wi', type=float, default=0.1, help='regularization for iAE loss in proposed')
parser.add_argument('--best_model_save', default=False, required=False, action='store_true')

if __name__ == "__main__":
    start_time = time()
    now = datetime.now()
    date_now = "%02d%02d_%02d%02d%02d" %(now.month, now.day, now.hour, now.minute, now.second)

    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()
    print("------------------------------------------------------------------------------")
    print(args)
    print("------------------------------------------------------------------------------")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # arguments
    model_name = args.model_name
    dataset_name = args.dataset
    unbiased_eval = args.unbiased_eval

    # hyper-parameters
    lr = args.learning_rate
    reg = args.regularization
    hidden = args.hidden
    batch_size = args.batch_size
    alpha = args.alpha
    threshold = args.threshold
    wu = args.wu
    wi = args.wi

    if dataset_name == 'coat':
        wu = 0.1
        wi = 0.5
    elif dataset_name == 'yahoo':
        wu = 0.9
        wi = 0.1

    if model_name == 'cjmf':
        batch_size = int(batch_size * 1. * (args.C - 1) / args.C)

    if args.preprocess_data:
        preprocess_dataset(data=dataset_name, threshold=threshold, alpha=alpha)

    trainer = Trainer(data=dataset_name, random_state=args.random_state, hidden=hidden, date_now=date_now, max_iters=args.max_epoch, lam=reg, batch_size=batch_size, wu=wu, wi=wi,
                    alpha=alpha, clip=args.clip, eta=lr, model_name=model_name, unbiased_eval=unbiased_eval, neg_sample=args.neg_sample, C=args.C, alpha_cjmf=args.alpha_cjmf, beta_cjmf=args.beta_cjmf, 
                    macr_c = args.macr_c, macr_alpha = args.macr_alpha, macr_beta = args.macr_beta, best_model_save=args.best_model_save)
    trainer.run()

    print('\n', '=' * 25, '\n')
    print(f'Finished Running {model_name}!')
    print(f'Total time: {time() - start_time}')
    print(f'date_now: {date_now}')
    print('\n', '=' * 25, '\n')
