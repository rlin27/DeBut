import torch
import matplotlib.pyplot as plt
import argparse
import os
import logging.config
import utils_init

parser = argparse.ArgumentParser('Input manually designed chain.')

parser.add_argument(
    '--sup',
    type=int,
    nargs='+',
    help='Superscripts of the designed chains',
    required=True)

parser.add_argument(
    '--sub',
    type=int,
    nargs='+',
    help='Subscripts of the designed chains',
    required=True)

parser.add_argument(
    '--log_path',
    type=str,
    help='Store the experimental info')

args = parser.parse_args()


# set up logger
def setup_logging(log_file='log.txt', filemode='a'):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def chain_test(sup, sub):
    # log path
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    setup_logging(log_file=args.log_path + '/log.txt', filemode='w')

    logging.info('Superscripts: ' + str(sup) + '.')
    logging.info('Subscripts: ' + str(sub) + '.')

    num_mat = len(sup)
    Butterfly = utils_init.Generate_Chain(sup, sub)
    NZs = utils_init.count_NZs(sup, sub)
    for i in range(0, len(Butterfly)):
        Butterfly[i] = torch.sign(Butterfly[i])

    plt.figure(1)
    for k in range(num_mat - 1, 0, -1):
        mid = Butterfly[num_mat-1]
        for n in range(num_mat-2, k-2, -1):
            mid = torch.mm(Butterfly[n], mid)
        plt.matshow(mid)
        plt.axis('off')
    plt.savefig(args.log_path + '/chain_test.png', dpi=1200)

    nzs = len(torch.eq(mid, 0).nonzero())
    if nzs == 0:
        logging.info('It is a correct chain.')
    else:
        logging.info('It is a wrong chain.')

if __name__ == '__main__':
    # deal with input sup & sub
    sup = utils_init.trans_args_sup(args.sup)
    sub = utils_init.trans_args_sub(args.sub)
    chain_test(sup, sub)