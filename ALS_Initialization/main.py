import torch
import time
import argparse
import os
import logging.config
import matplotlib.pyplot as plt
import utils_init
import numpy as np
from model import lenet, vgg
import torchvision.models as models

parser = argparse.ArgumentParser('Input manually designed chain.')

parser.add_argument(
    '--type_init',
    type=str,
    default='ALS',
    choices=['ALS', 'BP', 'ALS2', 'ALS3'],
    help='Chose the type of initialization method')

parser.add_argument(
    '--rsvd',
    type=int,
    default=1,
    choices=[0, 1],
    help='Use a random generated small matrix t reduce the matrix size or not')

parser.add_argument(
    '--rsvd_size',
    type=int,
    nargs='+',
    help='The #rows and #columns of the smaller random matrices used in random SVD, respectively')

parser.add_argument(
    '--F_path',
    type=str,
    default='none',
    help='Path of the matrix that need to be approximated')

parser.add_argument(
    '--pth_path',
    type=str,
    default='none',
    help='Path of the pretrained model params')

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
    '--iter',
    type=int,
    default=5,
    help='The number of iterations to initiate the DeBut factors')

parser.add_argument(
    '--lr',
    type=float,
    default=0.1,
    help='Learning rate for BP initialization')

parser.add_argument(
    '--lr_decay',
    type=float,
    default=0.5,
    help='Weight decay rate')

parser.add_argument(
    '--decay_epoch',
    type=int,
    default=10,
    help='Weight decay after required epochs')

parser.add_argument(
    '--log_path',
    type=str,
    help='Store the experimental info')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

parser.add_argument(
    '--pretrained_model',
    type=int,
    default=1,
    choices=[0, 1],
    help='To use real data or not')

parser.add_argument(
    '--model',
    type=str,
    default='vgg',
    choices=['vgg','lenet','vgg16_bn','resnet_50'],
    help='Model architecture')
    
parser.add_argument(
    '--layer_name',
    type=str,
    help='Decide which layer to be approximated')
    
parser.add_argument(
    '--layer_type',
    type=str,
    default='conv',
    choices=['conv', 'fc'],
    help='The type of the given layer')

args = parser.parse_args()

torch.cuda.device_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.manual_seed_all(806)


# shape test
# a = torch.randn([512*4068, 2048]).to(device)
# print(a.shape)

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


def DeBut_init():
    # log path
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    setup_logging(log_file=args.log_path + '/log.txt', filemode='w')

    # deal with input sup & sub
    sup = utils_init.trans_args_sup(args.sup)
    sub = utils_init.trans_args_sub(args.sub)

    # get the approximated matrix
    if args.F_path == 'none':
        if args.pretrained_model == 1:
            
            # vgg
            if args.model == 'vgg':
                model = vgg.vgg_16_bn([0]*13)
                print(model)
                checkpoint = torch.load(args.pth_path, map_location='cuda:0')
                model.load_state_dict(checkpoint['state_dict'])
                
            # lenet
            if args.model == 'lenet':
                model = lenet.LeNet()
                print(model)
                checkpoint = torch.load(args.pth_path, map_location='cuda:0')
                model.load_state_dict(checkpoint['state_dict'])
                
            # vgg16_bn
            if args.model == 'vgg16_bn':
                model = models.vgg16_bn(pretrained=True)
                print(model)
                
            # resnet_50
            if args.model == 'resnet_50':
                model = models.resnet50(pretrained=True)
                print(model)
            
            # load pretrianed weights
            loc = locals()
            exec_str = 'D=model.' + args.layer_name + '.weight.data'
            exec(exec_str)
            D = loc['D'].to(device)

            if args.layer_type == 'conv':
                # get the shape of D
                C_out = D.shape[0]
                kk_C_in = D.shape[2]*D.shape[3]*D.shape[1]
                # reshape 4-D tensor to the matrix
                D = D.permute(0,2,3,1)
                D = torch.reshape(D, [C_out, kk_C_in]).to(device)
                
            print(D, D.shape)
            
        else:
            num_factors = len(sup)
            r = sup[0][0]
            c = sup[num_factors - 1][1]
            D = torch.rand([r, c]).to(device)
    else:
        D = torch.load(args.F_path)
        D = D.to(device)
        # D = np.load(args.F_path)
        # D = D.to(device)

    if args.type_init == 'ALS':
        MaxItr = args.iter
        Butterfly = utils_init.Generate_Chain(sup, sub)
        num_mat = len(Butterfly)
        D_approx = torch.eye(sup[0][0]).to(device)
        error = []

        for i in range(0, num_mat):
            Butterfly[i].to(device)

        for k in range(0, num_mat):
            D_approx = torch.mm(D_approx, Butterfly[k].to(device))
        error_init = torch.norm(D_approx - D) / torch.norm(D)

        logging.info('-' * 30)
        logging.info('Size of the approximated matrix: [{}, {}].'.format(sup[0][0], sup[num_mat - 1][1]))
        logging.info('Number of Butterfly factors: {}.'.format(num_mat))
        logging.info('Maximum number of iterations: {}.'.format(MaxItr))
        logging.info('Superscripts: ' + str(sup) + '.')
        logging.info('Subscripts: ' + str(sub) + '.')
        logging.info('Initialization method: ' + args.type_init)
        logging.info('norm(D) = {}.'.format(torch.norm(D)))
        logging.info('Relative L2 error of D before Butterfly-ALS init: {}.'.format(error_init))
        if args.rsvd == 1:
            logging.info('Use Random SVD to help calculate the pseudo-inverse: True')
            logging.info('The #rows of the smaller random matrix: ' + str(args.rsvd_size))
        if args.rsvd == 0:
            logging.info('Use Random SVD to help calculate the pseudo-inverse: False')
        logging.info('-' * 30)

        flag_dirc = 1

        start = time.time()
        for i in range(0, args.iter):
            # flag_dirc = 1: from left to right
            if flag_dirc == 1:
                for k in range(0, num_mat):
                    sup_cur = sup[k]
                    sub_cur = sub[k]
                    # A_cell, C_cell = utils_init.Find_Blocks(Butterfly, sup, sub, k)
                    # AC_mat = utils_init.Get_Vector_Form(A_cell, C_cell, sup_cur, sub_cur)
                    # print(AC_mat.shape)

                    if args.rsvd == 1:
                        rsvd_mat1 = torch.randn([args.rsvd_size[0], sup[0][0]]).to(device)
                        rsvd_mat2 = torch.randn([sup[num_mat - 1][1], args.rsvd_size[1]]).to(device)
                        A_cell, C_cell = utils_init.Find_Blocks(device, Butterfly, sup, sub, k, rsvd=args.rsvd,
                                                                T1=rsvd_mat1, T2=rsvd_mat2)
                        AC_mat = utils_init.Get_Vector_Form(device, A_cell, C_cell, sup_cur, sub_cur)
                        D_small = torch.mm(rsvd_mat1.to(device), torch.mm(D, rsvd_mat2.to(device)))
                        B_vec = torch.mm(torch.pinverse(AC_mat).to(device),
                                         torch.reshape(D_small, [args.rsvd_size[0] * args.rsvd_size[1], 1]))
                    else:
                        A_cell, C_cell = utils_init.Find_Blocks(device, Butterfly, sup, sub, k)
                        AC_mat = utils_init.Get_Vector_Form(device, A_cell, C_cell, sup_cur, sub_cur)
                        B_vec = torch.mm(torch.pinverse(AC_mat).to(device),
                                         torch.reshape(D, [sup[0][0] * sup[num_mat - 1][1], 1]))

                    Butterfly[k] = utils_init.Vec_to_Mat(B_vec, sup_cur, sub_cur)

            # flag_dirc = 0: from right to left
            if flag_dirc == 0:
                for k in range(num_mat - 1, -1, -1):
                    sup_cur = sup[k]
                    sub_cur = sub[k]
                    # A_cell, C_cell = utils_init.Find_Blocks(Butterfly, sup, sub, k)
                    # AC_mat = utils_init.Get_Vector_Form(A_cell, C_cell, sup_cur, sub_cur)
                    # print(AC_mat.shape)

                    if args.rsvd == 1:
                        rsvd_mat1 = torch.randn([args.rsvd_size[0], sup[0][0]]).to(device)
                        rsvd_mat2 = torch.randn([sup[num_mat - 1][1], args.rsvd_size[1]]).to(device)
                        A_cell, C_cell = utils_init.Find_Blocks(device, Butterfly, sup, sub, k, rsvd=args.rsvd,
                                                                T1=rsvd_mat1, T2=rsvd_mat2)
                        AC_mat = utils_init.Get_Vector_Form(device, A_cell, C_cell, sup_cur, sub_cur)
                        D_small = torch.mm(rsvd_mat1.to(device), torch.mm(D, rsvd_mat2.to(device)))
                        B_vec = torch.mm(torch.pinverse(AC_mat).to(device),
                                         torch.reshape(D_small, [args.rsvd_size[0] * args.rsvd_size[1], 1]))
                    else:
                        A_cell, C_cell = utils_init.Find_Blocks(device, Butterfly, sup, sub, k)
                        AC_mat = utils_init.Get_Vector_Form(device, A_cell, C_cell, sup_cur, sub_cur)
                        B_vec = torch.mm(torch.pinverse(AC_mat).to(device),
                                         torch.reshape(D, [sup[0][0] * sup[num_mat - 1][1], 1]))

                    Butterfly[k] = utils_init.Vec_to_Mat(B_vec, sup_cur, sub_cur)

            # approximate D
            D_approx = torch.eye(sup[0][0])
            for k in range(0, num_mat):
                D_approx = torch.mm(D_approx.to(device), Butterfly[k].to(device))

            # check error when know the real D
            error.append(torch.norm(D_approx - D) / torch.norm(D))

            # change the flag
            if flag_dirc == 0:
                flag_dirc = 1
            else:
                flag_dirc = 0

            logging.info('No.' + str(i + 1) + ' Iter:\t' + '||D_approx - D||2/||D||2 = '
                         + str(error[i].detach()) + '.')

            is_best = False
            if i > 0:
                if error[i] < error[i - 1]:
                    is_best = True

            DeBut_factors = Butterfly

            # save the results
            utils_init.save_DeBut({
                'DeBut_factors': DeBut_factors,
                'D_approx': D_approx,
                'error': error,
            }, is_best, args.log_path)

        end = time.time()

        logging.info('-' * 30)
        logging.info('Minimum relative error (L2 norm) of approximated D: {}.'.format(min(error)))
        logging.info('Time used: ' + str(end - start) + 's.')
        logging.info('-' * 30)

    if args.type_init == 'BP':
        MaxItr = args.iter
        Butterfly = utils_init.Generate_Chain(sup, sub)
        num_mat = len(Butterfly)
        D_approx = torch.eye(sup[0][0]).to(device)
        error = []
        input_size = Butterfly[num_mat - 1].shape[1]
        DeBut_nz_idx = []
        lr = args.lr

        for k in range(0, num_mat):
            D_approx = torch.mm(D_approx, Butterfly[k].to(device))
        error_init = torch.norm(D_approx.to(device) - D.to(device)) / torch.norm(D)

        logging.info('-' * 30)
        logging.info('Size of the approximated matrix: [{}, {}].'.format(sup[0][0], sup[num_mat - 1][1]))
        logging.info('Number of Butterfly factors: {}.'.format(num_mat))
        logging.info('Maximum number of iterations: {}.'.format(MaxItr))
        logging.info('Superscripts: ' + str(sup) + '.')
        logging.info('Subscripts: ' + str(sub) + '.')
        logging.info('Initialization method: ' + args.type_init)
        logging.info('Learning rate at beginning: ' + str(args.lr))
        logging.info('Learning rate decay rate: ' + str(args.lr_decay))
        logging.info('Learning rate decay afte every {} epochs'.format(args.decay_epoch))
        logging.info('norm(D) = {}.'.format(torch.norm(D)))
        logging.info('Relative L2 error of D before Butterfly-BP init: {}.'.format(error_init))
        logging.info('-' * 30)

        # requires_grad = True
        for i in range(0, num_mat):
            Butterfly[i].to(device)
            Butterfly[i].requires_grad = True
            DeBut_nz_idx.append(torch.eq(Butterfly[i], 0).nonzero())

        start = time.time()
        # train to init
        for k in range(0, MaxItr):
            # generate random input x
            x = torch.rand([int(input_size), 1]).to(device)
            x.requires_grad = True

            # calculate the output
            y = torch.mm(D, x).to(device)

            # forward
            for j in range(num_mat - 1, -1, -1):
                x = torch.mm(Butterfly[j].to(device), x)

            # loss
            loss = torch.norm(y - x).to(device)

            # backward
            loss.backward(retain_graph=True)

            # delete the grad of the zeros
            for m in range(0, num_mat):
                for n in range(0, len(DeBut_nz_idx[m])):
                    Butterfly[m].grad[list(DeBut_nz_idx[m][n])] = 0

            # update the DeBut factors
            for p in range(0, num_mat):
                Butterfly[p] = Butterfly[p] - lr * Butterfly[p].grad

            # approximate D
            D_approx = torch.eye(sup[0][0])
            for q in range(0, num_mat):
                D_approx = torch.mm(D_approx.to(device), Butterfly[q].to(device))

            # check error when know the real D
            error.append(torch.norm(D_approx.to(device) - D.to(device)) / torch.norm(D.to(device)))

            logging.info('No.' + str(k + 1) + ' Iteration: ' + '||D_approx - D||2/||D||2 = '
                         + str(error[k].detach()) + '.')

            # renew DeBut factors
            for i in range(0, num_mat):
                Butterfly[i] = Butterfly[i].detach()
                Butterfly[i].requires_grad = True

            # adjust learning rate
            if k % (args.decay_epoch - 1) == 0:
                lr = 0.5 * args.lr_decay

            DeBut_factors = Butterfly

            # save the results
            is_best = False
            if k > 0:
                if error[k] < error[k - 1]:
                    is_best = True

            utils_init.save_DeBut({
                'DeBut_factors': DeBut_factors,
                'D_approx': D_approx,
                'error': error,
            }, is_best, args.log_path)

        end = time.time()

        logging.info('-' * 30)
        logging.info('Minimum relative error (L2 norm) of approximated D: {}.'.format(min(error)))
        logging.info('Time used: ' + str(end - start) + 's.')
        logging.info('-' * 30)

    if args.type_init == 'ALS2':
        MaxItr = args.iter
        Butterfly = utils_init.Generate_Chain(sup, sub)
        num_mat = len(Butterfly)
        D_approx = torch.eye(sup[0][0]).to(device)
        error = []
        DeBut_zeros_idx = []

        for k in range(0, num_mat):
            D_approx = torch.mm(D_approx, Butterfly[k].to(device))
        error_init = torch.norm(D_approx - D) / torch.norm(D)

        # restore the idx of zero elements
        for i in range(0, num_mat):
            DeBut_zeros_idx.append(torch.eq(Butterfly[i], 0).nonzero())

        logging.info('-' * 30)
        logging.info('Size of the approximated matrix: [{}, {}].'.format(sup[0][0], sup[num_mat - 1][1]))
        logging.info('Number of Butterfly factors: {}.'.format(num_mat))
        logging.info('Maximum number of iterations: {}.'.format(MaxItr))
        logging.info('Superscripts: ' + str(sup) + '.')
        logging.info('Subscripts: ' + str(sub) + '.')
        logging.info('Initialization method: ' + args.type_init)
        logging.info('norm(D) = {}.'.format(torch.norm(D)))
        logging.info('Relative L2 error of D before Butterfly-ALS init: {}.'.format(error_init))
        logging.info('-' * 30)

        flag_dirc = 1

        start = time.time()
        for i in range(0, args.iter):
            # flag_dirc = 1: from left to right
            if flag_dirc == 1:
                for k in range(0, num_mat):
                    A, C = utils_init.cal_A_C(Butterfly, sup, k)
                    Butterfly[k] = torch.mm(torch.mm(torch.pinverse(A).to(device), D), torch.pinverse(C).to(device))
                    # keep the DeBut structure
                    for n in range(0, len(DeBut_zeros_idx[k])):
                        Butterfly[k][list(DeBut_zeros_idx[k][n])] = 0

            # flag_dirc = 0: from right to left
            if flag_dirc == 0:
                for k in range(num_mat - 1, -1, -1):
                    A, C = utils_init.cal_A_C(Butterfly, sup, k)
                    Butterfly[k] = torch.mm(torch.mm(torch.pinverse(A).to(device), D), torch.pinverse(C).to(device))
                    # keep the DeBut structure
                    for n in range(0, len(DeBut_zeros_idx[k])):
                        Butterfly[k][list(DeBut_zeros_idx[k][n])] = 0

            # approximate D
            D_approx = torch.eye(sup[0][0])
            for k in range(0, num_mat):
                D_approx = torch.mm(D_approx.to(device), Butterfly[k].to(device))

            # check error when know the real D
            error.append(torch.norm(D_approx - D) / torch.norm(D))

            # change the flag
            if flag_dirc == 0:
                flag_dirc = 1
            else:
                flag_dirc = 0

            logging.info('No.' + str(i + 1) + ' Iter:\t' + '||D_approx - D||2/||D||2 = '
                         + str(error[i].detach()) + '.')

            is_best = False
            if i > 0:
                if error[i] < error[i - 1]:
                    is_best = True

            DeBut_factors = Butterfly

            # save the results
            utils_init.save_DeBut({
                'DeBut_factors': DeBut_factors,
                'D_approx': D_approx,
                'error': error,
            }, is_best, args.log_path)

        end = time.time()

        logging.info('-' * 30)
        logging.info('Minimum relative error (L2 norm) of approximated D: {}.'.format(min(error)))
        logging.info('Time used: ' + str(end - start) + 's.')
        logging.info('-' * 30)

    if args.type_init == 'ALS3':
        MaxItr = args.iter
        Butterfly = utils_init.Generate_Chain(sup, sub)
        # print(Butterfly)
        num_mat = len(Butterfly)
        D_approx = torch.eye(sup[0][0]).to(device)
        error = []

        for i in range(0, num_mat):
            Butterfly[i].to(device)

        for k in range(0, num_mat):
            D_approx = torch.mm(D_approx, Butterfly[k].to(device))
        error_init = torch.norm(D_approx - D) / torch.norm(D)

        logging.info('-' * 30)
        logging.info('Size of the approximated matrix: [{}, {}].'.format(sup[0][0], sup[num_mat - 1][1]))
        logging.info('Number of Butterfly factors: {}.'.format(num_mat))
        logging.info('Maximum number of iterations: {}.'.format(MaxItr))
        logging.info('Superscripts: ' + str(sup) + '.')
        logging.info('Subscripts: ' + str(sub) + '.')
        logging.info('Initialization method: ' + args.type_init)
        logging.info('norm(D) = {}.'.format(torch.norm(D)))
        logging.info('Relative L2 error of D before Butterfly-ALS init: {}.'.format(error_init))
        logging.info('-' * 30)

        flag_dirc = 1

        start = time.time()
        for i in range(0, args.iter):
            # flag_dirc = 1: from left to right
            if flag_dirc == 1:
                for k in range(0, num_mat):
                    # print('{}-th DeBut factor.'.format(k+1))
                    if k == 0:
                        sup_cur = sup[k]
                        sub_cur = sub[k]
                        B_vec = utils_init.segment(device, D, Butterfly, k, sup, sub)
                        Butterfly[k] = utils_init.seg_Vec_to_Mat(B_vec, sup_cur, sub_cur)
                    else:
                        sup_cur = sup[k]
                        sub_cur = sub[k]
                        # print(sup_cur)
                        # print(sub_cur)
                        # print(D.shape, Butterfly[k].shape, sup, sub)
                        B_vec = utils_init.subblocks_2toend(device, D, Butterfly, k, sup, sub)
                        Butterfly[k] = utils_init.Vec_to_Mat(B_vec, sup_cur, sub_cur)

            # flag_dirc = 0: from right to left
            if flag_dirc == 0:
                for k in range(num_mat - 1, -1, -1):
                    # print('{}-th DeBut factor.'.format(k+1))
                    if k == 0:
                        sup_cur = sup[k]
                        sub_cur = sub[k]
                        B_vec = utils_init.segment(device, D, Butterfly, k, sup, sub)
                        Butterfly[k] = utils_init.seg_Vec_to_Mat(B_vec, sup_cur, sub_cur)
                    else:
                        sup_cur = sup[k]
                        sub_cur = sub[k]
                        B_vec = utils_init.subblocks_2toend(device, D, Butterfly, k, sup, sub)
                        Butterfly[k] = utils_init.Vec_to_Mat(B_vec, sup_cur, sub_cur)

            # approximate D
            D_approx = torch.eye(sup[0][0])
            for k in range(0, num_mat):
                D_approx = torch.mm(D_approx.to(device), Butterfly[k].to(device))

            # check error when know the real D
            error.append(torch.norm(D_approx - D) / torch.norm(D))

            # change the flag
            if flag_dirc == 0:
                flag_dirc = 1
            else:
                flag_dirc = 0

            logging.info('No.' + str(i + 1) + ' Iter:\t' + '||D_approx - D||2/||D||2 = '
                         + str(error[i].detach()) + '.')

            is_best = False
            if i > 0:
                if error[i] < error[i - 1]:
                    is_best = True

            DeBut_factors = Butterfly

            # save the results
            utils_init.save_DeBut({
                'DeBut_factors': DeBut_factors,
                'D_approx': D_approx,
                'error': error,
            }, is_best, args.log_path)

        end = time.time()

        logging.info('-' * 30)
        logging.info('Minimum relative error (L2 norm) of approximated D: {}.'.format(min(error)))
        logging.info('Time used: ' + str(end - start) + 's.')
        logging.info('-' * 30)


def NZs():
    sup = utils_init.trans_args_sup(args.sup)
    sub = utils_init.trans_args_sub(args.sub)
    num_factors = len(sup)
    DeBut_NZs = [] 
    for i in range(0, num_factors):
        num_patterns = sup[i][0] / (sub[i][0] * sub[i][2])
        DeBut_NZs.append(num_patterns * sub[i][0] * sub[i][1] * sub[i][2])
    NZR = sum(DeBut_NZs) / (sup[0][0] * sup[num_factors-1][1])
    f = open("./NZR.txt", "a")
    f.write(args.layer_name)
    f.write('\n')
    f.write(str(sup))
    f.write('\n')
    f.write(str(sub))
    f.write('\n')
    f.write(str(NZR))
    f.write('\n')
    f.write('\n')
    f.close()
    print('NZR is saved!')

if __name__ == '__main__':
    DeBut_init()
    NZs()
