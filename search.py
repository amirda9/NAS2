

import myparser 
import os
import cifar10
import numpy as np
import torch.nn as nn
import torch
import data
import tqdm
from copy import deepcopy
import cifar10
import search_all as myarch
from inception_score import _init_inception
from fid_score import create_inception_graph, check_or_download_inception

from pathlib import Path
import torchvision
from PIL import Image

from network import train, validate, LinearLrDecay, load_params, copy_params
from candidate import candidate_G, candidate_G, draw_graph_G, draw_graph_D
import data
from architect import Architect_dis, Architect_gen

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True




def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

def main():
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(5321)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    _init_inception()
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    basemodel_gen = myarch.Generator(args=args)
    basemodel_dis = myarch.Discriminator(args=args)
    gen_net = basemodel_gen.cuda(torch.device('cuda:0'))
    dis_net = basemodel_dis.cuda(torch.device('cuda:0'))


    architect_gen = Architect_gen(gen_net, args)
    architect_dis = Architect_dis(dis_net, args)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set optimizer
    arch_params_gen = gen_net.arch_parameters()
    arch_params_gen_ids = list(map(id, arch_params_gen))
    weight_params_gen = filter(lambda p: id(p) not in arch_params_gen_ids, gen_net.parameters())
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, weight_params_gen),
                                     args.g_lr, (args.beta1, args.beta2))

    arch_params_dis = dis_net.arch_parameters()
    arch_params_dis_ids = list(map(id, arch_params_dis))
    weight_params_dis = filter(lambda p: id(p) not in arch_params_dis_ids, dis_net.parameters())
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, weight_params_dis),
                                     args.d_lr, (args.beta1, args.beta2))

    # set up data_loader
    if args.dataset.lower() == 'imagenette':
        dataset = data.ImagenetteDataset(patch_size=32, validation=False, should_normalize=True)
        train_loader = dataset.get_loader(batch_size=20, shuffle=True, num_workers=2)
    else:
        dataset = data.ImageDataset(args)
        train_loader = dataset.train


    # epoch number for dis_net
    args.max_epoch_D = 100 * args.n_critic
    args.max_iter_D = args.max_epoch_D * len(train_loader)

    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter_D)


    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() == 'imagenette':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)



    # initial                                 
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0



    # search loop
    for epoch in tqdm.tqdm(range(int(start_epoch), int(args.max_epoch_D)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) 

        # search arch and train weights
        if epoch > 0:
            train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
                  lr_schedulers, architect_gen=architect_gen, architect_dis=architect_dis)

        # save and visualise current searched arch
        if epoch == 0 or epoch % args.derive_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            genotype_G = candidate_G(gen_net.alphas_normal, gen_net.alphas_up, save=True,
                                        file_path=os.path.join('./exps/', str(epoch) + '_G.npy'))
            genotype_D = candidate_G(dis_net.alphas_normal, dis_net.alphas_down, save=True,
                                       file_path=os.path.join('./exps/', str(epoch) + '_D.npy'))
            draw_graph_G(genotype_G, save=True, file_path=os.path.join('./exps/', str(epoch) + '_G'))
            draw_graph_D(genotype_D, save=True, file_path=os.path.join('./exps/', str(epoch) + '_D'))


        # validate current searched arch
        if epoch == 0 or epoch % args.val_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)

            inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net)
            print(f'epoch: {epoch}, inception_score: {inception_score}, std: {std}, fid_score: {fid_score}')
            pass

if __name__ == '__main__':
    args = myparser.parse_args()
    main()

