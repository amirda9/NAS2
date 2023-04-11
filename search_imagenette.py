

import myparser 
import os
import cifar10
import numpy as np
import torch.nn as nn
import torch
import data
import tqdm
from copy import deepcopy
import archs
import archs.search_both_cifar10 as myarch
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

from pathlib import Path
import torchvision
from PIL import Image

from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.genotype import alpha2genotype, beta2genotype, draw_graph_G, draw_graph_D
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
        train_loader = dataset.get_loader(batch_size=20, shuffle=True, num_workers=args.num_workers)
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


    # create new log dir
    args.path_helper = set_log_dir('exps', args.exp_name)

    # search loop
    for epoch in tqdm.tqdm(range(int(start_epoch), int(args.max_epoch_D)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) 

        # search arch and train weights
        if epoch > 0:
            train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
                  lr_schedulers, architect_gen=architect_gen, architect_dis=architect_dis)

        # save and visualise current searched arch
        if epoch == 0 or epoch % args.derive_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            genotype_G = alpha2genotype(gen_net.alphas_normal, gen_net.alphas_up, save=True,
                                        file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch) + '_G.npy'))
            genotype_D = beta2genotype(dis_net.alphas_normal, dis_net.alphas_down, save=True,
                                       file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch) + '_D.npy'))
            draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch) + '_G'))
            draw_graph_D(genotype_D, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch) + '_D'))

            avg_gen_net = deepcopy(gen_net)
            load_params(avg_gen_net, gen_avg_param)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.arch,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'path_helper': args.path_helper
            }, False, args.path_helper['ckpt_path'])
            del avg_gen_net

        # validate current searched arch
        if epoch == 0 or epoch % args.val_freq == 0 or epoch == int(args.max_epoch_D) - 1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)

            inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net)
            logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                        f'FID score: {fid_score} || @ epoch {epoch}.')
            pass

if __name__ == '__main__':
    args = myparser.parse_args()
    main()

