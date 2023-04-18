import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='search', help='train or test')
    parser.add_argument('--data', type=str, default='cifar10', help='cfiar10 or imagenette or stl10')


    opt = parser.parse_args()

    # turn that config into opt
    opt.gpu_ids = 0
    opt.dataset =  opt.data
    opt.latent_dim = 120
    opt.arch = 'arch'+opt.data
    opt.exp_name = 'arch_search_'+opt.data
    opt.bottom_width = 4 if opt.data == 'cifar10' else 6
    opt.img_size = 32 if opt.data == 'cifar10' else 48
    opt.max_epoch_G = 120
    opt.n_critic = 5
    opt.g_lr = 0.0002
    opt.d_lr = 0.0002
    opt.beta1 = 0.5
    opt.beta2 = 0.9
    opt.val_freq = 5
    opt.num_eval_imgs = 50
    opt.eval_batch_size = 10
    opt.gf_dim = 64
    opt.df_dim = 32
    opt.d_spectral_norm = True
    opt.g_spectral_norm = True
    opt.dis_bs = 20
    opt.gen_bs = 40


    return opt