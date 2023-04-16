python3 search.py \
--gpu_ids 0 \
--dataset cifar10 \
--genotypes_exp arch_cifar10 \
--latent_dim 120 \
--arch arch_cifar10 \
--gen_bs 40 \
--dis_bs 20 \
--max_epoch_G 120 \
--n_critic 5 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.5 \
--beta2 0.9 \
--val_freq 5 \
--num_eval_imgs 5000 \
--exp_name search_cifar 

