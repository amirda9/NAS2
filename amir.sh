python3 search.py \
--gpu_ids 0 \
--dataset stl10 \
--genotypes_exp arch_cifar10 \
--latent_dim 120 \
--arch arch_cifar10 \
--gen_bs 40 \
--dis_bs 20 \
--max_epoch_G 120 \
--n_critic 5 \
--gf_dim 32 \
--df_dim 16 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--img_size 48 \
--beta1 0.5 \
--beta2 0.9 \
--val_freq 5 \
--num_eval_imgs 200 \
--eval_batch_size 20 \
--exp_name search_cifar 

