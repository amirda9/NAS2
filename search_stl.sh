python3 search.py \
--gpu_ids 0 \
--dataset stl10 \
--latent_dim 120 \
--arch arch_cifar10 \
--exp_name arch_search_stl10
--bottom_width 6 \
--img_size 48 \
--max_epoch_G 120 \
--n_critic 5 \
--latent_dim 120 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--val_freq 5 \

