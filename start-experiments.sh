# det experiment create experiments/full/celeba_msg_ema_ralsgan_0gp_pack2_prelu_pixel_128.yaml .
# det experiment create experiments/full/celeba_msg_ralsgan_0gp_pack2_prelu_pixel_128.yaml .
# det experiment create experiments/pairgan/celeba_msg_ema_ttur_pack2_prelu_pixel_128.yaml .
# det experiment create experiments/pairgan/celeba_msg_ema_ttur_pack4_prelu_pixel_128.yaml .

# det experiment create experiments/full/celeba_msg_ema_ralsgan_0gp_pack1_prelu_pixel_128.yaml .
# det experiment create experiments/full/celeba_msg_ema_ralsgan_0gp_pack1_lrelu_pixel_128.yaml .

# det experiment create experiments/full/celeba_msg_ema_ralsgan_0gp_pack2_prelu_pixel_128.yaml . # good
# det experiment create experiments/full/celeba_msg_ema_ralsgan_0gp_pack2_lrelu_pixel_128.yaml . # good

# det experiment create experiments/full/celeba_msg_ema_ralsgan_wgandiv_pack2_prelu_pixel_128.yaml .

det experiment create ./msg_gan_const.yaml . --test
