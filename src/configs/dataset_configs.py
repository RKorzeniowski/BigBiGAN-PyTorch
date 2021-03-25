# 28x28 FMNIST, MNIST, CIFAR10
# 496x387 Imagenet -> 256x256

# gen_disc_ch = 64
# enc_ch = 32

img_grayscale_32x32_config = {
    "gen_mult_chs": {"pre": (256, 128), "post": (64, ), "colors": 1},
    "disc_mult_chs": {"colors": 1, "pre": (64, ), "post": (128, 256)},
    "enc_mult_chs": {"colors": 1, "blocks": (32, 32, 64, 64, 128, 128)},
    "enc_hidden": 256,
    "enc_in_mlp_dim": 128,
    "ks": 3,
    "image_size": 32,
    "latent_disc_blocks": 3,
    "latent_disc_mlp_dim": 64,
    "comb_disc_blocks": 3,
    "comb_disc_mlp_dim": 64,
    "embedding_dim": 32,
    "latent_dim": 100,
    "enc_out_dim": 100,
    "bs": 256,
    # enc 32 ch
}

img_color_32x32_config = {
    "gen_mult_chs": {"pre": (1024, 512), "post": (512, ), "colors": 3},
    "disc_mult_chs": {"colors": 3, "pre": (512, ), "post": (512, 1024)},
    "enc_mult_chs": {"colors": 3, "blocks": (64, 64, 128, 128, 256, 256, 512, 512)}, # output -> enc_in_mlp_dim
    "enc_hidden": 256,
    "enc_in_mlp_dim": 512,
    "ks": 3,
    "image_size": 32,
    "latent_disc_blocks": 5,
    "latent_disc_mlp_dim": 128,
    "comb_disc_blocks": 5,
    "comb_disc_mlp_dim": 128,
    "embedding_dim": 64,
    "latent_dim": 100,
    "enc_out_dim": 100,
    "bs": 128,
    # enc 32 ch
}

# img_color_32x32_config = {
#     "gen_mult_chs": {"pre": (1024, 512), "post": (256, ), "colors": 3},
#     "disc_mult_chs": {"colors": 3, "pre": (256, ), "post": (512, 1024)},
#     "enc_mult_chs": {"colors": 3, "blocks": (64, 64, 128, 128, 256, 256)}, # output -> enc_in_mlp_dim
#     "enc_hidden": 256,
#     "enc_in_mlp_dim": 256,
#     "ks": 3,
#     "image_size": 32,
#     "latent_disc_blocks": 5,
#     "latent_disc_mlp_dim": 128,
#     "comb_disc_blocks": 5,
#     "comb_disc_mlp_dim": 128,
#     "embedding_dim": 64,
#     "latent_dim": 100,
#     "enc_out_dim": 100,
#     "bs": 128,
#     # enc 32 ch
# }

img_color_64x64_config = {
    "gen_mult_chs": {"pre": (1024, 512, 256), "post": (128, ), "colors": 3},
    "disc_mult_chs": {"colors": 3, "pre": (128, ), "post": (256, 512, 1024)},
    "enc_mult_chs": {"colors": 3, "blocks": (64, 64, 128, 128, 256, 256)},
    "enc_hidden": 256,
    "enc_in_mlp_dim": 256, #128,
    "ks": 3,
    "image_size": 64,
    "latent_disc_blocks": 5,
    "latent_disc_mlp_dim": 128,
    "comb_disc_blocks": 5,
    "comb_disc_mlp_dim": 128,
    "embedding_dim": 64,
    "latent_dim": 100,
    "enc_out_dim": 100,
    "bs": 64,
}
