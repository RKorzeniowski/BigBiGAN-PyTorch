hparams = {
    # training utils
    "seed": 420,
    "device": "cuda",
    "img_rows": 4,
    "save_img_count": 12,
    "real_imgs_save_path": "./data/{ds_name}/{model_architecture}/real_img/{hparams}",
    "gen_imgs_save_path": "./data/{ds_name}/{model_architecture}/gen_img/{hparams}",
    "logging_path": "./data/{ds_name}/{model_architecture}/logs/{name}",
    "save_model_path": "./data/{ds_name}/{model_architecture}/checkpoints/{hparams}",
    "save_name": "gan",
    "save_model_interval": 50,

    # hparams
    "clf_lr": 2e-4,
    "disc_steps": 2,
    "gen_steps": 1,
    "epochs": 2000,
    "lr_gen": 2e-4,
    "lr_disc": 2e-4,
    "betas": (0.5, 0.999),

    # model params
    "dropout": 0.2,
    "spectral_norm": True,
    "weight_cutoff": 0.00,
    "add_noise": 0,
}
