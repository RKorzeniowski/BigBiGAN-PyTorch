import torch

from src.training_utils import training_utils
from src.model import discriminators
from src.model import encoders
from src.model import generators


class GAN(torch.nn.Module):
    def generate_imgs(self, cls=None, noise=None, fixed=False):
        if fixed:
            noise = self.fixed_noise
        elif noise is None:
            noise = training_utils.truncated_normal(self.fixed_noise.shape).to(device=self.fixed_noise.device)
        cls = self.cls if cls is None else cls
        img_gen = self.generator(z=noise, cls=cls)
        return img_gen, noise


class BigBiGAN(GAN):
    def __init__(self, generator, encoder, latent_discriminator, img_discriminator, comb_discriminator, fixed_noise, add_noise):
        super().__init__()
        self.generator = generator
        self.encoder = encoder
        self.latent_discriminator = latent_discriminator
        self.img_discriminator = img_discriminator
        self.comb_discriminator = comb_discriminator
        self.fixed_noise = fixed_noise
        self.add_noise = add_noise
        self.cls = None

    def forward(self, img_real, img_gen, z_noise, z_img, cls):

        if self.add_noise > 0:
            noise = torch.randn_like(img_real) * self.add_noise
        else:
            noise = 0

        img_real_out, img_real_score = self.img_discriminator(x=img_real + noise, cls=cls)
        img_gen_out, img_gen_score = self.img_discriminator(x=img_gen + noise, cls=cls)

        z_noise_out, z_noise_score = self.latent_discriminator(z=z_noise)
        z_img_out, z_img_score = self.latent_discriminator(z=z_img)

        comb_real_score = self.comb_discriminator(latent=z_img_out, img=img_real_out)
        comb_gen_score = self.comb_discriminator(latent=z_noise_out, img=img_gen_out)

        output = {
            "img_real_score": img_real_score,
            "img_gen_score": img_gen_score,
            "z_noise_score": z_noise_score,
            "z_img_score": z_img_score,
            "comb_real_score": comb_real_score,
            "comb_gen_score": comb_gen_score,
        }

        if self.cls is None: self.cls = cls.detach()
        return output

    def generate_latent(self, img):
        z_img = self.encoder(img)  # two times bigger img_input
        return z_img

    def req_grad_disc(self, req_grad):
        for p in self.latent_discriminator.parameters():
            p.requires_grad = req_grad
        for p in self.img_discriminator.parameters():
            p.requires_grad = req_grad
        for p in self.comb_discriminator.parameters():
            p.requires_grad = req_grad

    def get_disc_params(self):
        return list(self.comb_discriminator.parameters()) \
               + list(self.latent_discriminator.parameters()) \
               + list(self.img_discriminator.parameters())

    def get_gen_enc_params(self):
        return list(self.generator.parameters()) + list(self.encoder.parameters())

    @classmethod
    def from_config(cls, config):
        fixed_noise = training_utils.truncated_normal((config.bs, config.latent_dim))
        return cls(
            generator=generators.GenBigGAN.from_config(config),
            encoder=encoders.RevNetEnc.from_config(config),
            img_discriminator=discriminators.DiscBigGAN.from_config(config),
            latent_discriminator=discriminators.LatentDisc.from_config(config),
            comb_discriminator=discriminators.CombDisc.from_config(config),
            fixed_noise=fixed_noise.to(config.device),
            add_noise=config.add_noise,
        )


class BigBiWGAN(BigBiGAN):
    def __init__(self, weight_cutoff, **kwargs):
        super(BigBiWGAN, self).__init__(**kwargs)
        self.weight_cutoff = weight_cutoff

    def clip_disc_weights(self):
        self.latent_discriminator.clip_weights(self.weight_cutoff)
        self.img_discriminator.clip_weights(self.weight_cutoff)
        self.comb_discriminator.clip_weights(self.weight_cutoff)

    @classmethod
    def from_config(cls, config):
        fixed_noise = training_utils.truncated_normal((config.bs, config.latent_dim))
        return cls(
            generator=generators.GenBigGAN.from_config(config),
            encoder=encoders.RevNetEnc.from_config(config),
            img_discriminator=discriminators.DiscBigWGAN.from_config(config),
            latent_discriminator=discriminators.LatentWDisc.from_config(config),
            comb_discriminator=discriminators.CombWDisc.from_config(config),
            fixed_noise=fixed_noise.to(config.device),
            weight_cutoff=config.weight_cutoff,
            add_noise=config.add_noise,
        )


class BigGAN(GAN):
    def __init__(self, generator, discriminator, fixed_noise):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.fixed_noise = fixed_noise
        self.cls = None

    def forward(self, img_real, img_gen, cls):
        _, img_real_score = self.img_discriminator(x=img_real, cls=cls)
        _, img_gen_score = self.img_discriminator(x=img_gen, cls=cls)

        output = {
            "img_real_score": img_real_score,
            "img_gen_score": img_gen_score,
        }

        if self.cls is None: self.cls = cls.detach()
        return output

    def req_grad_disc(self, req_grad):
        for p in self.discriminator.parameters():
            p.requires_grad = req_grad

    def get_disc_params(self):
        return self.discriminator.parameters()

    def get_gen_params(self):
        return self.generator.parameters()

    @classmethod
    def from_config(cls, config):
        fixed_noise = training_utils.truncated_normal((config.bs, config.latent_dim))
        return cls(
            generator=generators.GenBigGAN.from_config(config),
            discriminator=discriminators.DiscBigGAN.from_config(config),
            fixed_noise=fixed_noise.to(config.device),
        )
