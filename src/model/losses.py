import torch
import torch.nn.functional as F

# sprobowac uproszczona wersje lossu z BiGAN bez oddzielnych dyskryminatorow dla X i Z.

class BiGANLoss(torch.nn.Module):
    pass
    # def forward(self, output):
    #     real_loss = self.aggregate_scores(
    #         output["img_real_score"],
    #         output["z_img_score"],
    #         output["comb_real_score"],
    #         generated=False,
    #     )
    #
    #     gen_loss = self.aggregate_scores(
    #         output["img_gen_score"],
    #         output["z_noise_score"],
    #         output["comb_gen_score"],
    #         generated=True,
    #     )
    #     return real_loss + gen_loss


class WGeneratorEncoderLoss(BiGANLoss):
    def forward(self, output):
        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]

        gen_loss = torch.mean(img_output_fake) + torch.mean(comb_output_fake) + torch.mean(z_output_fake)
        real_loss = torch.mean(img_output_real) + torch.mean(comb_output_real) + torch.mean(z_output_real)
        total_loss = - gen_loss + real_loss
        return total_loss / 3


class BiWDiscriminatorLoss(BiGANLoss):
    def forward(self, output):
        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]

        gen_loss = torch.mean(img_output_fake) + torch.mean(comb_output_fake) + torch.mean(z_output_fake)
        real_loss = torch.mean(img_output_real) + torch.mean(comb_output_real) + torch.mean(z_output_real)
        total_loss = - real_loss + gen_loss
        return total_loss / 3


class BiDiscriminatorLoss(BiGANLoss):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, output):
        # bigbigan version
        # correct_real_disc = torch.mean(torch.nn.functional.relu(output["img_real_score"]) - output["img_real_score"]
        #            + torch.nn.functional.relu(output["z_img_score"]) - output["z_img_score"]
        #            + torch.nn.functional.relu(output["comb_real_score"]) - output["comb_real_score"]
        #            )
        # correct_fake_disc = torch.mean(torch.nn.functional.relu(output["z_noise_score"]) + output["z_noise_score"]
        #            + torch.nn.functional.relu(output["img_gen_score"]) + output["img_gen_score"]
        #            + torch.nn.functional.relu(output["comb_gen_score"]) + output["comb_gen_score"]
        #            )
        # correct_disc = correct_real_disc + correct_fake_disc

        # bigan version: criterion(out_true, y_true) + criterion(out_fake, y_fake)
        true_label = torch.ones_like(output["comb_real_score"])
        false_label = torch.zeros_like(output["comb_real_score"])

        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]

        # normal loss
        # real_output = self.bce(img_output_real, true_label) + self.bce(z_output_real, true_label) + self.bce(comb_output_real, true_label)
        # fake_output = self.bce(img_output_fake, false_label) + self.bce(z_output_fake, false_label) + self.bce(comb_output_fake, false_label)

        # hinge
        real_output = torch.mean(F.relu(1. - comb_output_real) + F.relu(1. - z_output_real) + F.relu(1. - img_output_real))
        fake_output = torch.mean(F.relu(1. + comb_output_fake) + F.relu(1. + z_output_fake) + F.relu(1. + img_output_fake))

        correct_disc = (real_output + fake_output) / 3

        return correct_disc

    # def aggregate_scores(self, *args, generated):
    #     inputs = [
    #         torch.clamp(arg, min=0) - arg if generated else
    #         torch.clamp(arg, min=0) + arg
    #         for arg in args]
    #     summed_inputs = torch.sum(torch.cat(inputs, dim=-1), dim=-1)
    #     loss = torch.mean(summed_inputs)
    #     return loss


class GeneratorEncoderLoss(BiGANLoss):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, output):
        # correct_real_gen = torch.mean(output["z_img_score"] + output["img_real_score"] + output["comb_real_score"])
        # correct_fake_gen = torch.mean(-1 * (output["z_noise_score"] + output["img_gen_score"] + output["comb_gen_score"]))
        # correct_gen = correct_real_gen + correct_fake_gen
        # return correct_gen

        # bigan version: criterion(out_fake, y_true) + criterion(out_true, y_fake)
        true_label = torch.ones_like(output["comb_real_score"])
        false_label = torch.zeros_like(output["comb_real_score"])

        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]

        # normal loss
        # real_output = self.bce(img_output_real, false_label) + self.bce(z_output_real, false_label) + self.bce(comb_output_real, false_label)
        # fake_output = self.bce(img_output_fake, true_label) + self.bce(z_output_fake, true_label) + self.bce(comb_output_fake, true_label)

        # hinge loss
        real_output = torch.mean(img_output_real + z_output_real + comb_output_real)
        fake_output = torch.mean(img_output_fake + z_output_fake + comb_output_fake)

        correct_gen = (real_output - fake_output) / 3

        return correct_gen

    # def aggregate_scores(self, *args, generated):
    #     inputs = torch.cat(args, dim=-1)
    #     summed_inputs = torch.sum(inputs, dim=-1)
    #     if generated: summed_inputs = summed_inputs * -1
    #     loss = torch.mean(summed_inputs)
    #     return loss
