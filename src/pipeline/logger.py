from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, logging_interval):
        self.writer = SummaryWriter(log_dir)
        self.logging_interval = logging_interval
        self.counter = 0

    @classmethod
    def from_config(cls, config, name):
        log_dir = config.logging_path.format(
            ds_name=config.ds_name,
            model_architecture=config.model_architecture,
            name=name
        )
        return cls(log_dir=log_dir, logging_interval=config.logging_interval)


class BiGANLogger(Logger):
    def __call__(self, epoch, step, disc_loss, gen_enc_loss, *args, **kwargs):
        self.counter += 1
        if step % self.logging_interval == 0:
            self.writer.add_scalar(f'Loss/Disc', disc_loss, self.counter)
            self.writer.add_scalar(f'Loss/GenEnc', gen_enc_loss, self.counter)

            print(f"epoch {epoch}, disc_loss {disc_loss}, gen_enc_loss {gen_enc_loss}")


class GANLogger(Logger):
    def __call__(self, epoch, step, disc_loss, gen_loss, gen_disc_acc,
                 disc_real_acc, disc_fake_acc, *args, **kwargs):
        self.counter += 1
        if step % self.logging_interval == 0:
            self.writer.add_scalar(f'Loss/Disc', disc_loss, self.counter)
            self.writer.add_scalar(f'Loss/Gen', gen_loss, self.counter)
            self.writer.add_scalar(f'Gen/DiscFake', gen_disc_acc, self.counter)
            self.writer.add_scalar(f'Disc/Real', disc_real_acc, self.counter)
            self.writer.add_scalar(f'Disc/Fake', disc_fake_acc, self.counter)
            print(f"epoch {epoch}, disc_loss {disc_loss}, gen_loss {gen_loss}")
            print(f"gen_disc_acc {gen_disc_acc}, disc_real_acc {disc_real_acc}, disc_fake_acc {disc_fake_acc}")


class ClfLogger(Logger):
    def __call__(self, epoch, loss, step, *args, **kwargs):
        self.counter += 1
        if step % self.logging_interval == 0:
            self.writer.add_scalar('Loss', loss, self.counter)
            print(f"epoch {epoch}, loss {loss}")
