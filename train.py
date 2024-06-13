import torch
import config
from torch import nn
from torch import optim
from utils import (
    load_checkpoint,
    save_checkpoint,
    plot_examples,
    plot_loss,
    plot_and_save_loss,
    plot_and_save_metrics,
)
from loss import VGGLoss
from torch.utils.data import DataLoader, random_split
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import RealVSR
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure

torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=True)
    disc_train_loss = 0.0
    gen_train_loss = 0.0
    avg_fake_ssim = 0.0
    avg_true_ssim = 0.0

    for idx, (low_res, high_res) in enumerate(loop):
        print(f"batch: {idx}")
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        fake = gen(low_res)
        ssim_value = structural_similarity_index_measure(fake, high_res)
        print(f"fake vs high res: {ssim_value}")
        avg_fake_ssim += ssim_value.detach().cpu()
        ssim_value = structural_similarity_index_measure(low_res, high_res * 0.5 + 0.5)
        print(f"low res vs high res: {ssim_value}")
        avg_true_ssim += ssim_value.detach().cpu()

        print(f"avg fake ssim: {avg_fake_ssim}")

        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real
        disc_train_loss += loss_disc.detach().cpu().numpy()
        print(f"disc loss: {loss_disc} = {disc_loss_fake} + {disc_loss_real}")

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        l2_loss = mse(fake, high_res)
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = l2_loss + loss_for_vgg + adversarial_loss
        gen_train_loss += gen_loss.detach().cpu()
        print(f"gen loss: {gen_loss}")
        # print(f"gen loss: {gen_loss} = {l2_loss} + {loss_for_vgg} + {adversarial_loss}")

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        # disc_fake = disc(fake)
        # # l2_loss = mse(fake, high_res)
        # adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        # # print(f"adversarial loss: {adversarial_loss}")
        # loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        # # print(f"vgg loss: {loss_for_vgg}")
        # gen_loss = loss_for_vgg + adversarial_loss
        # losses["gen"] += gen_loss
        # print(f"generator loss: {gen_loss} = {loss_for_vgg} + {adversarial_loss}")

        if idx % 10 == 0:
            plot_examples("test_images/", gen)

    dataset_length = len(loader.dataset)
    print(f"True average ssim: {avg_fake_ssim / dataset_length}")
    return (
        gen_train_loss / dataset_length,
        disc_train_loss / dataset_length,
        avg_fake_ssim / len(loader),
        avg_true_ssim / len(loader),
    )


def validate(loader, disc, gen, mse, bce, vgg_loss):
    disc.eval()
    gen.eval()
    disc_val_loss, gen_val_loss = 0.0, 0.0
    avg_fake_ssim = 0.0
    avg_true_ssim = 0.0
    with torch.no_grad():
        for low_res, high_res in loader:
            high_res = high_res.to(config.DEVICE)
            low_res = low_res.to(config.DEVICE)
            fake = gen(low_res)
            ssim_value = structural_similarity_index_measure(fake, high_res)
            print(f"fake vs high res: {ssim_value}")
            avg_fake_ssim += ssim_value.detach().cpu()
            ssim_value = structural_similarity_index_measure(
                low_res, high_res * 0.5 + 0.5
            )
            print(f"low res vs high res: {ssim_value}")
            avg_true_ssim += ssim_value.detach().cpu()

            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(
                disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
            )
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = disc_loss_fake + disc_loss_real
            disc_val_loss += disc_loss.detach().cpu().numpy()
            print(f"disc loss: {disc_loss}")

            disc_fake = disc(fake)
            l2_loss = mse(fake, high_res)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
            gen_loss = l2_loss + loss_for_vgg + adversarial_loss
            gen_val_loss += gen_loss.detach().cpu()
            print(f"gen loss: {gen_loss}")
    dataset_length = len(loader.dataset)
    return (
        gen_val_loss / dataset_length,
        disc_val_loss / dataset_length,
        avg_fake_ssim / len(loader),
        avg_true_ssim / len(loader),
    )


def test(loader, disc, gen, mse, bce, vgg_loss):
    disc.eval()
    gen.eval()
    disc_test_loss, gen_test_loss = 0.0, 0.0
    with torch.no_grad():
        for low_res, high_res in loader:
            high_res = high_res.to(config.DEVICE)
            low_res = low_res.to(config.DEVICE)
            fake = gen(low_res)
            # print(f"fake shape: {fake.shape}")
            # print(f"{fake}")
            # print(f"low_res shape: {low_res.shape}")
            # print(f"{low_res}")
            # print(f"high_res shape: {high_res.shape}")
            # print(f"{high_res}")
            ssim_value = structural_similarity_index_measure(fake, high_res)

            print(f"SSIM: {ssim_value}")
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(
                disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
            )
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = disc_loss_fake + disc_loss_real
            disc_test_loss += disc_loss.detach().cpu().numpy()
            print(f"disc loss: {disc_loss}")

            disc_fake = disc(fake)
            l2_loss = mse(fake, high_res)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
            gen_loss = l2_loss + loss_for_vgg + adversarial_loss
            gen_test_loss += gen_loss.detach().cpu().numpy()
            print(f"gen loss: {gen_loss}")
    return gen_test_loss / len(loader.dataset), disc_test_loss / len(loader.dataset)


def plot_losses_and_metrics(
    disc_losses,
    gen_losses,
    true_ssim_values,
    fake_ssim_values,
    true_psnr_values,
    fake_psnr_values,
    epoch,
):
    plot_and_save_loss(
        gen_losses,
        epoch,
        "losses_150",
        "generator",
    )
    plot_and_save_loss(
        disc_losses,
        epoch,
        "losses_150",
        "discriminator",
    )
    plot_and_save_metrics(fake_ssim_values, epoch, "fake_150", "SSIM")
    plot_and_save_metrics(true_ssim_values, epoch, "true_150", "SSIM")


def main():
    dataset = RealVSR(root_dir="datasets/RealVSR/train/")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    disc_losses = {"train": [], "valid": [], "test": []}
    gen_losses = {"train": [], "valid": [], "test": []}

    true_ssim_values = {"train": [], "valid": [], "test": []}
    fake_ssim_values = {"train": [], "valid": [], "test": []}

    true_psnr_values = {"train": [], "valid": [], "test": []}
    fake_psnr_values = {"train": [], "valid": [], "test": []}

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"train dataset length: {len(train_dataset)}")
    print(f"train dataset length: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=config.NUM_WORKERS,
    # )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999)
    )
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        print(f"epoch: {epoch}")
        gen_train_loss, disc_train_loss, avg_fake_ssim, avg_true_ssim = train_fn(
            train_loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss
        )
        disc_losses["train"].append(disc_train_loss)
        gen_losses["train"].append(gen_train_loss)
        true_ssim_values["train"].append(avg_true_ssim)
        fake_ssim_values["train"].append(avg_fake_ssim)

        gen_val_loss, disc_val_loss, avg_fake_ssim, avg_true_ssim = validate(
            val_loader, disc, gen, mse, bce, vgg_loss
        )
        disc_losses["valid"].append(disc_val_loss)
        gen_losses["valid"].append(gen_val_loss)
        true_ssim_values["valid"].append(avg_true_ssim)
        fake_ssim_values["valid"].append(avg_fake_ssim)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        plot_losses_and_metrics(
            disc_losses,
            gen_losses,
            true_ssim_values,
            fake_ssim_values,
            true_psnr_values,
            fake_psnr_values,
            epoch,
        )


if __name__ == "__main__":
    main()
