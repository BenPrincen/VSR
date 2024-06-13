import torch
import config
import os
from PIL import Image
from torch import nn
from torch import optim
from utils import (
    load_checkpoint,
    save_checkpoint,
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
from torchvision.utils import save_image
import numpy as np


def test(loader, disc, gen, mse, bce, vgg_loss):
    disc.eval()
    gen.eval()
    disc_test_loss, gen_test_loss = 0.0, 0.0
    avg_fake_ssim = 0.0
    avg_true_ssim = 0.0
    with torch.no_grad():
        for low_res, high_res in loader:
            high_res = high_res.to(config.DEVICE)
            low_res = low_res.to(config.DEVICE)
            fake = gen(low_res)
            print(f"fake: {fake}")
            ssim_value = structural_similarity_index_measure(fake, high_res)
            print(f"SSIM fake: {ssim_value}")
            avg_fake_ssim += ssim_value
            ssim_value = structural_similarity_index_measure(low_res, high_res * 0.5 + 0.5)
            avg_true_ssim += ssim_value
            print(f"SSIM true: {ssim_value}")
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(
                disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
            )
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = disc_loss_fake + disc_loss_real
            disc_test_loss += disc_loss
            print(f"disc loss: {disc_loss}")

            disc_fake = disc(fake)
            l2_loss = mse(fake, high_res)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
            gen_loss = l2_loss + loss_for_vgg + adversarial_loss
            gen_test_loss += gen_loss
            print(f"gen loss: {gen_loss}")
    dataset_length = len(loader.dataset)
    return (
        gen_test_loss / dataset_length,
        disc_test_loss / dataset_length,
        avg_fake_ssim / len(loader),
        avg_true_ssim / len(loader),
    )

def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open(os.path.join(low_res_folder, file))
        low_res_image = config.test_transform(image=np.asarray(image))["image"]
        with torch.no_grad():
            upscaled_img = gen(low_res_image.unsqueeze(0).to(config.DEVICE))
        save_image(upscaled_img * 0.5 + 0.5, f"new_test/sr_{file}")

def main():
    dataset = RealVSR(root_dir="datasets/RealVSR/test")

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
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

    gen_loss, disc_loss, true_ssim_value, fake_ssim_value = test(
        loader, disc, gen, mse, bce, vgg_loss
    )

    plot_examples("new_test/", gen)

    print(f"Generator loss: {gen_loss}")
    print(f"Discriminator loss: {disc_loss}")
    print(f"SSIM sr vs hr: {fake_ssim_value}")
    print(f"SSIM lr vs hr: {true_ssim_value}")


if __name__ == "__main__":
    main()
