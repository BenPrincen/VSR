import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_loss(loss, filename):
    for key in loss:
        x = list(range(len(loss[key])))
        y = loss[key]
        plt.plot(x, y, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.savefig(filename)
    plt.close()


# Function to plot and save the loss
def plot_and_save_loss(losses, epoch, plot_name, model_name, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(range(1, epoch + 2), losses["train"], label=f"{model_name} Training Loss")
    plt.plot(range(1, epoch + 2), losses["valid"], label=f"{model_name} Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(save_dir, f"{plot_name}_{model_name}_{epoch}.png"))
    plt.close()


def plot_and_save_metrics(metrics, epoch, plot_name, metric_name, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(range(1, epoch + 2), metrics["train"], label=f"Training {metric_name}")
    plt.plot(range(1, epoch + 2), metrics["valid"], label=f"Valid {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title("Training and Validation metrics")
    plt.savefig(os.path.join(save_dir, f"{plot_name}_{metric_name}_{epoch}.png"))
    plt.close()


def plot_examples(low_res_folder, gen):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open(os.path.join(low_res_folder, file))
        low_res_image = config.test_transform(image=np.asarray(image))["image"]
        save_image((low_res_image * 0.5), f"saved/lr_{file}")
        with torch.no_grad():
            upscaled_img = gen(low_res_image.unsqueeze(0).to(config.DEVICE))
        save_image(upscaled_img * 0.5 + 0.5, f"saved/sr_{file}")
    gen.train()
