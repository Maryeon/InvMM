import os
import os.path as osp
import yaml
import glob
import visdom
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from torchvision import transforms as T

from PIL import Image
from einops import reduce
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.util import instantiate_from_config, default, exists
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import extract_into_tensor
from utils import SmoothedValue


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-dp", "--data_path", type=str, required=True,
        help="root path of dataset"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="configs/latent-diffusion/inversion.yaml",
        help="config file"
    )
    parser.add_argument(
        "--ckpt_file", default="models/ldm/ffhq256/model.ckpt", type=str,
        help="pretrained checkpoint"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-1,
        help="learning rate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=2000,
        help="number of train steps"
    )
    parser.add_argument(
        "--results_folder", default="checkpoints/test", type=str,
        help="path to save training and sampling results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16,
        help="number of samples",
    )
    parser.add_argument(
        "--visdom", action="store_true",
        help="log with visdom",
    )
    parser.add_argument(
        "--init_kl_weight", type=float, default=1,
        help="initial weight of kl divergence"
    )

    return parser


def setup_logger(log_path=None, log_level=logging.INFO):
    logger = logging.root
    logger.setLevel(log_level)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    if log_path is not None:
        log_file = os.path.join(log_path, "log.txt")
        os.makedirs(log_path, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return


def divisible_by(numer, denom):
    return (numer % denom) == 0


def main():
    parser = get_parser()
    args = parser.parse_args()

    setup_logger(args.results_folder)
    logger = logging.getLogger("main")

    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    os.makedirs(args.results_folder, exist_ok=True)
    with open(os.path.join(args.results_folder, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    device = torch.device("cuda:0")
    logger.info(f"Using device {device}")

    dataset = FacesHQ(args.data_path, 256)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size = 1, 
        shuffle=False, pin_memory = True
    )

    config = OmegaConf.load(args.config)

    model = instantiate_from_config(config.model)
    model.to(device)
    ckpt = torch.load(args.ckpt_file, map_location = "cpu")
    m, u = model.load_state_dict(ckpt["state_dict"], strict = False)
    logger.info(f"missing keys: {m}")
    logger.info(f"unexpected keys: {u}")

    sampler = DDIMSampler(model)

    trainer = Trainer(
        model,
        sampler,
        dataloader,
        logger,
        device = device,
        train_lr = args.lr,
        train_num_steps = args.num_steps,
        num_samples = args.num_samples,
        results_folder = args.results_folder,
        enable_visdom = args.visdom,
        init_kl_weight = args.init_kl_weight
    )
    trainer.train()


class LatentDiffusionInversion(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.first_stage_model.requires_grad_(False)
        self.first_stage_model.eval()
        # self.model.requires_grad_(False)
        self.model.eval()

    @torch.inference_mode()
    def sample(self, sampler, noise):
        batch_size = noise.size(0)
        shape = (3, 64, 64)
        samples, _ = sampler.sample(50, batch_size = batch_size, shape = shape, eta = 0., verbose = False, x_T = noise)
        samples = self.decode_first_stage(samples)

        samples = (samples + 1.) / 2.
        samples = torch.clamp(samples, 0., 1.)

        return samples

    def p_losses(self, x_start, cond, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start = x_start, t = t, noise = noise)
        pred_noise = self.apply_model(x_noisy, t, cond)
        pred_x0 = self.predict_start_from_noise(x_noisy, t, pred_noise)
        loss = F.mse_loss(pred_x0, x_start, reduction = "none")
        loss = loss.flatten(start_dim = 1).mean(dim = 1)

        return loss.mean()

    def forward(self, z, c, noise, num_samples=8):
        t = torch.randint(0, self.num_timesteps, (num_samples,), device = self.device).long()

        z = z.expand(num_samples, -1, -1, -1)

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start = c, t = tc, noise = torch.randn_like(c.float()))
        
        return self.p_losses(z, c, t, noise = noise)


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        sampler,
        dataloader,
        logger,
        device = "cuda:0",
        train_lr = 1e-2,
        train_num_steps = 10000,
        num_samples = 16,
        enable_visdom = False,
        results_folder = '',
        init_kl_weight = 1,
        sample_num_noise = 10
    ):
        super().__init__()

        self.model = diffusion_model
        self.sampler = sampler

        self.logger = logger
        self.device = device

        self.train_num_steps = train_num_steps
        self.num_samples = num_samples
        self.train_lr = train_lr
        self.init_kl_weight = init_kl_weight
        self.sample_num_noise = sample_num_noise

        self.dl = dataloader

        self.results_folder = results_folder

        if enable_visdom:
            self.vis = visdom.Visdom(port=8097, env=results_folder)
        else:
            self.vis = None

        self.sscd_model = torch.jit.load("models/sscd/sscd_disc_large.torchscript.pt")
        self.sscd_model.to(self.device)
        self.sscd_model.eval()

        self.sscd_transforms = T.Compose([
            T.Resize([256, 256]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def add_scalar(self, step, score, title):
        self.vis.line(
            X=np.array([step]),
            Y=[score],
            update="append",
            win=title,
            opts={
                "title": title
            }
        )

    @torch.inference_mode()
    def random_sample_images(self, mu, logvar, num_samples=8):
        n = torch.randn(num_samples, 3, 64, 64, device=self.device) * logvar.div(2).exp() + mu
        samples = self.model.sample(self.sampler, n)

        return samples

    @torch.inference_mode()
    def sscd_similarity(self, q, v):
        q = self.sscd_transforms(q)
        v = self.sscd_transforms(v)

        q = self.sscd_model(q)
        v = self.sscd_model(v)

        return q.mm(v.T).squeeze()

    def train_one_batch(
            self, save_dir, img, img_id,
            C=10, ksi=1e-3, delta=5e-4, beta=0.5
        ):
        with torch.no_grad():
            encoder_posterior = self.model.encode_first_stage(img * 2 - 1)
            z = self.model.get_first_stage_encoding(encoder_posterior)
            z.requires_grad_(False)

        # noise to optimize
        mu = torch.zeros_like(z, device=self.device)
        logvar = torch.zeros_like(z, device=self.device)
        mu.requires_grad_(True)
        logvar.requires_grad_(True)

        params = [mu, logvar]
        opt = Adam(params, self.train_lr)
        n_trainable_parameters = sum([p.data.nelement() for p in params])
        self.logger.info(f"Number of trainable parameters: {n_trainable_parameters}")

        step = 0

        p_losses = SmoothedValue()
        r_losses = SmoothedValue()

        kl_weight = self.init_kl_weight
        p_loss_prev = torch.tensor(float("inf"), device=self.device)
        success = False

        for step in tqdm(range(self.train_num_steps)):
            n = torch.randn(self.num_samples, *z.shape[1:], device=self.device) * logvar.div(2).exp() + mu
            p_loss = self.model(z, None, n, num_samples=self.num_samples)

            loss = p_loss

            kl_div = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1)
            loss += kl_weight * kl_div.mean()

            loss.backward()

            opt.step()
            opt.zero_grad()

            p_losses.update(p_loss.item())
            r_losses.update(kl_div.mean().item())

            p_loss_now = torch.tensor(list(p_losses.deque), dtype=torch.float32)[-100:].mean()
            if step > 0 and (step + 1) % C == 0:
                if p_loss_prev - p_loss_now < ksi:
                    kl_weight = max(kl_weight / 2, 0)
                else:
                    kl_weight += delta
                p_loss_prev = p_loss_now
                
                samples = self.random_sample_images(mu, logvar, num_samples=8)

                sim_scores = self.sscd_similarity(img, samples)
                
                if sim_scores.min() >= beta:
                    success = True
                    break
            else:
                kl_weight += delta

            if self.vis is not None:
                self.add_scalar(step, p_losses.avg, f"{img_id}_p_loss")
                self.add_scalar(step, r_losses.value, f"{img_id}_r_loss")
                self.add_scalar(step, kl_weight, f"{img_id}_kl_weight")

        if success:
            with torch.inference_mode():
                invmm = 0.5 * torch.sum(mu ** 2 + logvar.exp() - logvar - 1).cpu()
        else:
            invmm = torch.tensor(float("inf"))

        stacked_samples = torch.cat([img.expand_as(samples),samples], dim=0)
        save_image(stacked_samples, osp.join(save_dir, f"stacked.jpg"), nrow=samples.shape[0])
        os.makedirs(osp.join(save_dir, "samples"), exist_ok=True)
        for i, sample in enumerate(samples):
            save_image(sample, osp.join(save_dir, "samples", f"{i}.jpg"))
        
        torch.save({
            "logvar": logvar.detach().clone().cpu(),
            "mu": mu.detach().clone().cpu(),
            "sim_scores": sim_scores.cpu(),
            "invmm": invmm,
            "step": step
        }, osp.join(save_dir, f'params.pt'))

    def train(self):
        for batch_idx, (img, img_id) in enumerate(self.dl):
            img = img.to(self.device)

            work_dir = osp.join(self.results_folder, img_id[0])
            os.makedirs(work_dir, exist_ok = True)
            
            self.train_one_batch(work_dir, img, img_id[0])

        self.logger.info('training complete')


# dataset classes

class FacesHQ(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = sorted([p for ext in exts for p in glob.glob(osp.join(folder, f"**/*.{ext}"), recursive=True)])

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img_path = self.paths[i]
        img_id = osp.basename(img_path).split(".")[0]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, img_id


if __name__ == "__main__":
    main()