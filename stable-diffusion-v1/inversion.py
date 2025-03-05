import os
import os.path as osp
import math
import glob
import argparse
import logging
import yaml
import visdom
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision import transforms as T

from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from contextlib import contextmanager

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils import SmoothedValue


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-dp", "--data_path", type=str, required=True,
        help="root path of dataset"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="configs/stable-diffusion/measure.yaml",
        help="config file"
    )
    parser.add_argument(
        "--ckpt_file", default="models/ldm/stable-diffusion-v1/sd-v1-4-original.ckpt", type=str,
        help="pretrained sd checkpoint file"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-1,
        help="learning rate"
    )
    parser.add_argument(
        "--init_kl_weight", type=float, default=1,
        help="initial weight of kl divergence"
    )
    parser.add_argument(
        "--train_num_steps", type=int, default=2000,
        help="number of train steps"
    )
    parser.add_argument(
        "--results_folder", default="checkpoints/test", type=str,
        help="path to save training and sampling results"
    )
    parser.add_argument(
        "--num_tokens", type=int, default=75,
        help="number of tokens to optimize",
    )
    parser.add_argument(
        "--num_samples", type=int, default=16,
        help="number of samples",
    )
    parser.add_argument(
        "--sample_num_prompt", type=int, default=1,
        help="number of prompt to sample",
    )
    parser.add_argument(
        "--sample_num_noise", type=int, default=8,
        help="number of noise to sample",
    )
    parser.add_argument(
        "--visdom", action="store_true",
        help="log with visdom",
    )
    parser.add_argument(
        "--tau", type=float, default=2.0,
        help="temperature",
    )

    return parser

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def divisible_by(numer, denom):
    return (numer % denom) == 0

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

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

def main():
    parser = get_parser()
    args = parser.parse_args()

    setup_logger(args.results_folder)
    logger = logging.getLogger("main")

    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    os.makedirs(args.results_folder, exist_ok=True)
    with open(os.path.join(args.results_folder, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    torch.set_num_threads(8)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    logger.info(f"Using device {device}")

    dataset = LAION(args.data_path, 512)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size = 1,
        shuffle=False, pin_memory = True
    )

    config = OmegaConf.load(args.config)

    model = load_model_from_config(config, args.ckpt_file, verbose=True)
    model.to(device)

    sampler = DDIMSampler(model)
    
    trainer = Trainer(
        model,
        sampler,
        dataloader,
        logger,
        device = device,
        train_lr = args.lr,
        train_num_steps = args.train_num_steps,
        results_folder = args.results_folder,
        num_tokens = args.num_tokens,
        num_samples = args.num_samples,
        enable_visdom = args.visdom,
        tau = args.tau,
        init_kl_weight = args.init_kl_weight,
        sample_num_prompt = args.sample_num_prompt,
        sample_num_noise = args.sample_num_noise
    )
    trainer.train()


class DistForward(nn.Module):
    def __init__(self, x, name):
        super().__init__()
        self.register_parameter(name, nn.parameter.Parameter(x))

    def forward(self):
        return self.x


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        sampler,
        dataloader,
        logger,
        device,
        train_lr = 1e-2,
        train_num_steps = 10000,
        results_folder = '',
        num_tokens = 5,
        num_samples = 16,
        enable_visdom = False,
        tau = 1.0,
        init_kl_weight = 1,
        sample_num_prompt = 8,
        sample_num_noise = 8
    ):
        super().__init__()

        self.model = diffusion_model
        self.sampler = sampler

        self.logger = logger
        self.device = device
        self.train_lr = train_lr
        self.num_tokens = num_tokens
        self.num_samples = num_samples
        self.tau = tau
        self.init_kl_weight = init_kl_weight
        self.sample_num_prompt = sample_num_prompt
        self.sample_num_noise = sample_num_noise

        self.train_num_steps = train_num_steps

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
            T.Resize([320, 320]),
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

    def add_bar(self, x, y, title):
        self.vis.bar(
            X=y,
            Y=x,
            win=title,
            opts={
                "title": title
            }
        )

    def initialize_one_batch(self, img):

        with torch.no_grad():
            encoder_posterior = self.model.encode_first_stage(img * 2 - 1)
            z = self.model.get_first_stage_encoding(encoder_posterior)
            z.requires_grad_(False)

        pseudo_prompt = [""]

        log_coeffs = torch.zeros(self.num_tokens, self.model.voc_emb.shape[0], device=self.device)

        log_coeffs.requires_grad_(True)

        return z, log_coeffs, pseudo_prompt
    
    @torch.inference_mode()
    def sample_tokens(self, log_coeffs, tau=2., hard=False):
        tokenizer = self.model.cond_stage_model.tokenizer
        coeffs = F.gumbel_softmax(log_coeffs, hard=hard, tau=tau)
        token_ids = coeffs.argmax(dim=1)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        prompt = tokenizer.convert_tokens_to_string(tokens)
        return coeffs, prompt

    @torch.inference_mode()
    def sscd_similarity(self, q, v):
        q = self.sscd_transforms(q)
        v = self.sscd_transforms(v)

        q = self.sscd_model(q)
        v = self.sscd_model(v)

        return q.mm(v.T).squeeze()

    @torch.inference_mode()
    def random_sample_images(self, pseudo_prompt, log_coeffs, mu, logvar, uc, num_samples=8):
        tok_emb = []
        coeffs, prompt = self.sample_tokens(log_coeffs, self.tau, False)
        tok_emb = coeffs.mm(self.model.voc_emb).unsqueeze(0)
        with self.model.modify_token_embedding(tok_emb, self.num_tokens):
            c = self.model.get_learned_conditioning(pseudo_prompt)
        c = c.expand(num_samples, -1, -1)
        
        n = torch.randn(num_samples, 4, 64, 64, device=self.device) * logvar.div(2).exp() + mu
        
        cfg_scales = [1, 2, 3, 4, 5, 6, 7]
        samples = []
        for scale in cfg_scales:
            samples.append(self.model.sample(self.sampler, n, uc.expand_as(c), c, scale=scale))

        return samples, cfg_scales

    def train_one_batch(
            self, batch_idx, save_dir, z, log_coeffs, img, pseudo_prompt, img_id,
            C=50, ksi=1e-3, delta=1e-3, beta=0.5
        ):

        mu = torch.zeros_like(z, device=self.device)
        logvar = torch.zeros_like(z, device=self.device)
        mu.requires_grad_(True)
        logvar.requires_grad_(True)

        params = [
            {"params": [log_coeffs], "weight_decay": 0},
            {"params": [mu], "weight_decay": 0},
            {"params": [logvar], "weight_decay": 0}
        ]
        opt = Adam(params, self.train_lr)

        p_losses = SmoothedValue()
        r_losses = SmoothedValue()

        tau = self.tau
        init_kl_weight = self.init_kl_weight
        p_loss_prev = torch.tensor(float("inf"))
        success = False

        with torch.no_grad():
            uc = self.model.get_learned_conditioning([""])

        for step in tqdm(range(self.train_num_steps), desc=img_id, dynamic_ncols=True):
            n = torch.randn(self.num_samples, *z.shape[1:], device=self.device) * logvar.div(2).exp() + mu
            
            coeffs = F.gumbel_softmax(log_coeffs.expand(self.num_samples, -1, -1), hard=False, tau=tau)
            tok_emb = torch.bmm(coeffs, self.model.voc_emb.expand(self.num_samples, -1, -1))
            
            p_loss = self.model(z, n, pseudo_prompt, tok_emb, num_tokens=self.num_tokens, num_samples=self.num_samples)

            loss = p_loss

            r_loss = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1)
            # kl divergence
            loss += init_kl_weight * r_loss
            
            loss.backward()
            opt.step()
            opt.zero_grad()

            p_losses.update(p_loss.item())
            r_losses.update(r_loss.item())
            
            p_loss_now = torch.tensor(list(p_losses.deque), dtype=torch.float32)[-100:].mean()
            if step > 0 and (step + 1) % C == 0:
                if p_loss_prev - p_loss_now < ksi:
                    init_kl_weight = max(init_kl_weight / 2, 0)
                else:
                    init_kl_weight += delta
                p_loss_prev = p_loss_now

                for _ in range(self.sample_num_prompt):
                    samples, cfg_scales = self.random_sample_images(
                        pseudo_prompt, log_coeffs, mu, logvar, uc, num_samples=self.sample_num_noise
                    )

                    sim_scores = [self.sscd_similarity(img, sample) for sample in samples]
                    sim_scores = torch.stack(sim_scores, dim=0)

                    if torch.any(sim_scores.min(dim=1)[0] >= beta):
                        success = True
                        break
            else:
                init_kl_weight += delta
            
            if self.vis is not None:
                self.add_scalar(step, p_losses.avg, f"{img_id}_p_loss")
                self.add_scalar(step, r_losses.value, f"{img_id}_r_loss")
                self.add_scalar(step, init_kl_weight, f"{img_id}_kl_weight")

            if success:
                break

        if success:
            with torch.inference_mode():
                invmm = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1).cpu()
        else:
            invmm = torch.tensor(float("inf"))

        stacked_samples = torch.cat([img.expand_as(samples[0])]+samples, dim=0)
        save_image(stacked_samples, osp.join(save_dir, f"stacked.jpg"), nrow=samples[0].shape[0])
        
        os.makedirs(osp.join(save_dir, "samples"), exist_ok=True)
        for sample_scale, cfg_scale in zip(samples, cfg_scales):
            os.makedirs(osp.join(save_dir, "samples", f"{cfg_scale:.1f}"), exist_ok=True)
            for i, sample in enumerate(sample_scale):
                save_image(sample, osp.join(save_dir, "samples", f"{cfg_scale:.1f}", f"{i}.jpg"))

        torch.save({
            "log_coeffs": log_coeffs.detach().clone().cpu(),
            "logvar": logvar.detach().clone().cpu(),
            "mu": mu.detach().clone().cpu(),
            "invmm": invmm,
            "sim_scores": sim_scores.cpu(),
            "step": step
        }, osp.join(save_dir, f'params.pt'))

    def train(self):
        for batch_idx, (img, id) in enumerate(self.dl):
            img = img.to(self.device)
            
            work_dir = osp.join(self.results_folder, id[0])
            os.makedirs(work_dir, exist_ok = True)
            
            z, log_coeffs, pseudo_prompt = self.initialize_one_batch(img)
            self.train_one_batch(batch_idx, work_dir, z, log_coeffs, img, pseudo_prompt, id[0])
        
        if self.vis is not None:
            self.vis.save([self.results_folder])
        self.logger.info('training complete')


class LatentDiffusionInversion(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.first_stage_model.requires_grad_(False)
        self.first_stage_model.eval()
        self.cond_stage_model.requires_grad_(False)
        self.cond_stage_model.eval()
        # self.model.requires_grad_(False)
        self.model.eval()

        voc_emb = self.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.detach().clone()
        self.register_buffer("voc_emb", voc_emb)

    @property
    def device(self):
        return self.betas.device

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.mode()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.inference_mode()
    def sample(self, sampler, n, uc, c, scale=7.5):
        shape = (4, 64, 64)
        batch_size = n.shape[0]

        samples, _ = sampler.sample(
            S=50, 
            batch_size=batch_size,
            conditioning=c,
            shape=shape, 
            eta=0., 
            verbose=False, 
            x_T=n,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc
        )
        samples = self.decode_first_stage(samples)
        
        samples = (samples + 1.) / 2.
        samples = torch.clamp(samples, 0., 1.)
        
        return samples

    def p_losses(self, x_start, cond, t, noise):
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        pred_noise = self.apply_model(x_noisy, t, cond)
        
        pred_x0 = self.predict_start_from_noise(x_noisy, t, pred_noise)
        loss = F.mse_loss(pred_x0, x_start, reduction="none").mean(dim=[1, 2, 3])
        # loss = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=[1, 2, 3])
        # loss = (self.weights[t] * loss).mean()
        loss = loss.mean()

        return loss
    
    def forward(self, z, n, txts, tok_emb, num_tokens=3, num_samples=8):
        t = torch.randint(0, self.num_timesteps, (num_samples,), device=self.device)

        z = z.expand(num_samples, -1, -1, -1)
        
        with self.modify_token_embedding(tok_emb, num_tokens):
            c = self.get_learned_conditioning(txts * num_samples)
        
        p_loss = self.p_losses(z, c, t, n)
        
        return p_loss

    @contextmanager
    def modify_token_embedding(self, tok_emb, num_tokens):
        def hook(modules, args, kwargs):
            with torch.no_grad():
                input_ids = kwargs["input_ids"]
                inputs_embeds = modules.token_embedding(input_ids)
            inputs_embeds[:, 1:1+num_tokens] = tok_emb
            
            kwargs["inputs_embeds"] = inputs_embeds
            return args, kwargs
        
        embedder = self.cond_stage_model.transformer.text_model.embeddings
        h = embedder.register_forward_pre_hook(hook, with_kwargs=True)
        
        try:
            yield None
        finally:
            h.remove()


class LAION(torch.utils.data.Dataset):
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
        basedir, img_file = osp.split(img_path)
        id = img_file.split(".")[0]
        
        img = Image.open(img_path).convert("RGB")
        
        return self.transform(img), id


if __name__ == "__main__":
    main()