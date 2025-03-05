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
from safetensors import safe_open

from tqdm import tqdm
from PIL import Image
from contextlib import contextmanager

import sd3_impls
from sd3_impls import BaseModel, SDVAE, SD3LatentFormat, CFGDenoiser, sample_euler
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from utils import SmoothedValue


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-dp", "--data_path", type=str, required=True,
        help="root path of dataset"
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
        "--num_samples", type=int, default=16,
        help="number of samples",
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

    with torch.no_grad():
        model = StableDiffusion3(device)
        model.load(
            "models/sd3.5_medium.safetensors",
            None,
            3.0,
            "models",
            device
        )
    
    trainer = Trainer(
        model,
        dataloader,
        logger,
        device = device,
        train_lr = args.lr,
        train_num_steps = args.train_num_steps,
        results_folder = args.results_folder,
        num_samples = args.num_samples,
        enable_visdom = args.visdom,
        tau = args.tau,
        init_kl_weight = args.init_kl_weight,
        sample_num_noise = args.sample_num_noise
    )
    trainer.train()


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataloader,
        logger,
        device,
        train_lr = 1e-2,
        train_num_steps = 10000,
        results_folder = '',
        num_samples = 16,
        enable_visdom = False,
        tau = 2.0,
        init_kl_weight = 1,
        sample_num_noise = 8
    ):
        super().__init__()

        self.model = diffusion_model

        self.logger = logger
        self.device = device
        self.train_lr = train_lr
        self.num_samples = num_samples
        self.tau = tau
        self.init_kl_weight = init_kl_weight
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

    @torch.inference_mode()
    def sscd_similarity(self, q, v):
        q = self.sscd_transforms(q)
        v = self.sscd_transforms(v)

        q = self.sscd_model(q)
        v = self.sscd_model(v)

        return q.mm(v.T).squeeze()

    def train_one_batch(
            self, save_dir, img, prompt, img_id,
            C=50, ksi=1e-3, delta=1e-3, beta=0.5
        ):
        with torch.no_grad():
            z = self.model.vae_encode(img * 2 - 1)
            z = SD3LatentFormat().process_in(z)
            uc = self.model.get_cond("")
            tokens = self.model.tokenizer.tokenize_with_weights(prompt)
            g_out, g_pooled = self.model.clip_g.model.encode_token_weights(tokens["g"])
            t5_out, t5_pooled = self.model.t5xxl.model.encode_token_weights(tokens["t5xxl"])
            g_out, g_pooled, t5_out = g_out.to(self.device), g_pooled.to(self.device), t5_out.to(self.device)

        token_emb_layer = self.model.clip_l.transformer.text_model.embeddings.token_embedding
        token_emb_weight = token_emb_layer.weight.detach().clone()
        log_coeffs = torch.zeros(75, token_emb_weight.shape[0], device=self.device)
        log_coeffs.requires_grad_(True)

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
        kl_weight = self.init_kl_weight
        p_loss_prev = torch.tensor(float("inf"))
        success = False

        scaler = torch.amp.GradScaler()

        pbar = tqdm(range(self.train_num_steps), desc=img_id, dynamic_ncols=True)
        for step in pbar:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                n = torch.randn(self.num_samples, *z.shape[1:], device=self.device) * logvar.div(2).exp() + mu
                
                coeffs = F.gumbel_softmax(log_coeffs.expand(self.num_samples, -1, -1), hard=False, tau=tau)
                tok_embs = torch.bmm(coeffs, token_emb_weight.expand(self.num_samples, -1, -1))
                
                p_loss = self.model(z, n, tok_embs, tokens, g_out, g_pooled, t5_out)
                loss = p_loss

                r_loss = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1)
                # kl divergence
                loss += kl_weight * r_loss

            p_losses.update(p_loss.item())
            r_losses.update(r_loss.item())
            p_loss_now = torch.tensor(list(p_losses.deque), dtype=torch.float32)[-100:].mean()
            pbar.set_description(f"p_loss: {p_loss_now.item():.4f}")

            scaler.scale(loss).backward()
            # opt.step()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            if step > 0 and (step + 1) % C == 0:
                if p_loss_prev - p_loss_now < ksi:
                    kl_weight = max(kl_weight / 2, 0)
                else:
                    kl_weight += delta
                p_loss_prev = p_loss_now

                with torch.no_grad():
                    n = torch.randn(self.sample_num_noise, *z.shape[1:], device=self.device) * logvar.div(2).exp() + mu
                    coeffs = F.gumbel_softmax(log_coeffs.expand(self.sample_num_noise, -1, -1), hard=False, tau=tau)
                    tok_embs = torch.bmm(coeffs, token_emb_weight.expand(self.sample_num_noise, -1, -1))

                    samples, cfg_scales = self.model.random_sample_images(
                        n, tok_embs, tokens, g_out, g_pooled, t5_out, uc
                    )

                    sim_scores = [self.sscd_similarity(img, sample) for sample in samples]
                    sim_scores = torch.stack(sim_scores, dim=0)

                    if torch.any(sim_scores.min(dim=1)[0] >= beta):
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
        for batch_idx, (img, prompt, id) in enumerate(self.dl):
            img = img.to(self.device)
            
            work_dir = osp.join(self.results_folder, id[0])
            os.makedirs(work_dir, exist_ok = True)
            
            self.train_one_batch(work_dir, img, prompt[0], id[0])
        
        if self.vis is not None:
            self.vis.save([self.results_folder])
        self.logger.info('training complete')


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/text_encoders/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


# CLIPL_CONFIG = {
#     "hidden_act": "quick_gelu",
#     "hidden_size": 768,
#     "intermediate_size": 3072,
#     "num_attention_heads": 12,
#     "num_hidden_layers": 12,
# }


# class ClipL:
#     def __init__(self, model_folder: str, device="cpu"):
#         with safe_open(
#             f"{model_folder}/text_encoders/clip_l.safetensors", framework="pt", device="cpu"
#         ) as f:
#             self.model = SDClipModel(
#                 layer="hidden",
#                 layer_idx=-2,
#                 device=device,
#                 dtype=torch.float32,
#                 layer_norm_hidden_state=False,
#                 return_projected_pooled=False,
#                 textmodel_json_config=CLIPL_CONFIG,
#             )
#             load_into(f, self.model.transformer, "", device, torch.float32)

from transformers import CLIPTextModel, CLIPTokenizer
class ClipL(nn.Module):
    def __init__(self, model_folder, device="cpu"):
        super().__init__()
        self.transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
    def forward(self, tokens):
        tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()
        outputs = self.transformer(input_ids=tokens)

        return outputs.last_hidden_state, outputs.pooler_output


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/text_encoders/t5xxl_fp16.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)


class SD3:
    def __init__(
        self, model, shift, verbose=False, device="cpu"
    ):

        # NOTE 8B ControlNets were trained with a slightly different forward pass and conditioning,
        # so this is a flag to enable that logic.
        self.using_8b_controlnet = False

        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device=device,
                dtype=torch.float32,
                control_model_ckpt=None,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", device, torch.float32)


class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float32):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)


class StableDiffusion3(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def load(
        self,
        model,
        vae,
        shift,
        model_folder,
        device,
        load_tokenizers=True,
    ):
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        if load_tokenizers:
            print("Loading Google T5-v1-XXL...")
            self.t5xxl = T5XXL(model_folder, "cpu", torch.float32)
            print("Loading OpenAI CLIP L...")
            self.clip_l = ClipL(model_folder)
            print("Loading OpenCLIP bigG...")
            self.clip_g = ClipG(model_folder)
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift)
        print("Loading VAE model...")
        self.vae = VAE(vae or model)

        self.t5xxl.model.eval()
        self.t5xxl.model.requires_grad_(False)
        self.clip_l.to(device).eval()
        self.clip_l.requires_grad_(False)
        self.clip_g.model.eval()
        self.clip_g.model.requires_grad_(False)
        self.sd3.model.to(device).eval()
        self.sd3.model.requires_grad_(False)

        print("Models loaded.")

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        # l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        tokens_l = list(map(lambda x:x[0], tokens["l"][0]))
        l_out, l_pooled = self.clip_l([tokens_l])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        g_out, g_pooled, t5_out = g_out.to(self.device), g_pooled.to(self.device), t5_out.to(self.device)
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m"
    ) -> torch.Tensor:
        print("Sampling...")
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale
        }
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        latent = sample_fn(
            CFGDenoiser(self.sd3.model, steps),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
        )
        latent = SD3LatentFormat().process_out(latent)
        self.sd3.model = self.sd3.model.cpu()
        print("Sampling done")
        return latent

    @torch.no_grad()
    def vae_encode(
        self, batch_images
    ) -> torch.Tensor:
        print("Encoding image to latent...")
        image_torch = batch_images.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch)
        self.vae.model = self.vae.model.cpu()
        print("Encoded")
        return latent

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        return image

    def _image_to_latent(
        self,
        image,
        width,
        height,
        using_2b_controlnet: bool = False,
        controlnet_type: int = 0,
    ) -> torch.Tensor:
        image_data = Image.open(image)
        image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image_data, using_2b_controlnet, controlnet_type)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def forward(self, z, n, token_embs, tokens, g_out, g_pooled, t5_out):
        batch_size = n.shape[0]

        sampling = self.sd3.model.model_sampling
        random_timestep = torch.randint(0, sampling.sigmas.shape[0], (batch_size,), device=self.device)
        sigmas = sampling.sigmas[random_timestep]

        z = z.expand_as(n)
        noisy_z = sampling.noise_scaling(sigmas.reshape(sigmas.shape[0], *((1,)*len(n.shape[1:]))), n, z)

        g_out = g_out.expand(batch_size, -1, -1)
        g_pooled = g_pooled.expand(batch_size, -1)
        t5_out = t5_out.expand(batch_size, -1, -1)
        tokens = list(map(lambda x:x[0], tokens["l"][0]))
        with self.modify_clip_l_inputs(token_embs):
            l_out, l_pooled = self.clip_l([tokens]*batch_size)
        lg_out = torch.cat([l_out, g_out], dim=-1)
        if lg_out.shape[-1] < 4096:
            lg_out = torch.cat([lg_out, torch.zeros(lg_out.shape[0], lg_out.shape[1], 4096-lg_out.shape[-1], device=lg_out.device)], dim=-1)
        c_crossattn, y = torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

        denoised = self.sd3.model(noisy_z, sigmas, c_crossattn=c_crossattn, y=y)
        loss = F.mse_loss(denoised, z, reduction="none").mean()

        return loss
        
    @contextmanager
    def modify_clip_l_inputs(self, token_embs):
        def hook(module, args, output):
            output[:, 1:-1] = token_embs
            
            return output
        
        # embedder = self.clip_l.model.transformer.text_model.embeddings.token_embedding
        embedder = self.clip_l.transformer.text_model.embeddings.token_embedding
        h = embedder.register_forward_hook(hook)
        
        try:
            yield None
        finally:
            h.remove()

    @torch.inference_mode()
    def random_sample_images(self, n, tok_embs, tokens, g_out, g_pooled, t5_out, uc):
        num_samples = n.shape[0]
        g_out = g_out.expand(num_samples, -1, -1)
        g_pooled = g_pooled.expand(num_samples, -1)
        t5_out = t5_out.expand(num_samples, -1, -1)
        tokens = list(map(lambda x:x[0], tokens["l"][0]))
        with self.modify_clip_l_inputs(tok_embs):
            l_out, l_pooled = self.clip_l([tokens]*num_samples)
        lg_out = torch.cat([l_out, g_out], dim=-1)
        if lg_out.shape[-1] < 4096:
            lg_out = torch.cat([lg_out, torch.zeros(lg_out.shape[0], lg_out.shape[1], 4096-lg_out.shape[-1], device=lg_out.device)], dim=-1)
        c_crossattn, y = torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

        uc = (uc[0].expand(num_samples, -1, -1), uc[1].expand(num_samples, -1))

        sigmas = self.get_sigmas(self.sd3.model.model_sampling, 50).cuda()

        all_samples = []
        cfg_scales = [1.0, 2.0, 3.0, 4.0, 5.0]
        for cfg_scale in cfg_scales:
            extra_args = {
                "cond": {"c_crossattn": c_crossattn, "y": y},
                "uncond": {"c_crossattn": uc[0], "y": uc[1]},
                "cond_scale": cfg_scale
            }
            latent = sample_euler(
                CFGDenoiser(self.sd3.model, 50),
                n,
                sigmas,
                extra_args=extra_args,
            )
            latent = SD3LatentFormat().process_out(latent)
            samples = self.vae_decode(latent)
            all_samples.append(samples)

        return all_samples, cfg_scales


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

        with open(osp.join(basedir, id+".txt"), "r") as f:
            prompt = f.readline()
        
        return self.transform(img), prompt, id


if __name__ == "__main__":
    main()