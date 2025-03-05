import os
import glob
import warnings

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from diffusion import GaussianDiffusionInversionTrainer, GaussianDiffusionDDIMSampler
from model import UNet
from utils import SmoothedValue


FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 1e-1, help='target learning rate')
flags.DEFINE_integer('total_steps', 2000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('batch_size', 32, help='batch size')
flags.DEFINE_integer('num_workers', 8, help='workers of Dataloader')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS_INVERSION', help='log directory')
flags.DEFINE_string('pretrained_ckpt_dir', './logs/DDPM_CIFAR10_EPS', help='pretrained checkpoint directory')
flags.DEFINE_string('sscd_ckpt', '/path/to/sscd_disc_large.torchscript.pt', help='pretrained sscd checkpoint')
flags.DEFINE_integer('sample_size', 8, "sampling size of images")
flags.DEFINE_integer('ST', 200, help='ddim sampling steps')

device = torch.device('cuda:0')


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        exts = ['jpg', 'jpeg', 'png', 'tiff']
    ):
        super().__init__()
        self.folder = folder
        self.paths = sorted([p for ext in exts for p in glob.glob(f'{folder}/*.{ext}')])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img_id = os.path.basename(path).split('.')[0]
        img = Image.open(path).convert("RGB")
        return self.transform(img), img_id


@torch.inference_mode()
def sscd_similarity(sscd_model, sscd_transform, x1, x2):
    x1 = sscd_transform(x1)
    x2 = sscd_transform(x2)
    x1_feat = sscd_model(x1)
    x2_feat = sscd_model(x2)

    return x1_feat.mm(x2_feat.T)


def inversion(
    model, sampler, sscd_model, sscd_transform,
    x_0, total_steps=2000, batch_size=64,
    img_size=32, lr=1e-1, C=50, ksi=1e-3,
    delta=1e-4, beta=0.5, writer=None
):
    mu = torch.zeros_like(x_0, device=device)
    logvar = torch.zeros_like(x_0, device=device)
    mu.requires_grad_(True)
    logvar.requires_grad_(True)

    p_losses = SmoothedValue(window_size=100)
    r_losses = SmoothedValue()

    optim = torch.optim.Adam([mu, logvar], lr=lr)

    kl_weight = 1
    p_loss_prev = torch.tensor(float("inf"), device=device)
    success = False

    # train
    for step in tqdm(range(total_steps), dynamic_ncols=True, leave=False):
        optim.zero_grad()
        noise = torch.randn(batch_size, 3, img_size, img_size, device=device) * logvar.div(2).exp() + mu
        p_loss = model(x_0.expand_as(noise), noise).mean()
        r_loss = 0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1)
        loss = p_loss + kl_weight * r_loss
        loss.backward()
        optim.step()

        p_losses.update(p_loss.item())
        r_losses.update(r_loss.item())

        if writer is not None:
            writer.add_scalar('p_loss', p_losses.avg, step)
            writer.add_scalar('r_loss', r_loss, step)
            writer.add_scalar('kl_weight', kl_weight, step)
        
        p_loss_now = p_losses.avg
        if step > 0 and (step + 1) % C == 0:
            if p_loss_prev - p_loss_now < ksi:
                kl_weight = max(kl_weight / 2, 0)
            else:
                kl_weight += delta
            p_loss_prev = p_loss_now

            with torch.inference_mode():
                x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size, device=device) * logvar.div(2).exp() + mu
                samples = sampler(x_T)
            sim = sscd_similarity(sscd_model, sscd_transform, x_0*0.5+0.5, samples*0.5+0.5)

            if sim.min() >= beta:
                success = True
                break
        else:
            kl_weight += delta

    if success:
        with torch.inference_mode():
            invmm =  0.5 * torch.mean(mu ** 2 + logvar.exp() - logvar - 1).cpu()
    else:
        invmm = torch.tensor(float("inf"))

    return mu.detach().clone().cpu(), logvar.detach().clone().cpu(), invmm, samples


def run():
    # dataset
    dataset = Dataset(folder='/path/to/cifar10')
    print("number of training samples:", len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=FLAGS.num_workers, drop_last=False
    )

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout
    )
    
    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.pretrained_ckpt_dir, 'ckpt.pt'))
    net_model.load_state_dict(ckpt['ema_model'])
    net_model.requires_grad_(False)
    net_model.eval()

    trainer = GaussianDiffusionInversionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    sampler = GaussianDiffusionDDIMSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.ST, img_size=FLAGS.img_size, eta=0,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    
    sscd_model = torch.jit.load(FLAGS.sscd_ckpt)
    sscd_model.to(device)
    sscd_model.eval()

    sscd_transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir), exist_ok=True)
    writer = SummaryWriter(FLAGS.logdir)
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # start training
    for idx, (x_0, img_id) in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        x_0 = x_0.to(device)
        img_id = img_id[0]

        save_dir = os.path.join(FLAGS.logdir, img_id)
        
        os.makedirs(save_dir, exist_ok=True)

        mu, logvar, invmm, samples = inversion(
            trainer, sampler, sscd_model, sscd_transform,
            x_0, writer=None,
            total_steps=FLAGS.total_steps,
            batch_size=FLAGS.batch_size,
            img_size=FLAGS.img_size,
            lr=FLAGS.lr
        )

        samples = torch.cat([x_0.expand_as(samples), samples], dim=0)
        grid = (make_grid(samples, nrow=FLAGS.sample_size) + 1) / 2
        path = os.path.join(save_dir, 'sample.png')
        save_image(grid, path)
        writer.add_image('sample', grid)

        ckpt = {
            'mu': mu,
            'logvar': logvar,
            'invmm': invmm
        }
        torch.save(ckpt, os.path.join(save_dir, 'params.pt'))

    writer.close()


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    run()


if __name__ == '__main__':
    app.run(main)
