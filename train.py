import os
import yaml
import torch
import wandb
import datetime
import argparse
import numpy as np

from tqdm import tqdm
from solvers.PICE import PICE
from plotting.Loggers import CostsLogger
from Dynamics.AlanineDynamics import AlanineDynamics
from Dynamics.PolytrimerDynamics import PolyDynamics
from Dynamics.ChignolinDynamics import ChignolinDynamics
import pytz

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/', type=str, help="Path to config file")
args = parser.parse_args()


def load_config(config_file):
  cfg = {}
  with open(config_file, 'r') as f:
    yaml_loaded = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in yaml_loaded.items():
        cfg[key] = value
  return cfg

cfg = load_config(args.config)
molecule = cfg['molecule']
kst = pytz.timezone('Asia/Seoul')
current_time = datetime.datetime.now(tz=kst).strftime("%m%d-%H%M%S")

log_dir = f'{cfg["result_dir"]}/{molecule}/{current_time}'
if not os.path.exists(log_dir):
  os.makedirs(log_dir)
logger = CostsLogger(log_dir)
if cfg['wandb']:
  wandb.init(
    project='pice',
    entity='eddy26',
    name=f'{current_time}-{molecule}',
    config=cfg
  )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_samples = cfg['n_samples']
if molecule == 'alanine':
  environment = AlanineDynamics(loss_func='pairwise_dist', n_samples=n_samples, device=device, save_file=log_dir)
  from policies.Alanine import NNPolicy
elif molecule == 'poly':
  environment = PolyDynamics(loss_func='pairwise_dist', n_samples=n_samples, device=device, save_file=log_dir)
  from policies.Poly import NNPolicy
elif molecule == 'chignolin':
  environment = ChignolinDynamics(loss_func='pairwise_dist', n_samples=n_samples, device=device, save_file=log_dir)
  from policies.Chignolin import NNPolicy
else:
  raise ValueError(f"Unknown molecule: {molecule}")
dims = environment.dims

force = cfg['force']
T = cfg['T']
std = torch.tensor(cfg['std']).to(device)
R = torch.eye(dims).to(device)
nn_policy = NNPolicy(device, dims=dims, force=force, T=T)

dt = torch.tensor(cfg['dt']).to(device)
n_steps = int(T / dt)
n_rollouts = cfg['n_rollouts']
n_samples = cfg['n_samples']
lr = cfg['lr']
PICE(
  environment, nn_policy, n_rollouts, n_samples, n_steps,
  dt, std, dims * 2, R, logger, force, [], True, log_dir, 
  device=device, lr=lr, wandb=cfg['wandb']
)

torch.save(nn_policy, f'{log_dir}/final_policy')