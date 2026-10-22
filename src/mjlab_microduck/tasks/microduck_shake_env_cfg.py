"""Microduck SHAKE task — secousse du corps (s'ebrouer) sur place, debout -> debout.

Geste episodique declenche comme hop / ground_pick : la commande est la phase d'un cycle
([cos(2*pi*phi), sin(2*pi*phi), 0], phi demarre a 0 debout). Un cycle dure SHAKE_PERIOD secondes :

    phi 0.00-0.15  debout, immobile
    phi 0.15-0.65  secousse : le tronc oscille en lacet (SHAKE_CYCLES allers-retours), pieds au sol, sans deriver
    phi 0.65-1.00  debout, immobile

La fenetre de secousse reprend exactement celle d'accroupissement + bond du Hop (0.15-0.65) : les recompenses de repos
du Hop (pose, hauteur, immobilite) s'appliquent donc telles quelles hors de la fenetre.

Le lacet suit une vitesse angulaire CIBLE sinusoidale (somme nulle sur un cycle) plutot que « la plus grande possible » :
sinon la politique tourne en continu. Le meme module sert de base a d'autres gestes de corps (frisson, tape du pied...).

PREMIERE VERSION : a verifier par un pilote court avant un run complet. Sentinelles : shake_track (doit monter),
shake_feet_penalty (<= 0, pieds au sol), shake_drift (derive horizontale).
"""

import math

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    CROUCH_START,
    LAND_END,
    _in,
    _phase,
    hop_low_penalty,
    hop_rest_height,
    hop_rest_pose,
    hop_rest_still,
    hop_tilt_penalty,
    make_microduck_hop_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
from mjlab_microduck.tasks.mdp import ShakePhaseCommandCfg

SHAKE_PERIOD = 2.0        # s
EPISODE_LENGTH_S = 6.0    # 3 cycles
SHAKE_START, SHAKE_END = CROUCH_START, LAND_END   # 0.15 - 0.65 (1 s)
SHAKE_CYCLES = 3          # allers-retours dans la fenetre (3 Hz)
SHAKE_WZ = 3.0            # rad/s, amplitude de la vitesse de lacet cible (~ +-9 deg)

_HOP_ONLY_REWARDS = (
    "hop_crouch", "hop_air", "hop_height", "hop_launch", "hop_air_penalty", "hop_land_upright",
    "hop_mid_contact_penalty",
)


def _feet_down(env) -> torch.Tensor:
    """1 si les DEUX pieds touchent le sol."""
    found = env.scene.sensors["feet_ground_contact"].data.found[:, :2]
    per_foot = found.reshape(found.shape[0], 2, -1).sum(dim=-1) > 0
    return per_foot.all(dim=-1).float()


def shake_track(env, command_name: str, std: float = 1.0,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vitesse de lacet du tronc proche de la cible sinusoidale pendant la fenetre de secousse."""
    p = _phase(env, command_name)
    u = ((p - SHAKE_START) / (SHAKE_END - SHAKE_START)).clamp(0.0, 1.0)
    target = SHAKE_WZ * torch.sin(2 * math.pi * SHAKE_CYCLES * u)
    wz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2], nan=0.0)
    return _in(p, SHAKE_START, SHAKE_END) * torch.exp(-((wz - target) / std) ** 2)


def shake_feet_penalty(env, command_name: str) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si un pied quitte le sol pendant la secousse : on s'ebroue sur place, on ne saute pas."""
    p = _phase(env, command_name)
    return -_in(p, SHAKE_START, SHAKE_END) * (1.0 - _feet_down(env))


def shake_drift(env, command_name: str, vel_std: float = 0.1,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pas de deplacement horizontal pendant la secousse (recompense)."""
    p = _phase(env, command_name)
    v = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2], nan=0.0).norm(dim=-1)
    return _in(p, SHAKE_START, SHAKE_END) * torch.exp(-(v / vel_std) ** 2)


def shake_yaw_lock(env, command_name: str, std: float = 0.5,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Cap (yaw) proche de celui du debut de l'episode (phase 0, debout) pendant le repos
    qui suit chaque secousse : corrige la derive mesuree en duck-sim (le canard tournait sur
    place au fil des cycles au lieu de revenir dans son axe). Actif hors fenetre de secousse
    (phase SHAKE_END..1.0) pour laisser la secousse elle-meme libre d'osciller."""
    p = _phase(env, command_name)
    term = env.command_manager.get_term(command_name)
    yaw = env.scene[asset_cfg.name].data.heading_w
    err = wrap_to_pi(yaw - term.yaw0)
    # v3 : actif sur TOUT le cycle (la v2, hors fenetre seulement, restait sans effet)
    return torch.exp(-(err / std) ** 2)


def make_microduck_shake_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_hop_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S
    for name in _HOP_ONLY_REWARDS:
        cfg.rewards.pop(name, None)
    base_fields = {k: v for k, v in vars(cfg.commands["twist"]).items() if k != "class_type"}
    cfg.commands["twist"] = ShakePhaseCommandCfg(**{**base_fields, "period": SHAKE_PERIOD})

    c = {"command_name": "twist"}
    cfg.rewards["shake_track"] = RewardTermCfg(func=shake_track, weight=4.0, params=dict(c))
    cfg.rewards["shake_feet_penalty"] = RewardTermCfg(func=shake_feet_penalty, weight=3.0, params=dict(c))
    cfg.rewards["shake_drift"] = RewardTermCfg(func=shake_drift, weight=1.5, params=dict(c))
    cfg.rewards["shake_yaw_lock"] = RewardTermCfg(func=shake_yaw_lock, weight=8.0, params=dict(c))
    return cfg


MicroduckShakeRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_shake",
    run_name="microduck_shake",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
