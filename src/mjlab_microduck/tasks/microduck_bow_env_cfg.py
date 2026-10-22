"""Microduck BOW task — salut marque : le tronc s'incline vers l'avant puis se redresse, pieds au sol.

Geste episodique declenche comme hop / shake : la commande est la phase d'un cycle
([cos(2*pi*phi), sin(2*pi*phi), 0], phi demarre a 0 debout). Un cycle dure 2,4 s (HOP_PERIOD) :

    phi 0.00-0.15  debout, immobile
    phi 0.15-0.65  salut : inclinaison vers l'avant (descente, maintien, remontee), pieds au sol, sans deriver
    phi 0.65-1.00  debout, immobile

Meme fenetre que Hop/Shake (0.15-0.65) : les recompenses de repos du Hop (pose, hauteur, immobilite) s'appliquent
telles quelles hors de la fenetre. L'inclinaison est mesuree par la composante x de la gravite projetee dans le
repere du tronc : sin(tangage) > 0 quand le nez descend. hop_tilt_penalty (qui punit TOUTE inclinaison) est retiree,
remplacee par une penalite de roulis seul.

PREMIERE VERSION : a verifier par un pilote de 400 iterations avant un run complet. Sentinelles : bow_pitch (doit
monter), bow_feet_penalty (<= 0), bow_roll_penalty (<= 0), bow_drift.
"""

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    CROUCH_START,
    LAND_END,
    _in,
    _phase,
    make_microduck_hop_env_cfg,
)
from mjlab_microduck.tasks.microduck_shake_env_cfg import _feet_down
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg

EPISODE_LENGTH_S = 7.2    # 3 cycles
BOW_START, BOW_END = CROUCH_START, LAND_END   # 0.15 - 0.65
BOW_PITCH = 0.40          # rad, inclinaison visee au maintien (~23 deg, nez vers le bas)
BOW_RAMP = 0.3            # part de la fenetre pour descendre (et autant pour remonter)


def _bow_target_sin(p: torch.Tensor) -> torch.Tensor:
    """sin du tangage vise : rampe de montee, plateau, rampe de descente (0 hors fenetre)."""
    u = ((p - BOW_START) / (BOW_END - BOW_START)).clamp(0.0, 1.0)
    shape = (torch.minimum(u, 1.0 - u) / BOW_RAMP).clamp(0.0, 1.0)
    return torch.sin(torch.tensor(BOW_PITCH, device=p.device)) * shape


def bow_pitch(env, command_name: str, std: float = 0.1,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Tangage du tronc proche du profil de salut pendant la fenetre (gaussienne sur sin(tangage))."""
    p = _phase(env, command_name)
    g = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b[:, 0], nan=0.0)
    return _in(p, BOW_START, BOW_END) * torch.exp(-((g - _bow_target_sin(p)) / std) ** 2)


def bow_feet_penalty(env, command_name: str) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si un pied quitte le sol pendant le salut."""
    p = _phase(env, command_name)
    return -_in(p, BOW_START, BOW_END) * (1.0 - _feet_down(env))


def bow_roll_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) sur le roulis seul (composante y de la gravite projetee) : on salue de face,
    sans pencher de cote. Dans toutes les phases."""
    g = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b[:, 1], nan=0.0)
    return -torch.clamp(g.abs() / 0.3, 0.0, 1.0)


def bow_drift(env, command_name: str, vel_std: float = 0.1,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pas de deplacement horizontal pendant le salut (recompense)."""
    p = _phase(env, command_name)
    v = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2], nan=0.0).norm(dim=-1)
    return _in(p, BOW_START, BOW_END) * torch.exp(-(v / vel_std) ** 2)


def bow_rest_upright(env, command_name: str, std: float = 0.1,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Tronc droit (tangage ~ 0) hors de la fenetre de salut : le canard se redresse completement."""
    p = _phase(env, command_name)
    g = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b[:, 0], nan=0.0)
    return (1.0 - _in(p, BOW_START, BOW_END)) * torch.exp(-(g / std) ** 2)


def make_microduck_bow_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_hop_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S
    for name in ("hop_crouch", "hop_air", "hop_height", "hop_launch", "hop_land_upright",
                 "hop_mid_contact_penalty", "hop_tilt_penalty"):
        cfg.rewards.pop(name, None)

    c = {"command_name": "twist"}
    cfg.rewards["bow_pitch"] = RewardTermCfg(func=bow_pitch, weight=6.0, params=dict(c))
    cfg.rewards["bow_feet_penalty"] = RewardTermCfg(func=bow_feet_penalty, weight=3.0, params=dict(c))
    cfg.rewards["bow_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=3.0)
    cfg.rewards["bow_drift"] = RewardTermCfg(func=bow_drift, weight=1.5, params=dict(c))
    cfg.rewards["bow_rest_upright"] = RewardTermCfg(func=bow_rest_upright, weight=3.0, params=dict(c))
    return cfg


MicroduckBowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_bow",
    run_name="microduck_bow",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
