"""Microduck HOP task — petit bond de joie sur place, debout -> debout.

Geste episodique declenche comme ground_pick / sit_toggle : la commande est la phase d'un cycle
([cos(2*pi*phi), sin(2*pi*phi), 0], phi demarre a 0 debout). Un cycle dure HOP_PERIOD secondes :

    phi 0.00-0.15  debout, immobile
    phi 0.15-0.30  s'accroupit un peu (prise d'elan)
    phi 0.30-0.52  bond : les deux pieds doivent quitter le sol, le tronc monter
    phi 0.52-0.65  reception (droit, sans glisser)
    phi 0.65-1.00  debout, immobile

Base : l'environnement sitstand (meme recette sim2real : DR, bruit, retards, observation 61D, tete commandable).
On ne garde que la machinerie ; recompenses et commande sont remplacees. Aucun prerequis nouveau cote runtime : meme
contrat d'observation que les autres politiques, donc chargeable dans un slot de robotd.

PREMIERE VERSION : a verifier par un essai court (voir scripts de smoke test) avant tout entrainement complet.
Sentinelles a surveiller dans tensorboard : hop_air (doit monter), hop_air_penalty (<= 0), Episode_Termination.
"""

import math

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    MicroduckSitStandRlCfg,
    make_microduck_sitstand_env_cfg,
)

HOP_PERIOD = 2.4          # s, un bond par cycle
EPISODE_LENGTH_S = 7.2    # 3 cycles
STAND_Z = 0.115           # hauteur du tronc debout (mesuree)
CROUCH_Z = 0.098          # prise d'elan
HOP_DZ = 0.03             # gain de hauteur du tronc vise pendant le bond (m)

# Fenetres de phase
CROUCH_START, CROUCH_END = 0.15, 0.30
AIR_START, AIR_END = 0.30, 0.52
LAND_END = 0.65

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


# --------------------------------------------------------------------------- #
# Fonctions de recompense (locales a cette tache)                              #
# --------------------------------------------------------------------------- #

def _phase(env, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * math.pi)) % 1.0


def _in(phase: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return ((phase >= lo) & (phase < hi)).float()


def _trunk_z(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )


def _both_feet_off(env) -> torch.Tensor:
    found = env.scene.sensors["feet_ground_contact"].data.found[:, :2]
    return (found.reshape(found.shape[0], -1).sum(dim=-1) == 0).float()


def hop_rest_pose(env, command_name: str, std: float = 0.5,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pose debout (articulations des jambes) hors de la fenetre bond : debout au debut et a la fin du cycle."""
    p = _phase(env, command_name)
    w = 1.0 - _in(p, CROUCH_START, LAND_END)
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos
    q0 = q0[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return w * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def hop_rest_height(env, command_name: str, std: float = 0.02,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    p = _phase(env, command_name)
    w = 1.0 - _in(p, CROUCH_START, LAND_END)
    return w * torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def hop_rest_still(env, command_name: str, vel_std: float = 0.1,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    p = _phase(env, command_name)
    w = 1.0 - _in(p, CROUCH_START, LAND_END)
    v = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w, nan=0.0)
    return w * torch.exp(-(v.norm(dim=-1) / vel_std) ** 2)


def hop_crouch(env, command_name: str, std: float = 0.012,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    p = _phase(env, command_name)
    return _in(p, CROUCH_START, CROUCH_END) * torch.exp(-((_trunk_z(env, asset_cfg) - CROUCH_Z) / std) ** 2)


def hop_air(env, command_name: str) -> torch.Tensor:
    """Les deux pieds hors du sol pendant la fenetre de bond."""
    p = _phase(env, command_name)
    return _in(p, AIR_START, AIR_END) * _both_feet_off(env)


def hop_height(env, command_name: str,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    """Hauteur gagnee par le tronc, comptee seulement en l'air (pas de grandissement des jambes)."""
    p = _phase(env, command_name)
    gain = torch.clamp((_trunk_z(env, asset_cfg) - STAND_Z) / HOP_DZ, 0.0, 1.0)
    return _in(p, AIR_START, AIR_END) * _both_feet_off(env) * gain


def hop_air_penalty(env, command_name: str) -> torch.Tensor:
    """Penalite (valeur negative, poids POSITIF) si les deux pieds quittent le sol hors de la fenetre de bond."""
    p = _phase(env, command_name)
    return -(1.0 - _in(p, AIR_START, AIR_END + 0.05)) * _both_feet_off(env)


def hop_land_upright(env, command_name: str,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    """Reception : tronc droit et sans vitesse horizontale."""
    p = _phase(env, command_name)
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, :2], nan=0.0).norm(dim=-1)
    return _in(p, AIR_END, LAND_END + 0.1) * torch.clamp(cos_tilt, 0.0, 1.0) * torch.exp(-(v / 0.15) ** 2)


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

_REMOVED_REWARDS = (
    "posture_pose_legs", "posture_pose_l1", "posture_height", "posture_height_sharp", "posture_height_l1",
    "posture_stillness", "posture_composite", "rise_bootstrap", "descent_speed", "rise_speed", "gentle_motion",
    "upright_while_tall",
)


def make_microduck_hop_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_sitstand_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S

    for name in _REMOVED_REWARDS:
        cfg.rewards.pop(name, None)
    for name in ("descent_speed_weight", "rise_speed_weight"):
        cfg.curriculum.pop(name, None)

    # Commande de phase (departure debout a phi = 0, comme au declenchement par le runtime)
    base = make_velocity_env_cfg().commands["twist"]
    base.rel_standing_envs = 0.0
    base.rel_heading_envs = 0.0
    base.heading_command = False
    base.ranges.heading = None
    base.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(base),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": HOP_PERIOD,
            "randomize_phase": False,
        }
    )

    # Toujours au depart debout
    gs = cfg.events["set_ground_state"].params
    gs["sitting_prob"] = 0.0
    gs["standing_prob"] = 1.0

    c = {"command_name": "twist"}
    cfg.rewards["hop_rest_pose"] = RewardTermCfg(func=hop_rest_pose, weight=3.0, params=dict(c))
    cfg.rewards["hop_rest_height"] = RewardTermCfg(func=hop_rest_height, weight=2.0, params=dict(c))
    cfg.rewards["hop_rest_still"] = RewardTermCfg(func=hop_rest_still, weight=1.5, params=dict(c))
    cfg.rewards["hop_crouch"] = RewardTermCfg(func=hop_crouch, weight=2.0, params=dict(c))
    cfg.rewards["hop_air"] = RewardTermCfg(func=hop_air, weight=6.0, params=dict(c))
    cfg.rewards["hop_height"] = RewardTermCfg(func=hop_height, weight=4.0, params=dict(c))
    cfg.rewards["hop_air_penalty"] = RewardTermCfg(func=hop_air_penalty, weight=3.0, params=dict(c))
    cfg.rewards["hop_land_upright"] = RewardTermCfg(func=hop_land_upright, weight=3.0, params=dict(c))
    return cfg


MicroduckHopRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_hop",
    run_name="microduck_hop",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,
)
