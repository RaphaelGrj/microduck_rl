"""Microduck SHIVER task — frisson : le tronc tremble vite de gauche a droite (roulis), pieds au sol, sur place.

Meme famille que Startle : commande CONSTANTE (aucune phase) -> politique publiable comme vrai skill episodique.
Profil temporel mesure depuis le debut de l'episode :

    t 0.00-0.10  debout, immobile
    t 0.10-1.30  tremblement : vitesse angulaire de roulis elevee, angle de roulis petit (oscillation, pas un penche)
    t 1.30-1.60  debout, immobile

Recompense du tremblement = |vitesse de roulis| (plafonnee) x (angle de roulis reste petit) : pour garder une grande
vitesse sans grand angle, la politique n'a pas d'autre choix que d'osciller. Cou au neutre sur tout l'episode (lecon du
Sursaut v2 : sans cela la tete s'affaisse), pieds au sol, pas de deplacement, pas de rotation de lacet.

PREMIERE VERSION : a verifier par un pilote de 400 iterations. Sentinelles : shiver_roll_rate (doit monter),
shiver_lean_penalty / shiver_yaw_penalty (<= 0), startle_feet_penalty, startle_displacement_penalty.
"""

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z, _trunk_z
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
from mjlab_microduck.tasks.microduck_startle_env_cfg import (
    _t,
    make_microduck_startle_env_cfg,
    startle_displacement_penalty,
    startle_feet_penalty,
)

SHIVER_DURATION_S = 1.6   # = duration_s du manifest
T_START, T_END = 0.10, 1.30
T_REST = 1.40
ROLL_RATE_FULL = 2.5      # rad/s : vitesse de roulis payee a 100 %
ROLL_ANGLE_STD = 0.15     # sin(roulis) : au-dela, le tremblement devient un penche


def _window(env) -> torch.Tensor:
    t = _t(env)
    return ((t >= T_START) & (t < T_END)).float()


def shiver_roll_rate(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vitesse angulaire de roulis (axe x du tronc) dans la fenetre, payee seulement si l'angle de roulis reste petit."""
    data = env.scene[asset_cfg.name].data
    wx = torch.nan_to_num(data.root_link_ang_vel_b[:, 0], nan=0.0)
    gy = torch.nan_to_num(data.projected_gravity_b[:, 1], nan=0.0)
    return _window(env) * torch.clamp(wx.abs() / ROLL_RATE_FULL, 0.0, 1.0) * torch.exp(-((gy / ROLL_ANGLE_STD) ** 2))


def shiver_lean_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) sur le tangage (composante x de la gravite) : on tremble, on ne se cabre pas."""
    gx = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b[:, 0], nan=0.0)
    return -torch.clamp(gx.abs() / 0.2, 0.0, 1.0)


def shiver_yaw_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) sur la rotation de lacet : on ne pivote pas."""
    wz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2], nan=0.0)
    return -torch.clamp(wz.abs() / 1.0, 0.0, 1.0)


def _rest(env) -> torch.Tensor:
    return (_t(env) >= T_REST).float()


def shiver_rest_pose(env, std: float = 0.5,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return _rest(env) * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def shiver_rest_height(env, std: float = 0.01,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    return _rest(env) * torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def shiver_rest_still(env, vel_std: float = 0.3,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    w = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_ang_vel_b, nan=0.0)
    return _rest(env) * torch.exp(-(w.norm(dim=-1) / vel_std) ** 2)


def make_microduck_shiver_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_startle_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = SHIVER_DURATION_S
    for name in [n for n in cfg.rewards if n.startswith("startle_") and n not in
                 ("startle_feet_penalty", "startle_displacement_penalty", "startle_neck_home")]:
        cfg.rewards.pop(name)
    cfg.rewards.pop("bow_roll_penalty", None)   # le roulis est justement le geste

    # Penalites heritees qui puniraient le tremblement lui-meme (vitesse angulaire, moment cinetique, cadence d'action)
    for name in ("body_ang_vel", "angular_momentum"):
        cfg.rewards.pop(name, None)
    cfg.rewards["action_rate_l2"].weight = -0.02

    cfg.commands["twist"].resampling_time_range = (SHIVER_DURATION_S, SHIVER_DURATION_S * 2)

    cfg.rewards["shiver_roll_rate"] = RewardTermCfg(func=shiver_roll_rate, weight=6.0)
    cfg.rewards["shiver_lean_penalty"] = RewardTermCfg(func=shiver_lean_penalty, weight=3.0)
    cfg.rewards["shiver_yaw_penalty"] = RewardTermCfg(func=shiver_yaw_penalty, weight=2.0)
    cfg.rewards["shiver_rest_pose"] = RewardTermCfg(func=shiver_rest_pose, weight=3.0)
    cfg.rewards["shiver_rest_height"] = RewardTermCfg(func=shiver_rest_height, weight=3.0)
    cfg.rewards["shiver_rest_still"] = RewardTermCfg(func=shiver_rest_still, weight=1.5)
    return cfg


MicroduckShiverRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_shiver",
    run_name="microduck_shiver",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
