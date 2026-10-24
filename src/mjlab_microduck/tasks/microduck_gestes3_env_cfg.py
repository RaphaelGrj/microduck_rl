"""Microduck gestes III (2026-10-24) : ETIREMENT SUR UNE JAMBE, ARRET THEATRAL.

Meme famille que les gestes II : commande CONSTANTE (aucune phase), profil temporel mesure depuis le debut de l'episode.

LEGSTRETCH — un pied se leve et RESTE en l'air (contrairement au tape / trepigner), l'autre porte. Fenetre 0.30-1.50 s,
    episode 2.0 s (= duration_s).
STOPTHEATRE — course avant a ~STOP_VX m/s pendant STOP_T (commande = zeros) puis arret franc, nez qui plonge un peu, sur le
    signal d'unwind (commande [1,0,0], comme la pirouette). Manifest vise : perpetual, --hold STOP_T, unwind_s STOP_UNWIND,
    idle [1,0,0].
"""

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import bow_roll_penalty
from mjlab_microduck.tasks.microduck_gestes2_env_cfg import _base, _rl_cfg
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z, _trunk_z, hop_tilt_penalty
from mjlab_microduck.tasks.microduck_shiver_env_cfg import shiver_lean_penalty, shiver_yaw_penalty
from mjlab_microduck.tasks.microduck_startle_env_cfg import _t
from mjlab_microduck.tasks.microduck_stomp_env_cfg import _feet_found, _update

# =========================================================================== #
# LEGSTRETCH                                                                  #
# =========================================================================== #

LS_DURATION_S = 2.0
LS_START, LS_END = 0.30, 1.50
LS_REST_FROM = 1.70
LS_FOOT = 1             # pied leve ; l'autre porte


def _ls_window(env) -> torch.Tensor:
    t = _t(env)
    return ((t >= LS_START) & (t < LS_END)).float()


def ls_raised(env) -> torch.Tensor:
    """Pied leve en l'air ET pied d'appui au sol, dans la fenetre : paye la jambe tenue."""
    _update(env)
    on = _feet_found(env)
    return _ls_window(env) * ((~on[:, LS_FOOT]) & on[:, 1 - LS_FOOT]).float()


def ls_support_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si le pied d'appui quitte le sol, dans TOUT l'episode."""
    _update(env)
    return -(~_feet_found(env)[:, 1 - LS_FOOT]).float()


def ls_down_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si le pied leve touche le sol pendant la fenetre (pas de tape)."""
    _update(env)
    return -_ls_window(env) * _feet_found(env)[:, LS_FOOT].float()


def ls_rest_pose(env, std: float = 0.5,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return (_t(env) >= LS_REST_FROM).float() * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def ls_rest_height(env, std: float = 0.015,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    return (_t(env) >= LS_REST_FROM).float() * torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def ls_height(env, std: float = 0.02,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    """Tronc pres de la hauteur debout pendant tout l'episode : on s'etire, on ne s'effondre pas."""
    return torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def make_microduck_legstretch_env_cfg(play: bool = False, rough: bool = False):
    cfg = _base(play, rough, LS_DURATION_S)
    cfg.rewards["ls_raised"] = RewardTermCfg(func=ls_raised, weight=4.0)
    cfg.rewards["ls_support_penalty"] = RewardTermCfg(func=ls_support_penalty, weight=6.0)
    cfg.rewards["ls_down_penalty"] = RewardTermCfg(func=ls_down_penalty, weight=2.0)
    cfg.rewards["ls_lean_penalty"] = RewardTermCfg(func=shiver_lean_penalty, weight=3.0)
    cfg.rewards["ls_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=3.0)
    cfg.rewards["ls_tilt_penalty"] = RewardTermCfg(func=hop_tilt_penalty, weight=6.0)
    cfg.rewards["ls_height"] = RewardTermCfg(func=ls_height, weight=3.0)
    cfg.rewards["ls_yaw_penalty"] = RewardTermCfg(func=shiver_yaw_penalty, weight=2.0)
    cfg.rewards["ls_rest_pose"] = RewardTermCfg(func=ls_rest_pose, weight=3.0)
    cfg.rewards["ls_rest_height"] = RewardTermCfg(func=ls_rest_height, weight=3.0)
    return cfg


MicroduckLegStretchRlCfg = _rl_cfg("microduck_legstretch")


# =========================================================================== #
# STOPTHEATRE                                                                 #
# =========================================================================== #

STOP_VX = 0.30          # m/s, vers l'avant
STOP_T = 0.8            # s de course (commande = zeros)  = --hold
STOP_UNWIND = 0.9       # s d'arret (commande = [1,0,0])  = unwind_s
STOP_DURATION_S = STOP_T + STOP_UNWIND
STOP_BRAKE_END = STOP_T + 0.5
STOP_REST_FROM = STOP_T + 0.6
STOP_DIP = 0.15         # sin(tangage) vise pendant le freinage : le nez plonge


def stop_vx(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vitesse avant (repere du tronc) payee jusqu'a STOP_VX pendant la course."""
    vx = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0], nan=0.0)
    return (_t(env) < STOP_T).float() * torch.clamp(vx / STOP_VX, 0.0, 1.0)


def stop_lateral_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) sur la vitesse laterale : une course droite."""
    vy = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 1], nan=0.0)
    return -torch.clamp(vy.abs() / 0.15, 0.0, 1.0)


def stop_brake(env, vel_std: float = 0.12,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Arret franc : vitesse avant quasi nulle des STOP_BRAKE_END (la remise de main n'attend pas)."""
    vx = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0], nan=0.0)
    return (_t(env) >= STOP_BRAKE_END).float() * torch.exp(-(vx / vel_std) ** 2)


def stop_dip(env, std: float = 0.08,
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Le nez plonge pendant le freinage (theatral) : sin(tangage) proche de STOP_DIP."""
    t = _t(env)
    g = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b[:, 0], nan=0.0)
    return ((t >= STOP_T) & (t < STOP_BRAKE_END)).float() * torch.exp(-((g - STOP_DIP) / std) ** 2)


def stop_both_off_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si les deux pieds sont en l'air : une course, pas un saut."""
    _update(env)
    return -((~_feet_found(env)).sum(dim=-1) == 2).float()


def stop_rest_still(env, vel_std: float = 0.2,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    w = torch.nan_to_num(data.root_link_ang_vel_w, nan=0.0).norm(dim=-1)
    v = torch.nan_to_num(data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    return (_t(env) >= STOP_REST_FROM).float() * torch.exp(-((w + v) / vel_std) ** 2)


def stop_rest_upright(env, std: float = 0.1,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Retour droit en fin d'episode : tangage et hauteur de repos (la remise de main ne doit pas trouver le nez plonge)."""
    g = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b[:, 0], nan=0.0)
    return (_t(env) >= STOP_REST_FROM).float() * torch.exp(-(g / std) ** 2)


def stop_rest_pose(env, std: float = 0.5,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return (_t(env) >= STOP_REST_FROM).float() * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def stop_height(env, std: float = 0.02,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    return torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def make_microduck_stoptheatre_env_cfg(play: bool = False, rough: bool = False):
    cfg = _base(play, rough, STOP_DURATION_S)
    cfg.rewards.pop("startle_displacement_penalty", None)

    base = make_velocity_env_cfg().commands["twist"]
    base.rel_standing_envs = 0.0
    base.rel_heading_envs = 0.0
    base.heading_command = False
    base.ranges.heading = None
    base.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SpinStopCommandCfg(
        **{**vars(base), "class_type": microduck_mdp.SpinStopCommand, "period": STOP_T, "randomize_phase": False}
    )

    cfg.rewards["stop_vx"] = RewardTermCfg(func=stop_vx, weight=6.0)
    cfg.rewards["stop_lateral_penalty"] = RewardTermCfg(func=stop_lateral_penalty, weight=3.0)
    cfg.rewards["stop_yaw_penalty"] = RewardTermCfg(func=shiver_yaw_penalty, weight=2.0)
    cfg.rewards["stop_brake"] = RewardTermCfg(func=stop_brake, weight=6.0)
    cfg.rewards["stop_tilt_penalty"] = RewardTermCfg(func=hop_tilt_penalty, weight=4.0)
    cfg.rewards["stop_height"] = RewardTermCfg(func=stop_height, weight=3.0)
    cfg.rewards["stop_dip"] = RewardTermCfg(func=stop_dip, weight=2.0)
    cfg.rewards["stop_both_off_penalty"] = RewardTermCfg(func=stop_both_off_penalty, weight=4.0)
    cfg.rewards["stop_rest_still"] = RewardTermCfg(func=stop_rest_still, weight=6.0)
    cfg.rewards["stop_rest_upright"] = RewardTermCfg(func=stop_rest_upright, weight=3.0)
    cfg.rewards["stop_rest_pose"] = RewardTermCfg(func=stop_rest_pose, weight=2.0)
    return cfg


MicroduckStopTheatreRlCfg = _rl_cfg("microduck_stoptheatre")
