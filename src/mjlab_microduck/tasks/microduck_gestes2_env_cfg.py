"""Microduck gestes II (2026-10-24) : TAPE DU PIED, ACCROUPISSEMENT CURIEUX, PAS DE COTE.

Meme famille que Startle / Shiver / Stomp / Pirouette : commande CONSTANTE (aucune phase) -> politiques publiables comme vrais
skills. Profil temporel mesure depuis le debut de l'episode ; la politique n'a pas d'horloge.

TAP — un pied frappe le sol a repetition, l'autre reste pose. Fenetre 0.10-1.30 s, episode 1.6 s (= duration_s).
CROUCH — le tronc descend, tient la pose, remonte. 0.10-0.50 descente, 0.50-1.00 maintien, 1.00-1.50 remontee, episode 1.8 s.
SIDESTEP — un pas de cote (+y, a gauche) a ~SIDE_VY m/s pendant SIDE_T, puis arret sur le signal d'unwind (commande [1,0,0],
    comme la pirouette : sans horloge la politique ne sait pas qu'elle a deja fait son pas). Episode SIDE_T + SIDE_UNWIND.
    Manifest vise : perpetual, --hold SIDE_T, unwind_s SIDE_UNWIND, idle [1,0,0].
"""

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import bow_roll_penalty
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z, _trunk_z, hop_tilt_penalty
from mjlab_microduck.tasks.microduck_shiver_env_cfg import shiver_lean_penalty, shiver_yaw_penalty
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
from mjlab_microduck.tasks.microduck_startle_env_cfg import (
    _t,
    make_microduck_startle_env_cfg,
    startle_feet_penalty,
)
from mjlab_microduck.tasks.microduck_stomp_env_cfg import _feet_found, _update

_KEEP = ("startle_displacement_penalty", "startle_neck_home")


def _base(play: bool, rough: bool, episode_s: float):
    cfg = make_microduck_startle_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = episode_s
    for name in [n for n in cfg.rewards if n.startswith("startle_") and n not in _KEEP]:
        cfg.rewards.pop(name)
    for name in ("bow_roll_penalty", "body_ang_vel", "angular_momentum"):
        cfg.rewards.pop(name, None)
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.commands["twist"].resampling_time_range = (episode_s, episode_s * 2)
    return cfg


def _rl_cfg(name: str, iterations: int = 2_000) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=MicroduckSitStandRlCfg.actor,
        critic=MicroduckSitStandRlCfg.critic,
        algorithm=MicroduckSitStandRlCfg.algorithm,
        wandb_project="mjlab_microduck",
        experiment_name=name,
        run_name=name,
        save_interval=250,
        num_steps_per_env=24,
        max_iterations=iterations,
    )


# =========================================================================== #
# TAP — tape du pied                                                          #
# =========================================================================== #

TAP_DURATION_S = 1.6
TAP_START, TAP_END = 0.10, 1.30
TAP_FOOT = 0            # indice (dans feet_ground_contact) du pied qui tape ; l'autre est le pied d'appui
TAP_HOLD_FREE_S = 0.30


def _tap_window(env) -> torch.Tensor:
    t = _t(env)
    return ((t >= TAP_START) & (t < TAP_END)).float()


def tap_raised(env) -> torch.Tensor:
    """Le pied qui tape est en l'air ET le pied d'appui au sol (dans la fenetre) : paye le decollage."""
    _update(env)
    on = _feet_found(env)
    return _tap_window(env) * ((~on[:, TAP_FOOT]) & on[:, 1 - TAP_FOOT]).float()


def tap_touchdown(env) -> torch.Tensor:
    """Repose du pied qui tape apres un vrai decollage : un coup au sol, a repeter."""
    _update(env)
    return _tap_window(env) * env._stomp_touch_each[:, TAP_FOOT]


def tap_support_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si le pied d'appui quitte le sol, dans TOUT l'episode."""
    _update(env)
    return -(~_feet_found(env)[:, 1 - TAP_FOOT]).float()


def tap_hold_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si le pied qui tape reste en l'air plus de TAP_HOLD_FREE_S (pas un flamant)."""
    _update(env)
    return -torch.clamp((env._stomp_air[:, TAP_FOOT] - TAP_HOLD_FREE_S) / 0.3, 0.0, 1.0)


def make_microduck_tap_env_cfg(play: bool = False, rough: bool = False):
    cfg = _base(play, rough, TAP_DURATION_S)
    cfg.rewards["tap_raised"] = RewardTermCfg(func=tap_raised, weight=3.0)
    cfg.rewards["tap_touchdown"] = RewardTermCfg(func=tap_touchdown, weight=4.0)
    cfg.rewards["tap_support_penalty"] = RewardTermCfg(func=tap_support_penalty, weight=6.0)
    cfg.rewards["tap_hold_penalty"] = RewardTermCfg(func=tap_hold_penalty, weight=4.0)
    cfg.rewards["tap_lean_penalty"] = RewardTermCfg(func=shiver_lean_penalty, weight=2.0)
    cfg.rewards["tap_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=1.0)
    cfg.rewards["tap_yaw_penalty"] = RewardTermCfg(func=shiver_yaw_penalty, weight=2.0)
    return cfg


MicroduckTapRlCfg = _rl_cfg("microduck_tap")


# =========================================================================== #
# CROUCH — accroupissement curieux                                            #
# =========================================================================== #

CROUCH_DURATION_S = 1.8
CROUCH_Z = 0.090        # hauteur de tronc au maintien (debout : STAND_Z ~ 0.115)
T_DOWN0, T_DOWN1, T_UP0, T_UP1 = 0.10, 0.50, 1.00, 1.50


def _crouch_target_z(t: torch.Tensor) -> torch.Tensor:
    down = ((t - T_DOWN0) / (T_DOWN1 - T_DOWN0)).clamp(0.0, 1.0)
    up = ((t - T_UP0) / (T_UP1 - T_UP0)).clamp(0.0, 1.0)
    return STAND_Z - (STAND_Z - CROUCH_Z) * (down - up)


def crouch_height(env, std: float = 0.012,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    """Hauteur du tronc proche du profil descente / maintien / remontee (gaussienne)."""
    return torch.exp(-((_trunk_z(env, asset_cfg) - _crouch_target_z(_t(env))) / std) ** 2)


def crouch_drift(env, vel_std: float = 0.1,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    v = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2], nan=0.0).norm(dim=-1)
    return torch.exp(-(v / vel_std) ** 2)


def make_microduck_crouch_env_cfg(play: bool = False, rough: bool = False):
    cfg = _base(play, rough, CROUCH_DURATION_S)
    cfg.rewards["crouch_height"] = RewardTermCfg(func=crouch_height, weight=6.0)
    cfg.rewards["crouch_feet_penalty"] = RewardTermCfg(func=startle_feet_penalty, weight=5.0)
    cfg.rewards["crouch_drift"] = RewardTermCfg(func=crouch_drift, weight=1.5)
    cfg.rewards["crouch_lean_penalty"] = RewardTermCfg(func=shiver_lean_penalty, weight=3.0)
    cfg.rewards["crouch_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=2.0)
    cfg.rewards["crouch_yaw_penalty"] = RewardTermCfg(func=shiver_yaw_penalty, weight=2.0)
    return cfg


MicroduckCrouchRlCfg = _rl_cfg("microduck_crouch")


# =========================================================================== #
# SIDESTEP — pas de cote                                                      #
# =========================================================================== #

SIDE_VY = 0.25          # m/s, vers la gauche (+y du tronc)
SIDE_T = 0.8            # s de deplacement (commande = zeros)  = --hold
SIDE_UNWIND = 0.8       # s de retour (commande = [1,0,0])     = unwind_s
SIDE_DURATION_S = SIDE_T + SIDE_UNWIND
SIDE_REST_FROM = SIDE_T + 0.4


def sidestep_vy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vitesse laterale (repere du tronc, +y = gauche) payee jusqu'a SIDE_VY pendant le deplacement."""
    vy = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 1], nan=0.0)
    return (_t(env) < SIDE_T).float() * torch.clamp(vy / SIDE_VY, 0.0, 1.0)


def sidestep_forward_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) sur la vitesse avant/arriere : un pas de cote, pas une marche en diagonale."""
    vx = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0], nan=0.0)
    return -torch.clamp(vx.abs() / 0.15, 0.0, 1.0)


def sidestep_both_off_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si les deux pieds sont en l'air : on glisse un pied, on ne saute pas."""
    _update(env)
    return -((~_feet_found(env)).sum(dim=-1) == 2).float()


def sidestep_rest_still(env, vel_std: float = 0.2,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Immobilite (lineaire + angulaire) en fin de retour : le canard a fini de freiner."""
    data = env.scene[asset_cfg.name].data
    w = torch.nan_to_num(data.root_link_ang_vel_w, nan=0.0).norm(dim=-1)
    v = torch.nan_to_num(data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    return (_t(env) >= SIDE_REST_FROM).float() * torch.exp(-((w + v) / vel_std) ** 2)


def sidestep_rest_pose(env, std: float = 0.5,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return (_t(env) >= SIDE_REST_FROM).float() * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def sidestep_height(env, std: float = 0.02,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    return torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def make_microduck_sidestep_env_cfg(play: bool = False, rough: bool = False):
    cfg = _base(play, rough, SIDE_DURATION_S)
    # deplacement autorise (c'est le geste) : on retire la penalite de deplacement
    cfg.rewards.pop("startle_displacement_penalty", None)

    base = make_velocity_env_cfg().commands["twist"]
    base.rel_standing_envs = 0.0
    base.rel_heading_envs = 0.0
    base.heading_command = False
    base.ranges.heading = None
    base.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SpinStopCommandCfg(
        **{**vars(base), "class_type": microduck_mdp.SpinStopCommand, "period": SIDE_T, "randomize_phase": False}
    )

    cfg.rewards["sidestep_vy"] = RewardTermCfg(func=sidestep_vy, weight=6.0)
    cfg.rewards["sidestep_forward_penalty"] = RewardTermCfg(func=sidestep_forward_penalty, weight=3.0)
    cfg.rewards["sidestep_yaw_penalty"] = RewardTermCfg(func=shiver_yaw_penalty, weight=2.0)
    cfg.rewards["sidestep_both_off_penalty"] = RewardTermCfg(func=sidestep_both_off_penalty, weight=4.0)
    cfg.rewards["sidestep_tilt_penalty"] = RewardTermCfg(func=hop_tilt_penalty, weight=4.0)
    cfg.rewards["sidestep_height"] = RewardTermCfg(func=sidestep_height, weight=2.0)
    cfg.rewards["sidestep_rest_still"] = RewardTermCfg(func=sidestep_rest_still, weight=6.0)
    cfg.rewards["sidestep_rest_pose"] = RewardTermCfg(func=sidestep_rest_pose, weight=2.0)
    return cfg


MicroduckSidestepRlCfg = _rl_cfg("microduck_sidestep")
