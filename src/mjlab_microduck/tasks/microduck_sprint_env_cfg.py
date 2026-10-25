"""Microduck SPRINT task — course droite courte a vitesse fixe, puis arret propre, tronc droit.

Meme famille que la Pirouette (commande constante -> vrai skill, signal d'arret via `unwind`) :

    t 0.0 - SPRINT_T         commande = zeros (le skill court)           -> avance a SPRINT_VX m/s, cap tenu
    t SPRINT_T - +UNWIND_S   commande = [1, 0, 0] (drapeau d'arret)      -> freine, se redresse, immobile

Manifest vise : duration_s = SPRINT_T, unwind_s = UNWIND_S, command.idle = [1, 0, 0].

Usage : rattraper une balle qui roule, Zoomies. La marche officielle plafonne a ~0,4 m/s (zone morte en dessous de 0,25) ;
le sprint vise le double. Recompenses : vitesse avant (repere du tronc) payee jusqu'a SPRINT_VX, plafonnee ; cap tenu
(vitesse laterale et lacet punis) ; tronc droit et a hauteur debout ; immobilite en fin de retour.
Sentinelles : sprint_speed (doit monter), sprint_rest_still (doit monter en fin d'episode), sprint_tilt_penalty.
"""

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import bow_roll_penalty
from mjlab_microduck.tasks.microduck_hop_env_cfg import hop_tilt_penalty
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS
from mjlab_microduck.tasks.microduck_pirouette_env_cfg import spin_height
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
from mjlab_microduck.tasks.microduck_startle_env_cfg import _t, make_microduck_startle_env_cfg

SPRINT_VX = 0.8                        # m/s (marche officielle : ~0,4)
SPRINT_T = 1.5                         # s de course = duration_s du manifest
UNWIND_S = 1.0                         # s de retour = unwind_s du manifest
SPRINT_DURATION_S = SPRINT_T + UNWIND_S
REST_FROM = SPRINT_T + 0.4



def _running(env) -> torch.Tensor:
    return (_t(env) < SPRINT_T).float()


def sprint_speed(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vitesse avant (repere du tronc) payee jusqu'a SPRINT_VX pendant la course."""
    vx = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0], nan=0.0)
    return _running(env) * torch.clamp(vx / SPRINT_VX, 0.0, 1.0)


def sprint_overshoot_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (poids POSITIF) au-dela de 1,3 x SPRINT_VX : un sprint, pas une fusee."""
    vx = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0], nan=0.0)
    return -torch.clamp((vx - 1.3 * SPRINT_VX) / SPRINT_VX, 0.0, 1.0)


def sprint_heading_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Cap tenu pendant la course : vitesse laterale + lacet punis (poids POSITIF)."""
    data = env.scene[asset_cfg.name].data
    vy = torch.nan_to_num(data.root_link_lin_vel_b[:, 1], nan=0.0).abs()
    wz = torch.nan_to_num(data.root_link_ang_vel_b[:, 2], nan=0.0).abs()
    return -_running(env) * torch.clamp(vy / 0.3 + wz / 1.5, 0.0, 2.0)


def sprint_rest_still(env, vel_std: float = 0.3,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Immobilite (vitesses angulaire ET lineaire) en fin de retour."""
    data = env.scene[asset_cfg.name].data
    w = torch.nan_to_num(data.root_link_ang_vel_w, nan=0.0).norm(dim=-1)
    v = torch.nan_to_num(data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    return (_t(env) >= REST_FROM).float() * torch.exp(-((w + v) / vel_std) ** 2)


def sprint_rest_pose(env, std: float = 0.5,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pose debout (jambes) en fin de retour."""
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return (_t(env) >= REST_FROM).float() * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def make_microduck_sprint_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_startle_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = SPRINT_DURATION_S
    for name in [n for n in cfg.rewards if n.startswith("startle_") and n != "startle_neck_home"]:
        cfg.rewards.pop(name)
    for name in ("bow_roll_penalty", "body_ang_vel", "angular_momentum"):
        cfg.rewards.pop(name, None)
    cfg.rewards["action_rate_l2"].weight = -0.02

    base = make_velocity_env_cfg().commands["twist"]
    base.rel_standing_envs = 0.0
    base.rel_heading_envs = 0.0
    base.heading_command = False
    base.ranges.heading = None
    base.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SpinStopCommandCfg(
        **{**vars(base), "class_type": microduck_mdp.SpinStopCommand, "period": SPRINT_T, "randomize_phase": False}
    )

    cfg.rewards["sprint_speed"] = RewardTermCfg(func=sprint_speed, weight=6.0)
    cfg.rewards["sprint_overshoot_penalty"] = RewardTermCfg(func=sprint_overshoot_penalty, weight=3.0)
    cfg.rewards["sprint_heading_penalty"] = RewardTermCfg(func=sprint_heading_penalty, weight=1.5)
    cfg.rewards["sprint_tilt_penalty"] = RewardTermCfg(func=hop_tilt_penalty, weight=5.0)
    cfg.rewards["sprint_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=1.5)
    cfg.rewards["sprint_height"] = RewardTermCfg(func=spin_height, weight=2.0)
    cfg.rewards["sprint_rest_still"] = RewardTermCfg(func=sprint_rest_still, weight=6.0)
    cfg.rewards["sprint_rest_pose"] = RewardTermCfg(func=sprint_rest_pose, weight=2.0)
    return cfg


MicroduckSprintRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_sprint",
    run_name="microduck_sprint",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
