"""Microduck STARTLE task — sursaut : le tronc se cabre brievement (nez vers le haut) puis se redresse, pieds au sol.

Contrairement a Hop / Bow / Shake, AUCUNE commande de phase : la commande est constante (twist ~ 0, tete et corps a zero),
comme les tirs. C'est ce qui rend la politique publiable comme VRAI skill (`robotctl policy add`, `kind: episodic`,
`command.encoding: constant`) : robotd refuse les politiques a phase hors du slot ground_pick.

Le geste demarre des le declenchement (debout, immobile) et suit un profil temporel mesure depuis le debut de l'episode :

    t 0.00-0.15  le tronc se cabre vite (sin(tangage) -> -STARTLE_PITCH, nez vers le haut)
    t 0.15-0.25  maintien
    t 0.25-0.75  retour a l'horizontale
    t 0.75-1.20  debout, immobile

L'episode dure STARTLE_DURATION_S = la duree declaree dans le manifest (`duration_s`) : le demon rend la main ensuite.
La politique n'a pas d'horloge : elle ne peut pas savoir si elle est a t=0 ou a t=1 (meme etat debout) ; l'episode
s'arrete donc avant qu'un second sursaut puisse etre appris ou paye.

PREMIERE VERSION : a verifier par un pilote de 400 iterations. Sentinelles : startle_pitch (doit monter),
startle_feet_penalty (<= 0), bow_roll_penalty (<= 0), startle_drift.
"""

import torch

from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import bow_roll_penalty
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    _LEG_JOINTS,
    STAND_Z,
    _trunk_z,
    make_microduck_hop_env_cfg,
)
from mjlab_microduck.tasks.microduck_shake_env_cfg import _feet_down
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg

STARTLE_DURATION_S = 1.2   # = duration_s du manifest
STARTLE_PITCH = 0.25       # rad, cabrage vise au maintien (~14 deg)
T_UP, T_HOLD_END, T_DOWN_END = 0.15, 0.25, 0.75
T_REST = 0.80              # au-dela, repos exige


def _t(env) -> torch.Tensor:
    """Temps ecoule depuis le debut de l'episode (s)."""
    return env.episode_length_buf.float() * env.step_dt


def _target_sin(t: torch.Tensor) -> torch.Tensor:
    """sin du tangage vise (negatif = nez vers le haut) : montee rapide, maintien, descente lente."""
    up = (t / T_UP).clamp(0.0, 1.0)
    down = ((T_DOWN_END - t) / (T_DOWN_END - T_HOLD_END)).clamp(0.0, 1.0)
    shape = torch.minimum(up, down)
    return -torch.sin(torch.tensor(STARTLE_PITCH, device=t.device)) * shape


def startle_pitch(env, std: float = 0.15,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Tangage du tronc proche du profil de sursaut (gaussienne sur sin(tangage)), tout au long de l'episode : apres
    T_DOWN_END la cible est 0, donc le meme terme exige de se redresser."""
    g = torch.nan_to_num(env.scene[asset_cfg.name].data.projected_gravity_b[:, 0], nan=0.0)
    return torch.exp(-((g - _target_sin(_t(env))) / std) ** 2)


def startle_feet_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si un pied quitte le sol : un sursaut sans sauter."""
    return -(1.0 - _feet_down(env))


def startle_drift(env, vel_std: float = 0.1,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pas de deplacement horizontal (recompense)."""
    v = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2], nan=0.0).norm(dim=-1)
    return torch.exp(-(v / vel_std) ** 2)


def startle_displacement_penalty(env, full_m: float = 0.10,
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) proportionnelle au DEPLACEMENT cumule depuis le debut de l'episode (0 -> 1
    jusqu'a full_m). Pilote v1 (2026-10-24) : startle_drift (vitesse instantanee) valait 0,97 mais le canard finissait
    a 50 cm : le cabrage etait un coup de pied vers l'avant."""
    pos = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_pos_w[:, :2], nan=0.0)
    start = env.episode_length_buf <= 1
    if not hasattr(env, "_startle_x0"):
        env._startle_x0 = pos.clone()
    env._startle_x0[start] = pos[start]
    return -torch.clamp((pos - env._startle_x0).norm(dim=-1) / full_m, 0.0, 1.0)


def _rest(env) -> torch.Tensor:
    return (_t(env) >= T_REST).float()


def startle_rest_pose(env, std: float = 0.5,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pose debout (jambes) a la fin du geste."""
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return _rest(env) * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def startle_rest_height(env, std: float = 0.01,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    return _rest(env) * torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def startle_rest_still(env, vel_std: float = 0.1,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    v = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w, nan=0.0)
    return _rest(env) * torch.exp(-(v.norm(dim=-1) / vel_std) ** 2)


def make_microduck_startle_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_hop_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = STARTLE_DURATION_S
    for name in [n for n in cfg.rewards if n.startswith("hop_")]:
        cfg.rewards.pop(name)

    # Commande constante ~ 0 (parite de forme de l'observation uniquement), comme les tirs
    base = make_velocity_env_cfg().commands["twist"]
    base.rel_standing_envs = 0.0
    base.rel_heading_envs = 0.0
    base.heading_command = False
    base.ranges.heading = None
    base.debug_vis = False
    base.resampling_time_range = (STARTLE_DURATION_S, STARTLE_DURATION_S * 2)
    base.ranges.lin_vel_x = (-0.01, 0.01)
    base.ranges.lin_vel_y = (-0.01, 0.01)
    base.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(base))

    # Le runtime met la tete a zero pour tout skill : plus de commande de tete aleatoire
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
    # head_pose_tracking viserait une commande que la politique ne voit plus ; upright_linear luttait contre le cabrage
    for name in ("head_pose_tracking", "upright_linear"):
        cfg.rewards.pop(name, None)

    cfg.rewards["startle_pitch"] = RewardTermCfg(func=startle_pitch, weight=6.0)
    cfg.rewards["startle_feet_penalty"] = RewardTermCfg(func=startle_feet_penalty, weight=6.0)
    cfg.rewards["startle_displacement_penalty"] = RewardTermCfg(func=startle_displacement_penalty, weight=4.0)
    cfg.rewards["bow_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=3.0)
    cfg.rewards["startle_drift"] = RewardTermCfg(func=startle_drift, weight=1.5)
    cfg.rewards["startle_rest_pose"] = RewardTermCfg(func=startle_rest_pose, weight=3.0)
    cfg.rewards["startle_rest_height"] = RewardTermCfg(func=startle_rest_height, weight=3.0)
    cfg.rewards["startle_rest_still"] = RewardTermCfg(func=startle_rest_still, weight=1.5)
    # Cou/tete au neutre sur TOUT l'episode : pilote v2 (2026-10-24), rien ne retenait la tete (commande zeros, pas de
    # recompense de tete) et elle s'affaissait vers l'avant des le milieu du geste (vu sur le rendu video).
    cfg.rewards["startle_neck_home"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=3.0,
        params={"std": 0.3, "joint_indices": [5, 6, 7, 8], "target_overrides": None},
    )
    return cfg


MicroduckStartleRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_startle",
    run_name="microduck_startle",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
