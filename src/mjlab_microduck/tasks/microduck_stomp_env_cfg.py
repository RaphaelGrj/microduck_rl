"""Microduck STOMP task — trepigner : les pieds se levent a tour de role, sur place, tronc droit, pieds au sol entre deux.

Meme famille que Startle / Shiver : commande CONSTANTE (aucune phase) -> politique publiable comme vrai skill episodique.
Profil temporel mesure depuis le debut de l'episode :

    t 0.00-0.10  debout, immobile
    t 0.10-1.60  trepignement : appui sur UN seul pied a la fois, alternance rapide
    t 1.60-2.00  debout, immobile

Recompenses : appui monopode paye (un pied exactement en l'air), mais un pied tenu en l'air plus de ~0.3 s est puni
(sinon la politique leve un pied et le garde = flamant) et chaque repose de pied apres un vrai decollage est payee
(sinon un seul pas suffit). Cou au neutre, pas de deplacement, pas de rotation de lacet, tronc droit.

PREMIERE VERSION : a verifier par un pilote de 400 iterations. Sentinelles : stomp_single_support, stomp_touchdown (doivent
monter), stomp_hold_penalty (<= 0), startle_displacement_penalty.
"""

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z, _trunk_z
from mjlab_microduck.tasks.microduck_shiver_env_cfg import shiver_lean_penalty, shiver_yaw_penalty
from mjlab_microduck.tasks.microduck_bow_env_cfg import bow_roll_penalty
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
from mjlab_microduck.tasks.microduck_startle_env_cfg import _t, make_microduck_startle_env_cfg

STOMP_DURATION_S = 2.0   # = duration_s du manifest
T_START, T_END = 0.10, 1.60
T_REST = 1.70
HOLD_FREE_S = 0.30       # un pied peut rester en l'air jusque-la sans penalite
MIN_AIR_S = 0.08         # decollage minimal pour qu'une repose compte


def _feet_found(env) -> torch.Tensor:
    """[envs, 2] bool : chaque pied touche le sol."""
    found = env.scene.sensors["feet_ground_contact"].data.found[:, :2]
    return found.reshape(found.shape[0], 2, -1).sum(dim=-1) > 0


def _update(env) -> None:
    """Met a jour (une fois par pas) le temps en l'air de chaque pied et le drapeau de repose. Idempotent dans un pas."""
    step = int(env.common_step_counter)
    if getattr(env, "_stomp_step", None) == step:
        return
    env._stomp_step = step
    on = _feet_found(env)
    if not hasattr(env, "_stomp_air"):
        env._stomp_air = torch.zeros(on.shape, device=on.device)
        env._stomp_touch = torch.zeros(on.shape[0], device=on.device)
    start = env.episode_length_buf <= 1
    env._stomp_air[start] = 0.0
    prev = env._stomp_air.clone()
    env._stomp_air = torch.where(on, torch.zeros_like(prev), prev + env.step_dt)
    # repose : un pied qui etait en l'air depuis au moins MIN_AIR_S touche de nouveau le sol
    env._stomp_touch = ((prev >= MIN_AIR_S) & on).any(dim=-1).float()


def _window(env) -> torch.Tensor:
    t = _t(env)
    return ((t >= T_START) & (t < T_END)).float()


def stomp_single_support(env) -> torch.Tensor:
    """Exactement un pied en l'air, dans la fenetre de trepignement."""
    _update(env)
    return _window(env) * ((~_feet_found(env)).sum(dim=-1) == 1).float()


def stomp_touchdown(env) -> torch.Tensor:
    """Repose d'un pied apres un vrai decollage (un pas de trepignement), dans la fenetre."""
    _update(env)
    return _window(env) * env._stomp_touch


def stomp_hold_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) quand un pied reste en l'air plus de HOLD_FREE_S : on trepigne, on ne pose pas
    le pied en l'air comme un flamant."""
    _update(env)
    return -torch.clamp((env._stomp_air.amax(dim=-1) - HOLD_FREE_S) / 0.3, 0.0, 1.0)


def stomp_both_off_penalty(env) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) si les DEUX pieds sont en l'air : un trepignement n'est pas un saut."""
    _update(env)
    return -((~_feet_found(env)).sum(dim=-1) == 2).float()


def _rest(env) -> torch.Tensor:
    return (_t(env) >= T_REST).float()


def stomp_rest_pose(env, std: float = 0.5,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return _rest(env) * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def stomp_rest_height(env, std: float = 0.01,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    return _rest(env) * torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def make_microduck_stomp_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_startle_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = STOMP_DURATION_S
    for name in ("startle_pitch", "startle_feet_penalty", "startle_rest_pose", "startle_rest_height",
                 "startle_rest_still", "startle_drift", "bow_roll_penalty"):
        cfg.rewards.pop(name, None)

    cfg.commands["twist"].resampling_time_range = (STOMP_DURATION_S, STOMP_DURATION_S * 2)

    cfg.rewards["stomp_single_support"] = RewardTermCfg(func=stomp_single_support, weight=4.0)
    cfg.rewards["stomp_touchdown"] = RewardTermCfg(func=stomp_touchdown, weight=3.0)
    cfg.rewards["stomp_hold_penalty"] = RewardTermCfg(func=stomp_hold_penalty, weight=4.0)
    cfg.rewards["stomp_both_off_penalty"] = RewardTermCfg(func=stomp_both_off_penalty, weight=4.0)
    cfg.rewards["stomp_lean_penalty"] = RewardTermCfg(func=shiver_lean_penalty, weight=2.0)
    cfg.rewards["stomp_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=1.5)
    cfg.rewards["stomp_yaw_penalty"] = RewardTermCfg(func=shiver_yaw_penalty, weight=2.0)
    cfg.rewards["stomp_rest_pose"] = RewardTermCfg(func=stomp_rest_pose, weight=3.0)
    cfg.rewards["stomp_rest_height"] = RewardTermCfg(func=stomp_rest_height, weight=3.0)
    return cfg


MicroduckStompRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_stomp",
    run_name="microduck_stomp",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
