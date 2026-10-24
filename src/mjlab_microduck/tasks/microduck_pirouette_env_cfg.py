"""Microduck PIROUETTE task — un tour complet sur place, puis arret propre, tronc droit.

Meme famille que Startle / Shiver / Stomp : politique PUBLIABLE comme vrai skill episodique, mais avec un signal d'arret.
La politique n'a ni horloge ni cap : v1 (2026-10-24) tournait a ~3 rad/s jusqu'a la remise de main du demon et le canard
TOMBAIT au retour (2 essais sur 3 dans duck-sim). Remede : le skill est declare avec un `unwind` (le demon envoie le twist
`idle` pendant `unwind_s` avant de rendre la main), et la politique apprend que ce twist veut dire « freine » :

    t 0.0 - SPIN_T          commande = zeros (le skill tourne)        -> tour a SPIN_WZ rad/s, tronc droit
    t SPIN_T - SPIN_T+UNW   commande = [1, 0, 0] (drapeau d'arret)    -> freine, se redresse, immobile

Manifest vise : duration_s = SPIN_T, unwind_s = UNWIND_S, command.idle = [1, 0, 0].

Recompenses : vitesse de lacet payee jusqu'a SPIN_WZ pendant le tour (plafonnee, penalite au-dela), puis immobilite
(vitesses lineaire et angulaire) et pose debout pendant le retour ; tronc droit, hauteur debout, cou au neutre,
deplacement limite dans tout l'episode. Les penalites heritees sur la vitesse angulaire sont retirees.

Sentinelles : spin_rate (doit monter), spin_rest_still (doit monter en fin d'episode), spin_tilt_penalty, spin_height.
"""

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import bow_roll_penalty
from mjlab_microduck.tasks.microduck_hop_env_cfg import _LEG_JOINTS, STAND_Z, _trunk_z, hop_tilt_penalty
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
from mjlab_microduck.tasks.microduck_startle_env_cfg import _t, make_microduck_startle_env_cfg

SPIN_WZ = 3.0                          # rad/s
SPIN_T = 2.1                           # s : un tour a SPIN_WZ (2*pi/3) = duration_s du manifest
UNWIND_S = 1.0                         # s de retour = unwind_s du manifest
PIROUETTE_DURATION_S = SPIN_T + UNWIND_S
REST_FROM = SPIN_T + 0.4               # l'immobilite est exigee 0,4 s apres le drapeau d'arret


def _spinning(env) -> torch.Tensor:
    return (_t(env) < SPIN_T).float()


def spin_rate(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vitesse de lacet (axe z monde, antihoraire > 0) payee jusqu'a SPIN_WZ pendant le tour."""
    wz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2], nan=0.0)
    return _spinning(env) * torch.clamp(wz / SPIN_WZ, 0.0, 1.0)


def spin_overshoot_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalite (negative, poids POSITIF) au-dela de 1,3 x SPIN_WZ : un tour, pas une toupie."""
    wz = torch.nan_to_num(env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2], nan=0.0)
    return -torch.clamp((wz - 1.3 * SPIN_WZ) / SPIN_WZ, 0.0, 1.0)


def spin_height(env, std: float = 0.02,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> torch.Tensor:
    """Tronc a hauteur debout : on pirouette debout, pas accroupi."""
    return torch.exp(-((_trunk_z(env, asset_cfg) - STAND_Z) / std) ** 2)


def spin_rest_still(env, vel_std: float = 0.3,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Immobilite (vitesses angulaire ET lineaire) pendant la fin du retour : le canard a fini de freiner."""
    data = env.scene[asset_cfg.name].data
    w = torch.nan_to_num(data.root_link_ang_vel_w, nan=0.0).norm(dim=-1)
    v = torch.nan_to_num(data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    return (_t(env) >= REST_FROM).float() * torch.exp(-((w + v) / vel_std) ** 2)


def spin_rest_pose(env, std: float = 0.5,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Pose debout (jambes) en fin de retour."""
    asset = env.scene[asset_cfg.name]
    qpos = microduck_mdp._servo_joint_pos(env, asset)[:, _LEG_JOINTS]
    q0 = asset.data.default_joint_pos[:, microduck_mdp._servo_joint_ids(env, asset)][:, _LEG_JOINTS]
    return (_t(env) >= REST_FROM).float() * torch.exp(-((qpos - q0) / std) ** 2).mean(dim=-1)


def make_microduck_pirouette_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_startle_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = PIROUETTE_DURATION_S
    for name in [n for n in cfg.rewards if n.startswith("startle_") and n not in
                 ("startle_displacement_penalty", "startle_neck_home")]:
        cfg.rewards.pop(name)
    for name in ("bow_roll_penalty", "body_ang_vel", "angular_momentum"):
        cfg.rewards.pop(name, None)
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["startle_displacement_penalty"].weight = 2.0
    cfg.rewards["startle_displacement_penalty"].params = {"full_m": 0.30}

    # Commande a deux etats : zeros pendant le tour, puis drapeau d'arret [1, 0, 0] (voir SpinStopCommand)
    base = make_velocity_env_cfg().commands["twist"]
    base.rel_standing_envs = 0.0
    base.rel_heading_envs = 0.0
    base.heading_command = False
    base.ranges.heading = None
    base.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SpinStopCommandCfg(
        **{**vars(base), "class_type": microduck_mdp.SpinStopCommand, "period": SPIN_T, "randomize_phase": False}
    )

    cfg.rewards["spin_rate"] = RewardTermCfg(func=spin_rate, weight=6.0)
    cfg.rewards["spin_overshoot_penalty"] = RewardTermCfg(func=spin_overshoot_penalty, weight=3.0)
    cfg.rewards["spin_tilt_penalty"] = RewardTermCfg(func=hop_tilt_penalty, weight=5.0)
    cfg.rewards["spin_roll_penalty"] = RewardTermCfg(func=bow_roll_penalty, weight=1.5)
    cfg.rewards["spin_height"] = RewardTermCfg(func=spin_height, weight=2.0)
    cfg.rewards["spin_rest_still"] = RewardTermCfg(func=spin_rest_still, weight=6.0)
    cfg.rewards["spin_rest_pose"] = RewardTermCfg(func=spin_rest_pose, weight=2.0)
    return cfg


MicroduckPirouetteRlCfg = RslRlOnPolicyRunnerCfg(
    actor=MicroduckSitStandRlCfg.actor,
    critic=MicroduckSitStandRlCfg.critic,
    algorithm=MicroduckSitStandRlCfg.algorithm,
    wandb_project="mjlab_microduck",
    experiment_name="microduck_pirouette",
    run_name="microduck_pirouette",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
