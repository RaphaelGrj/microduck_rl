#!/usr/bin/env python3
"""Rend une video propre d'une politique de geste episodique (Sursaut) : 1 canard, camera fixe de trois quarts,
sans voisins ni poussees, N episodes a la suite (le canard repart debout au meme endroit), encodage mp4.

Usage (dans ~/microduck_rl) :
  MUJOCO_GL=egl uv run python scripts/render_shiver.py <checkpoint.pt> <sortie.mp4> [--episodes 3] [--fps 25]
--fps 25 avec un pas de 50 Hz = ralenti x0,5 (plus lisible pour un geste de 1,2 s) ; --fps 50 = temps reel.
"""
import argparse
from dataclasses import asdict

import imageio.v2 as imageio
import numpy as np
import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

TACHE = "Mjlab-Shiver-Flat-Backlash-MicroDuck"


def _ciel(img):
    """Remplace le ciel noir (pixels exactement nuls au-dessus de l horizon) par un degrade clair."""
    h = img.shape[0]
    noir = (img.sum(axis=-1) == 0)
    noir[int(h * 0.7):] = False
    t = (np.arange(h)[:, None] / (h * 0.35)).clip(0, 1)
    haut, bas = np.array([226, 232, 240]), np.array([186, 200, 216])
    deg = (haut * (1 - t[..., None]) + bas * t[..., None]).astype(np.uint8)
    out = img.copy()
    out[noir] = np.broadcast_to(deg, img.shape)[noir]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("sortie")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--largeur", type=int, default=1280)
    ap.add_argument("--hauteur", type=int, default=720)
    ap.add_argument("--azimut", type=float, default=70.0)
    a = ap.parse_args()
    device = "cuda:0"
    env_cfg = load_env_cfg(TACHE, play=True)
    agent_cfg = load_rl_cfg(TACHE)
    env_cfg.scene.num_envs = 1
    env_cfg.events.pop("push_robot", None)
    env_cfg.curriculum.pop("push_magnitude", None)
    v = env_cfg.viewer
    v.width, v.height = a.largeur, a.hauteur
    v.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    v.entity_name = "robot"
    v.lookat = (0.0, 0.0, 0.05)
    v.distance = 0.55
    v.elevation = -8.0
    v.azimuth = a.azimut
    v.enable_shadows = False
    v.enable_reflections = False
    v.max_extra_envs = 0
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array"), clip_actions=agent_cfg.clip_actions
    )
    runner = (load_runner_cls(TACHE) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    brut = env.unwrapped
    n_pas = int(round(env_cfg.episode_length_s / brut.step_dt))
    obs = env.get_observations()
    frames = []
    with torch.inference_mode():
        for _ in range(a.episodes):
            for _k in range(n_pas):
                obs, _, _, _ = env.step(policy(obs))
                frames.append(_ciel(np.asarray(brut.render())))
    imageio.mimwrite(a.sortie, frames, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"{len(frames)} images -> {a.sortie}")
    env.close()


if __name__ == "__main__":
    main()
