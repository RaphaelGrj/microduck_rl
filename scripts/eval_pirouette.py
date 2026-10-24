#!/usr/bin/env python3
"""Diagnostic de la politique Pirouette : rejoue un checkpoint et mesure, sur un episode (= un tour vise), l'angle de lacet
total tourne, la vitesse moyenne et residuelle, l'inclinaison max, la hauteur et le deplacement.

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_pirouette.py <checkpoint.pt> [--envs 16]
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_microduck.tasks.microduck_pirouette_env_cfg import SPIN_T

TACHE = "Mjlab-Pirouette-Flat-Backlash-MicroDuck"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--envs", type=int, default=16)
    a = ap.parse_args()
    device = "cuda:0"
    env_cfg = load_env_cfg(TACHE, play=True)
    agent_cfg = load_rl_cfg(TACHE)
    env_cfg.scene.num_envs = a.envs
    env_cfg.events.pop("push_robot", None)
    env_cfg.curriculum.pop("push_magnitude", None)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TACHE) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    brut = env.unwrapped
    robot = brut.scene["robot"]
    dt = brut.step_dt
    n_pas = int(round(env_cfg.episode_length_s / dt)) - 2   # le dernier pas remet l'episode a zero
    obs = env.get_observations()
    x0 = robot.data.root_link_pos_w[:, :2].clone()
    angle = torch.zeros(a.envs, device=device)
    tilt = torch.zeros(a.envs, device=device)
    zmin = torch.full((a.envs,), 9.0, device=device)
    trace = []
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            wz = robot.data.root_link_ang_vel_w[:, 2]
            if k * dt < SPIN_T:
                angle += wz * dt
                wz_fin_tour = wz.clone()
            g = robot.data.projected_gravity_b
            tilt = torch.maximum(tilt, g[:, :2].norm(dim=-1))
            zmin = torch.minimum(zmin, robot.data.root_link_pos_w[:, 2])
            if k % 6 == 0:
                trace.append(f"{wz[0].item():+.1f}")
    dep = (robot.data.root_link_pos_w[:, :2] - x0).norm(dim=-1)
    print("env 0 : vitesse de lacet (rad/s) toutes les 120 ms :")
    print(" ".join(trace))
    print(f"angle tourne pendant le tour {math.degrees(angle.mean().item()):.0f} deg (vise 360) [min {math.degrees(angle.min().item()):.0f}, "
          f"max {math.degrees(angle.max().item()):.0f}] ; vitesse a la fin du tour {wz_fin_tour.mean().item():+.2f} rad/s ; "
          f"vitesse en fin d'episode {wz.abs().mean().item():.2f} rad/s (doit etre ~0) ; inclinaison max {math.degrees(math.asin(min(tilt.mean().item(), 1))):.1f} deg ; "
          f"hauteur min {zmin.mean().item() * 100:.1f} cm ; deplacement {dep.mean().item() * 100:.1f} cm")
    env.close()


if __name__ == "__main__":
    main()
