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

from mjlab_microduck.tasks.microduck_sprint_env_cfg import SPRINT_T as SPIN_T, SPRINT_VX

TACHE = "Mjlab-Sprint-Flat-Backlash-MicroDuck"


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
    n_pas = int(round(env_cfg.episode_length_s / dt)) - 2
    obs = env.get_observations()
    x0 = robot.data.root_link_pos_w[:, :2].clone()
    tilt = torch.zeros(a.envs, device=device)
    zmin = torch.full((a.envs,), 9.0, device=device)
    vmax = torch.zeros(a.envs, device=device)
    vsum = torch.zeros(a.envs, device=device)
    n_course = 0
    xfin = None
    trace = []
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            vx = robot.data.root_link_lin_vel_b[:, 0]
            if k * dt < SPIN_T:
                vmax = torch.maximum(vmax, vx); vsum += vx; n_course += 1
                xfin = robot.data.root_link_pos_w[:, :2].clone()
            g = robot.data.projected_gravity_b
            tilt = torch.maximum(tilt, g[:, :2].norm(dim=-1))
            zmin = torch.minimum(zmin, robot.data.root_link_pos_w[:, 2])
            if k % 6 == 0:
                trace.append(f"{vx[0].item():+.2f}")
    dep = (xfin - x0).norm(dim=-1)
    arret = (robot.data.root_link_pos_w[:, :2] - xfin).norm(dim=-1)
    vfin = robot.data.root_link_lin_vel_w.norm(dim=-1)
    print("env 0 : vitesse avant (m/s) toutes les 120 ms :")
    print(" ".join(trace))
    print(f"vitesse moyenne pendant la course {(vsum / max(n_course, 1)).mean().item():.2f} m/s (visee {SPRINT_VX}) ; pic {vmax.mean().item():.2f} ; "
          f"distance parcourue {dep.mean().item() * 100:.0f} cm ; distance de freinage {arret.mean().item() * 100:.0f} cm ; "
          f"vitesse en fin d'episode {vfin.mean().item():.2f} m/s (doit etre ~0) ; inclinaison max {math.degrees(math.asin(min(tilt.mean().item(), 1))):.1f} deg ; "
          f"hauteur min {zmin.mean().item() * 100:.1f} cm")
    env.close()


if __name__ == "__main__":
    main()
