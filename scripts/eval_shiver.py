#!/usr/bin/env python3
"""Diagnostic de la politique Shiver (frisson) : rejoue un checkpoint et mesure, sur un episode, l'amplitude et la
frequence du roulis, le tangage, le lacet, le deplacement et les pieds.

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_shiver.py <checkpoint.pt> [--envs 16]
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TACHE = "Mjlab-Shiver-Flat-Backlash-MicroDuck"


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
    capteur = brut.scene.sensors["feet_ground_contact"]
    n_pas = int(round(env_cfg.episode_length_s / brut.step_dt)) - 2   # le dernier pas remet l'episode a zero
    obs = env.get_observations()
    x0 = robot.data.root_link_pos_w[:, :2].clone()
    roulis, tang, lacet0 = [], [], None
    pied_off = torch.zeros(a.envs, device=device)
    wx_hist = []
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            g = robot.data.projected_gravity_b
            roulis.append(torch.asin(g[:, 1].clamp(-1, 1)))
            tang.append(torch.asin(g[:, 0].clamp(-1, 1)))
            wx_hist.append(robot.data.root_link_ang_vel_b[:, 0])
            found = capteur.data.found[:, :2].reshape(a.envs, 2, -1).sum(dim=-1) > 0
            pied_off += (~found.all(dim=-1)).float()
    roulis = torch.stack(roulis)             # [pas, envs]
    wx = torch.stack(wx_hist)
    dt = brut.step_dt
    # frequence : passages par zero du roulis (une oscillation = 2 passages), pendant la fenetre de tremblement
    d0, d1 = int(0.10 / dt), int(1.30 / dt)
    r = roulis[d0:d1]
    passages = ((r[1:] * r[:-1]) < 0).float().sum(dim=0)
    freq = passages / 2.0 / ((d1 - d0) * dt)
    if True:
        print("env 0 : roulis (deg) toutes les 40 ms")
        print(" ".join(f"{math.degrees(roulis[k, 0].item()):+.1f}" for k in range(0, n_pas, 2)))
    dep = (robot.data.root_link_pos_w[:, :2] - x0).norm(dim=-1)
    print(f"amplitude de roulis (crete) {math.degrees(r.abs().amax(dim=0).mean().item()):.1f} deg ; frequence "
          f"{freq.mean().item():.1f} Hz ; |vitesse roulis| moyenne {wx[d0:d1].abs().mean().item():.2f} rad/s ; "
          f"tangage max {math.degrees(torch.stack(tang).abs().amax(dim=0).mean().item()):.1f} deg ; "
          f"deplacement {dep.mean().item() * 100:.1f} cm ; pas avec un pied en l air {pied_off.mean().item():.1f} sur {n_pas}")
    env.close()


if __name__ == "__main__":
    main()
