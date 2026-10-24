#!/usr/bin/env python3
"""Diagnostic de la politique Stomp (trepignement) : rejoue un checkpoint et mesure, sur un episode, le nombre de pas
(reposes apres decollage), la part d'appui monopode, le plus long maintien d'un pied en l'air, le tangage/roulis max,
le lacet et le deplacement.

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_stomp.py <checkpoint.pt> [--envs 16]
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TACHE = "Mjlab-Stomp-Flat-Backlash-MicroDuck"


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
    dt = brut.step_dt
    n_pas = int(round(env_cfg.episode_length_s / dt)) - 2   # le dernier pas remet l'episode a zero
    obs = env.get_observations()
    x0 = robot.data.root_link_pos_w[:, :2].clone()
    air = torch.zeros(a.envs, 2, device=device)
    reposes = torch.zeros(a.envs, device=device)
    mono = torch.zeros(a.envs, device=device)
    deux = torch.zeros(a.envs, device=device)
    maintien = torch.zeros(a.envs, device=device)
    tang = torch.zeros(a.envs, device=device)
    roulis = torch.zeros(a.envs, device=device)
    trace = []
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            on = capteur.data.found[:, :2].reshape(a.envs, 2, -1).sum(dim=-1) > 0
            prev = air.clone()
            air = torch.where(on, torch.zeros_like(prev), prev + dt)
            reposes += ((prev >= 0.08) & on).any(dim=-1).float()
            off = (~on).sum(dim=-1)
            mono += (off == 1).float()
            deux += (off == 2).float()
            maintien = torch.maximum(maintien, air.amax(dim=-1))
            g = robot.data.projected_gravity_b
            tang = torch.maximum(tang, g[:, 0].abs())
            roulis = torch.maximum(roulis, g[:, 1].abs())
            trace.append("".join("-" if on[0, i] else "^" for i in range(2)))
    dep = (robot.data.root_link_pos_w[:, :2] - x0).norm(dim=-1)
    print("env 0 : pieds par pas de 20 ms (G D ; ^ = en l'air) :")
    print(" ".join(trace))
    print(f"pas (reposes) {reposes.mean().item():.1f} en {n_pas * dt:.1f} s ; appui monopode {mono.mean().item():.0f} pas sur {n_pas} ; "
          f"deux pieds en l'air {deux.mean().item():.1f} ; maintien max {maintien.mean().item():.2f} s ; "
          f"tangage max {math.degrees(math.asin(min(tang.mean().item(), 1))):.1f} deg ; roulis max "
          f"{math.degrees(math.asin(min(roulis.mean().item(), 1))):.1f} deg ; deplacement {dep.mean().item() * 100:.1f} cm")
    env.close()


if __name__ == "__main__":
    main()
