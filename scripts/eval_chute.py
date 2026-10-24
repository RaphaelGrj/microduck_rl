#!/usr/bin/env python3
"""Taux de chute d'une politique episodique : N envs, un episode, le tronc est dit tombe si son axe z s'ecarte de plus de
60 deg de la verticale a un instant. Sert a trancher « le rendu video montre une chute : artefact ou vrai defaut ? ».

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_chute.py <tache> <checkpoint.pt> [--envs 64] [--avec-poussees] [--hauteur-balle]
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tache")
    ap.add_argument("checkpoint")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--avec-poussees", action="store_true")
    ap.add_argument("--train-cfg", action="store_true", help="config d'entrainement (play=False) au lieu de play")
    a = ap.parse_args()
    device = "cuda:0"
    env_cfg = load_env_cfg(a.tache, play=not a.train_cfg)
    agent_cfg = load_rl_cfg(a.tache)
    env_cfg.scene.num_envs = a.envs
    if not a.avec_poussees:
        env_cfg.events.pop("push_robot", None)
        env_cfg.curriculum.pop("push_magnitude", None)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(a.tache) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    brut = env.unwrapped
    robot = brut.scene["robot"]
    n_pas = int(round(env_cfg.episode_length_s / brut.step_dt)) - 2
    obs = env.get_observations()
    tombe = torch.zeros(a.envs, dtype=torch.bool, device=device)
    t_chute = torch.full((a.envs,), -1.0, device=device)
    zmin = torch.full((a.envs,), 9.0, device=device)
    incmax = torch.zeros(a.envs, device=device)
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            inc = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
            zmin = torch.minimum(zmin, robot.data.root_link_pos_w[:, 2])
            incmax = torch.maximum(incmax, inc)
            nouv = (inc > math.radians(60)) & ~tombe
            t_chute[nouv] = k * brut.step_dt
            tombe |= nouv
    zfin = robot.data.root_link_pos_w[:, 2]
    incfin = inc
    n = int(tombe.sum().item())
    tc = t_chute[tombe]
    print(f"{a.tache} [{'train-cfg' if a.train_cfg else 'play-cfg'}, poussees={a.avec_poussees}] : "
          f"{n}/{a.envs} chutes ({100 * n / a.envs:.0f} %)"
          + (f", instant moyen de la chute {tc.mean().item():.2f} s" if n else "")
          + f" ; hauteur min du tronc {zmin.mean().item() * 100:.1f} cm ; inclinaison max {math.degrees(incmax.mean().item()):.0f} deg"
          + f" ; FIN : hauteur {zfin.mean().item() * 100:.1f} cm, inclinaison {math.degrees(incfin.mean().item()):.0f} deg"
          + f" ; episode {n_pas * brut.step_dt:.1f} s")
    env.close()


if __name__ == "__main__":
    main()
