#!/usr/bin/env python3
"""Diagnostic de la politique Bow (salut marque) : rejoue un checkpoint et trace, au fil d un cycle, l angle de
tangage, les contacts des pieds, le roulis et la derive horizontale.

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_bow.py <checkpoint.pt> [--envs 16] [--sans-poussees]
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TACHE = "Mjlab-Bow-Flat-Backlash-MicroDuck"
PERIODE = 2.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--sans-poussees", action="store_true")
    a = ap.parse_args()
    device = "cuda:0"
    env_cfg = load_env_cfg(TACHE, play=True)
    agent_cfg = load_rl_cfg(TACHE)
    env_cfg.scene.num_envs = a.envs
    if a.sans_poussees:
        env_cfg.events.pop("push_robot", None)
        env_cfg.curriculum.pop("push_magnitude", None)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TACHE) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    brut = env.unwrapped
    robot = brut.scene["robot"]
    capteur = brut.scene.sensors["feet_ground_contact"]
    n_pas = int(PERIODE / brut.step_dt)                  # un cycle
    obs = env.get_observations()
    pied_off = torch.zeros(a.envs, device=device)
    x0 = robot.data.root_link_pos_w[:, :2].clone()
    tang_max = torch.full((a.envs,), -9.0, device=device)
    tang_fin = torch.zeros(a.envs, device=device)
    roulis_max = torch.zeros(a.envs, device=device)
    print(f"pas de {brut.step_dt * 1000:.0f} ms, {n_pas} pas par cycle ; env 0 detaille")
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            found = capteur.data.found[:, :2].reshape(a.envs, 2, -1).sum(dim=-1) > 0
            pied_off += (~found.all(dim=-1)).float()
            g = robot.data.projected_gravity_b   # x : sin(tangage, nez vers le bas > 0), y : roulis
            tang = torch.asin(g[:, 0].clamp(-1, 1))
            tang_max = torch.maximum(tang_max, tang)
            tang_fin = tang
            roulis_max = torch.maximum(roulis_max, g[:, 1].abs())
            cmd = brut.command_manager.get_command("twist")
            phi = (math.atan2(cmd[0, 1].item(), cmd[0, 0].item()) / (2 * math.pi)) % 1.0
            if k % 3 == 0:
                print(f"  phi={phi:4.2f}  tangage={math.degrees(tang[0].item()):+6.1f} deg  "
                      f"pieds_au_sol={"oui" if found[0].all() else "NON"}")
    dep = (robot.data.root_link_pos_w[:, :2] - x0).norm(dim=-1)
    print(f"tangage max moyen {math.degrees(tang_max.mean().item()):.1f} deg (cible 23) ; tangage final "
          f"{math.degrees(tang_fin.mean().item()):+.1f} deg (cible 0) ; roulis max {math.degrees(math.asin(min(roulis_max.mean().item(), 1))):.1f} deg ; "
          f"deplacement final {dep.mean().item() * 100:.1f} cm ; pas avec un pied en l air {pied_off.mean().item():.1f} sur {n_pas}")
    env.close()


if __name__ == "__main__":
    main()
