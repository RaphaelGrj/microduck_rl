#!/usr/bin/env python3
"""Diagnostic de la politique Shake (secousse en lacet) : rejoue un checkpoint et trace, au fil d'un cycle, la vitesse
de lacet, l'angle de lacet, les contacts des pieds et la derive horizontale.

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_shake.py <checkpoint.pt> [--envs 16] [--sans-poussees]
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TACHE = "Mjlab-Shake-Flat-Backlash-MicroDuck"
PERIODE = 2.0


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
    wz_max = torch.zeros(a.envs, device=device)
    pied_off = torch.zeros(a.envs, device=device)
    x0 = robot.data.root_link_pos_w[:, :2].clone()
    yaw0 = None
    yaw_min = torch.full((a.envs,), 9.0, device=device)
    yaw_max = torch.full((a.envs,), -9.0, device=device)
    print(f"pas de {brut.step_dt * 1000:.0f} ms, {n_pas} pas par cycle ; env 0 detaille")
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            wz = robot.data.root_link_ang_vel_w[:, 2]
            found = capteur.data.found[:, :2].reshape(a.envs, 2, -1).sum(dim=-1) > 0
            pied_off += (~found.all(dim=-1)).float()
            wz_max = torch.maximum(wz_max, wz.abs())
            q = robot.data.root_link_quat_w   # w, x, y, z
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            if yaw0 is None:
                yaw0 = yaw.clone()
            d = torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))
            yaw_min = torch.minimum(yaw_min, d)
            yaw_max = torch.maximum(yaw_max, d)
            cmd = brut.command_manager.get_command("twist")
            phi = (math.atan2(cmd[0, 1].item(), cmd[0, 0].item()) / (2 * math.pi)) % 1.0
            if k % 3 == 0:
                print(f"  phi={phi:4.2f}  wz={wz[0].item():+5.2f} rad/s  lacet={math.degrees(d[0].item()):+6.1f} deg  "
                      f"pieds_au_sol={'oui' if found[0].all() else 'NON'}")
    dep = (robot.data.root_link_pos_w[:, :2] - x0).norm(dim=-1)
    print(f"|wz| max moyen {wz_max.mean().item():.2f} rad/s (cible 3.0) ; amplitude du lacet "
          f"{math.degrees((yaw_max - yaw_min).mean().item()):.1f} deg ; deplacement final {dep.mean().item() * 100:.1f} cm ; "
          f"pas avec un pied en l'air {pied_off.mean().item():.1f} sur {n_pas}")
    env.close()


if __name__ == "__main__":
    main()
