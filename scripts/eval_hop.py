#!/usr/bin/env python3
"""Diagnostic de la politique Hop : rejoue un checkpoint et trace, au fil d'un cycle, la hauteur du tronc, les contacts
des deux pieds, la phase et l'inclinaison. Sert a verifier qu'un bond est physique (pieds en l'air + tronc qui monte).

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_hop.py <checkpoint.pt> [--envs 16]
"""
import argparse
import math
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TACHE = "Mjlab-Hop-Flat-Backlash-MicroDuck"


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
    n_pas = int(2.4 / brut.step_dt)                      # un cycle
    obs = env.get_observations()
    z_max = torch.zeros(a.envs, device=device)
    air_pas = torch.zeros(a.envs, device=device)
    print(f"pas de {brut.step_dt * 1000:.0f} ms, {n_pas} pas par cycle ; env 0 detaille, autres en statistiques")
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            z = robot.data.root_link_pos_w[:, 2] - brut.scene.env_origins[:, 2]
            found = capteur.data.found[:, :2].reshape(a.envs, -1)
            deux_off = (found.sum(dim=-1) == 0)
            z_max = torch.maximum(z_max, z)
            air_pas += deux_off.float()
            cmd = brut.command_manager.get_command("twist")
            phi = (math.atan2(cmd[0, 1].item(), cmd[0, 0].item()) / (2 * math.pi)) % 1.0
            if k % 5 == 0:
                g = robot.data.projected_gravity_b[0, 2].item()
                print(f"  phi={phi:4.2f}  z={z[0].item():.3f} m  contacts={found[0].tolist()}  "
                      f"pieds_en_l_air={'OUI' if deux_off[0] else 'non'}  gravite_z={g:+.2f}")
    print(f"z max moyen {z_max.mean().item():.3f} m (debout = 0,115) ; pas avec les deux pieds en l'air : "
          f"{air_pas.mean().item():.0f} sur {n_pas} en moyenne")
    env.close()


if __name__ == "__main__":
    main()
