#!/usr/bin/env python3
"""Diagnostic commun des gestes II (Tap, Crouch, Sidestep) : rejoue un checkpoint sur un episode et mesure hauteur du tronc
(min / max / fin), deplacement dans le repere de DEPART du tronc (avant, lateral), variation de lacet, inclinaison max,
pas par pied (reposes apres decollage), temps d'appui monopode / de vol, vitesse en fin d'episode.

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_gestes2.py <Tap|Crouch|Sidestep> <checkpoint.pt> [--envs 16]
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
    ap.add_argument("geste", choices=["Tap", "Crouch", "Sidestep", "LegStretch", "StopTheatre"])
    ap.add_argument("checkpoint")
    ap.add_argument("--envs", type=int, default=16)
    a = ap.parse_args()
    tache = f"Mjlab-{a.geste}-Flat-Backlash-MicroDuck"
    device = "cuda:0"
    env_cfg = load_env_cfg(tache, play=True)
    agent_cfg = load_rl_cfg(tache)
    env_cfg.scene.num_envs = a.envs
    env_cfg.events.pop("push_robot", None)
    env_cfg.curriculum.pop("push_magnitude", None)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(tache) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    brut = env.unwrapped
    robot = brut.scene["robot"]
    capteur = brut.scene.sensors["feet_ground_contact"]
    dt = brut.step_dt
    n_pas = int(round(env_cfg.episode_length_s / dt)) - 2   # le dernier pas remet l'episode a zero
    obs = env.get_observations()
    p0 = robot.data.root_link_pos_w[:, :2].clone()
    q0 = robot.data.root_link_quat_w.clone()
    yaw0 = torch.atan2(2 * (q0[:, 0] * q0[:, 3] + q0[:, 1] * q0[:, 2]), 1 - 2 * (q0[:, 2] ** 2 + q0[:, 3] ** 2))
    air = torch.zeros(a.envs, 2, device=device)
    reposes = torch.zeros(a.envs, 2, device=device)
    mono = torch.zeros(a.envs, device=device)
    vol = torch.zeros(a.envs, device=device)
    zmin = torch.full((a.envs,), 9.0, device=device)
    zmax = torch.zeros(a.envs, device=device)
    inc_max = torch.zeros(a.envs, device=device)
    trace_z, trace_y = [], []
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            on = capteur.data.found[:, :2].reshape(a.envs, 2, -1).sum(dim=-1) > 0
            prev = air.clone()
            air = torch.where(on, torch.zeros_like(prev), prev + dt)
            reposes += ((prev >= 0.08) & on).float()
            off = (~on).sum(dim=-1)
            mono += (off == 1).float()
            vol += (off == 2).float()
            z = robot.data.root_link_pos_w[:, 2]
            zmin, zmax = torch.minimum(zmin, z), torch.maximum(zmax, z)
            g = robot.data.projected_gravity_b
            inc_max = torch.maximum(inc_max, torch.acos((-g[:, 2]).clamp(-1, 1)))
            if k % 5 == 0:
                trace_z.append(f"{z[0].item() * 100:.1f}")
    d = robot.data.root_link_pos_w[:, :2] - p0
    avant = d[:, 0] * torch.cos(yaw0) + d[:, 1] * torch.sin(yaw0)
    lat = -d[:, 0] * torch.sin(yaw0) + d[:, 1] * torch.cos(yaw0)
    qf = robot.data.root_link_quat_w
    yawf = torch.atan2(2 * (qf[:, 0] * qf[:, 3] + qf[:, 1] * qf[:, 2]), 1 - 2 * (qf[:, 2] ** 2 + qf[:, 3] ** 2))
    dyaw = (yawf - yaw0 + math.pi) % (2 * math.pi) - math.pi
    vfin = robot.data.root_link_lin_vel_w.norm(dim=-1)
    print(f"[{a.geste}] env 0 : hauteur du tronc (cm) toutes les {5 * dt * 1000:.0f} ms :")
    print(" ".join(trace_z))
    print(f"hauteur min {zmin.mean().item() * 100:.1f} / max {zmax.mean().item() * 100:.1f} / fin {robot.data.root_link_pos_w[:, 2].mean().item() * 100:.1f} cm ; "
          f"deplacement avant {avant.mean().item() * 100:+.1f} cm, lateral (+ = gauche) {lat.mean().item() * 100:+.1f} cm ; "
          f"lacet {math.degrees(dyaw.mean().item()):+.0f} deg ; inclinaison max {math.degrees(inc_max.mean().item()):.0f} deg ; "
          f"pas pied0 {reposes[:, 0].mean().item():.1f} / pied1 {reposes[:, 1].mean().item():.1f} ; monopode {mono.mean().item():.0f} pas ; "
          f"vol {vol.mean().item():.1f} pas ; vitesse finale {vfin.mean().item():.2f} m/s ; sur {n_pas} pas")
    env.close()


if __name__ == "__main__":
    main()
