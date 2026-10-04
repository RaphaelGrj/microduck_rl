#!/usr/bin/env python3
"""Evaluation chiffree de la politique StandUp (se relever) : pour chaque type de position de depart (sur le ventre,
sur le dos, assis, debout), N canards simules, un episode complet ; on compte ceux qui finissent DEBOUT et le temps mis
pour se relever. Avec ou sans poussees (le mode play en met toutes les 0,5-1 s).

Debout = gravite projetee z < -0,9 (tronc a moins de ~25 deg de la verticale) et tronc a plus de 9 cm au-dessus de son
origine de terrain (debout : ~11,5 cm ; assis : ~6 cm), tenu jusqu'a la fin de l'episode.

Usage (dans ~/microduck_rl) :
  uv run python scripts/eval_standup.py <checkpoint.pt> [--envs 256] [--sans-poussees]
"""
import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registre)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TACHE = "Mjlab-StandUp-Rough-Backlash-MicroDuck"
import os
TRACE = bool(os.environ.get("TRACE"))
DEPARTS = {"ventre": "face_down_prob", "dos": "face_up_prob", "assis": "sitting_prob", "debout": "standing_prob"}


def evaluer(checkpoint, depart, n_envs, poussees, device="cuda:0"):
    env_cfg = load_env_cfg(TACHE, play=True)
    agent_cfg = load_rl_cfg(TACHE)
    env_cfg.scene.num_envs = n_envs
    env_cfg.curriculum.pop("ground_state_mix", None)        # sinon le curriculum reecrit le melange de departs
    p = env_cfg.events["set_ground_state"].params
    for cle in DEPARTS.values():
        p[cle] = 0.0
    p[DEPARTS[depart]] = 1.0
    if not poussees:
        env_cfg.events.pop("push_robot", None)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TACHE) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    brut = env.unwrapped
    robot = brut.scene["robot"]
    n_pas = int(brut.max_episode_length) - 2                 # on s'arrete juste avant la fin d'episode (reset)
    obs = env.get_observations()
    debout_depuis = torch.full((n_envs,), -1, device=device, dtype=torch.long)
    with torch.inference_mode():
        for k in range(n_pas):
            obs, _, _, _ = env.step(policy(obs))
            g = robot.data.projected_gravity_b[:, 2]
            h = robot.data.root_link_pos_w[:, 2] - brut.scene.env_origins[:, 2]
            debout = (g < -0.9) & (h > 0.09)
            if TRACE and k in (0, 2, 5, 10, 20, 40, 80, 150, n_pas - 1):
                print(f"    [{depart}] pas {k:3d} ({k * brut.step_dt:4.2f} s) : gravite z moyenne {g.mean().item():+.2f}, "
                      f"hauteur moyenne {h.mean().item():.3f} m, debout {debout.float().mean().item() * 100:.0f} %", flush=True)
            debout_depuis = torch.where(debout, torch.where(debout_depuis < 0, torch.full_like(debout_depuis, k),
                                                            debout_depuis), torch.full_like(debout_depuis, -1))
    reussi = debout_depuis >= 0
    dt = brut.step_dt
    t_releve = (debout_depuis[reussi].float() * dt)
    env.close()
    return reussi.float().mean().item(), (t_releve.median().item() if reussi.any() else float("nan")), n_pas * dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--sans-poussees", action="store_true")
    a = ap.parse_args()
    print(f"politique : {a.checkpoint}\n{a.envs} canards par type de depart, poussees : {'non' if a.sans_poussees else 'oui'}")
    for depart in DEPARTS:
        taux, t, duree = evaluer(a.checkpoint, depart, a.envs, not a.sans_poussees)
        print(f"  depart {depart:7s}: {100 * taux:5.1f} % debout a la fin ({duree:.1f} s) ; "
              f"releve en {t:.2f} s (mediane, depuis le debut de l'episode)", flush=True)


if __name__ == "__main__":
    main()
