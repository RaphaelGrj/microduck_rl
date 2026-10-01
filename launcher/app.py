#!/usr/bin/env python3
"""Serveur local du launcher Microduck : sert l'interface et pilote infer_policy.py."""
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
POLICIES_DIR = REPO_DIR / "policies" / "official"
STATIC_DIR = Path(__file__).resolve().parent
PORT = 8787

SKILLS = [
    {"id": "walking", "flag": "--walking", "file": "alpha_walking.onnx",
     "label": "Marche", "desc": "Déplacement omnidirectionnel (avant/arrière/strafe/rotation)",
     "required": True},
    {"id": "standing", "flag": "--standing", "file": "alpha_stand.onnx",
     "label": "Debout", "desc": "Maintien en équilibre statique", "required": True},
    {"id": "sitstand", "flag": "--sitstand", "file": "alpha_sitstand.onnx",
     "label": "Assis / Debout", "desc": "S'assoit et se relève (touche Y)", "required": False},
    {"id": "ground_pick", "flag": "--ground-pick", "file": "alpha_ground_pick.onnx",
     "label": "Ramassage au sol", "desc": "Se penche pour ramasser un objet (touche G)", "required": False},
    {"id": "roulade", "flag": "--roulade", "file": "roulade.onnx",
     "label": "Roulade", "desc": "Roulade avant (touche R)", "required": False},
    {"id": "kick_left", "flag": "--kick-left", "file": "ball_kick_left.onnx",
     "label": "Tir pied gauche", "desc": "Shoot dans un ballon, pied gauche (touche K)", "required": False},
    {"id": "kick_right", "flag": "--kick-right", "file": "ball_kick_right.onnx",
     "label": "Tir pied droit", "desc": "Shoot dans un ballon, pied droit (touche L)", "required": False},
]

STATE = {"proc_pid": None, "command": None, "started_at": None}


def available_skills():
    out = []
    for s in SKILLS:
        path = POLICIES_DIR / s["file"]
        out.append({**s, "available": path.exists()})
    return out


def is_running():
    pid = STATE["proc_pid"]
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        STATE["proc_pid"] = None
        return False


def stop_sim():
    subprocess.run(["pkill", "-f", "infer_policy.py"], check=False)
    STATE["proc_pid"] = None
    STATE["command"] = None


def launch_sim(selected_ids):
    stop_sim()
    time.sleep(0.5)

    args = ["uv", "run", "scripts/infer_policy.py"]
    needs_new_obs = False
    for s in SKILLS:
        if s["id"] in selected_ids or s["required"]:
            path = POLICIES_DIR / s["file"]
            if not path.exists():
                continue
            args += [s["flag"], str(path.relative_to(REPO_DIR))]
            if s["id"] not in ("walking", "standing"):
                needs_new_obs = True
    # Les fichiers officiels actuels sont tous au format 61D (voir manifest.json) :
    # --new-cmd-obs est requis dès qu'une seule politique non-walking/standing est chargée,
    # et empiriquement aussi nécessaire pour walking/standing seuls avec ces fichiers.
    args.append("--new-cmd-obs")

    inner_cmd = (
        f"cd {REPO_DIR} && source \"$HOME/.local/bin/env\" && "
        + " ".join(args)
        + "; echo; echo '--- Simulation terminee. Appuie sur Entree pour fermer cette fenetre. ---'; read _"
    )
    # Ouvre une VRAIE fenêtre console Windows (TTY requis par infer_policy.py pour le clavier) via l'interop WSL->Windows.
    win_cmd = [
        "cmd.exe", "/c", "start", "Microduck - Simulation",
        "wsl.exe", "-d", "Ubuntu", "--", "bash", "-c", inner_cmd,
    ]
    proc = subprocess.Popen(win_cmd)
    proc.wait()  # `cmd /c start` retourne immédiatement une fois la fenêtre ouverte

    STATE["command"] = " ".join(args)
    STATE["started_at"] = time.time()
    # Le process réel tourne dans la fenêtre ouverte, pas comme enfant direct : on le retrouve par nom.
    time.sleep(1.5)
    pid = find_infer_pid()
    STATE["proc_pid"] = pid
    return STATE["command"]


def find_infer_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "infer_policy.py"], text=True)
        pids = [int(p) for p in out.split()]
        return pids[-1] if pids else None
    except subprocess.CalledProcessError:
        return None


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/api/skills":
            self._json({"skills": available_skills()})
        elif self.path == "/api/status":
            self._json({"running": is_running(), "command": STATE["command"]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/launch":
            try:
                cmd = launch_sim(body.get("skills", []))
                self._json({"ok": True, "command": cmd})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
        elif self.path == "/api/stop":
            stop_sim()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        pass  # silence les logs d'accès


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Microduck launcher sur http://localhost:{PORT}")
    server.serve_forever()
