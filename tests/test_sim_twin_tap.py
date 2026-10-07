"""Fork additions to the duck-sim body server used by the AR twin (microduck-brain « Jumeau »): every part's pose in
the ground-truth file, and the `throw` control command. CPU only."""
import json

import mujoco
import numpy as np

from mjlab_microduck.sim import body_server as bs

SCENE = bs.SCENES / "scene_arena_testball.xml"


def _world(count=1):
    world = bs.World(SCENE, count)
    for index in range(count):
        body = bs.Body(world, index)
        body.place(None, 0.125, offset_y=index * bs.SPACING)
        world.bodies.append(body)
    mujoco.mj_forward(world.model, world.data)
    return world


def test_groundtruth_has_every_part(tmp_path):
    world = _world(2)
    out = tmp_path / "gt.json"
    bs.write_groundtruth(world, str(out), bs.groundtruth_targets(world))
    gt = json.loads(out.read_text())
    assert gt["scene"] == "scene_arena_testball.xml"
    for duck in gt["ducks"]:
        parts = duck["parts"]
        assert len(parts) == 15 and "trunk_base" in parts and "jaw_soft" in parts
        assert all(len(v) == 7 for v in parts.values())
        # unprefixed names for every duck, and the trunk part sits where the trunk joint says
        assert np.allclose(parts["trunk_base"][:3], duck["pos"], atol=1e-4)
    assert gt["ducks"][1]["parts"]["trunk_base"][1] > 0.4          # the second duck, half a metre to the left


def test_throw_sets_position_and_capped_velocity(tmp_path):
    world = _world()
    ctl = tmp_path / "ctl.json"
    ctl.write_text(json.dumps({"throw": {"testball": {"pos": [0.5, 0.2, 0.3], "vel": [20.0, -1.0, 0.5]}}}))
    bs.apply_control(world, str(ctl))
    assert not ctl.exists()
    m, d = world.model, world.data
    j = int(m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "testball")])
    assert np.allclose(d.qpos[m.jnt_qposadr[j]: m.jnt_qposadr[j] + 3], [0.5, 0.2, 0.3])
    assert np.allclose(d.qvel[m.jnt_dofadr[j]: m.jnt_dofadr[j] + 3], [6.0, -1.0, 0.5])
