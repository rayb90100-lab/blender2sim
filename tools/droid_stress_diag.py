import sys, json, collections
sys.path.insert(0, 'tools')
import numpy as np, mujoco
import droid_common as C, droid_stage2_physics as P
T = 'Real2Gym/tests/test4_droid_panda_blockstack'
ep = C.load_episode(f'{T}/input/episode.parquet')
K = json.load(open(f'{T}/stage0_replay/intrinsics_adopted.json'))
m, d, scene = P.build(T, ep, K['fx'])
qa = C.arm_qadr(m); arm = [m.actuator(f'actuator{i}').id for i in range(1,8)]; g = m.actuator('g_fingers_actuator').id
lo = np.array([m.jnt_range[m.joint(f'joint{i}').id] for i in range(1,8)])
J0 = ep['observation.state.joint_position'][0]
rng = np.random.default_rng(0)
bname = lambda gid: m.body(m.geom_bodyid[gid]).name
pen = collections.Counter(); worst = {}
fast = collections.Counter()
for e in range(20):
    mujoco.mj_resetData(m, d); d.qpos[qa] = J0; d.ctrl[arm] = J0; mujoco.mj_forward(m, d)
    for _ in range(250): mujoco.mj_step(m, d)
    q = J0.copy()
    for k in range(60):
        q = np.clip(q + rng.normal(0, 0.35, 7), lo[:,0], lo[:,1])
        d.ctrl[arm] = q; d.ctrl[g] = rng.choice([0, 255])
        for _ in range(250):
            mujoco.mj_step(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist < -0.005:
                    key = tuple(sorted((bname(c.geom1).replace('g_',''), bname(c.geom2).replace('g_',''))))
                    pen[key] += 1; worst[key] = min(worst.get(key, 0), c.dist*1000)
            for n in scene['objects']:
                j = m.joint(f'{n}_free'); v = np.linalg.norm(d.qvel[j.dofadr[0]:j.dofadr[0]+3])
                if v > 5: fast[n] += 1
print("contacts deeper than 5 mm (steps, worst mm):")
for k, v in pen.most_common(12): print(f"  {k}: {v} steps, worst {worst[k]:.1f}")
print("object >5 m/s steps:", dict(fast))
