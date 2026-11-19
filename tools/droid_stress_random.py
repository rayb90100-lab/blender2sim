import sys, json, os, time
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
stats = dict(nan=0, warn=0, min_pen_mm=0.0, max_obj_speed=0.0, episodes=0, blocks_off_table=0)
t0 = time.time()
for e in range(20):
    mujoco.mj_resetData(m, d); d.qpos[qa] = J0; d.ctrl[arm] = J0; mujoco.mj_forward(m, d)
    for _ in range(250): mujoco.mj_step(m, d)
    q = J0.copy()
    for k in range(60):                      # 60 random "rounds", each a random joint target, ~0.5 s
        q = np.clip(q + rng.normal(0, 0.35, 7), lo[:,0], lo[:,1])
        d.ctrl[arm] = q; d.ctrl[g] = rng.choice([0, 255])
        for _ in range(250):
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all(): stats['nan'] += 1; break
            if d.ncon: stats['min_pen_mm'] = min(stats['min_pen_mm'], min(d.contact[i].dist for i in range(d.ncon))*1000)
        if not np.isfinite(d.qpos).all(): break
        for n in scene['objects']:
            j = m.joint(f'{n}_free'); v = d.qvel[j.dofadr[0]:j.dofadr[0]+3]
            stats['max_obj_speed'] = max(stats['max_obj_speed'], float(np.linalg.norm(v)))
    for n in scene['objects']:
        z = d.qpos[m.joint(f'{n}_free').qposadr[0]+2]
        if z < scene['table_z'] - 0.05: stats['blocks_off_table'] += 1
    stats['episodes'] += 1
stats['warnings'] = {mujoco.mjtWarning(i).name: int(m_.number) for i, m_ in enumerate(d.warning) if m_.number}
stats['sim_seconds'] = 20*60*0.5; stats['wall_s'] = round(time.time()-t0, 1)
print(json.dumps(stats, indent=1))
