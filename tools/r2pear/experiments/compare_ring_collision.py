"""Bounded CoACD/SDF comparison; does not modify production specs or services."""
import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import sim as S
import droid_common as C
import predicates as PR


def numbers(values):
    return ' '.join(f'{float(x):.12g}' for x in np.asarray(values).ravel())


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def add_ring(root, body, assets, info, method):
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    I = np.asarray(info['inertia_kg_m2'])
    ET.SubElement(body, 'inertial', mass=str(info['mass_kg']), pos=numbers(info['center_of_mass_m']),
                  fullinertia=numbers([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]]))
    ET.SubElement(asset, 'mesh', name='ring_original', file=str((assets/'visual.obj').resolve()))
    ET.SubElement(body, 'geom', name='ring_visual', type='mesh', mesh='ring_original',
                  contype='0', conaffinity='0', density='0', group='2', rgba='.04 .35 .8 1')
    common = dict(density='0', condim='4', friction='1 .005 .0001', solref='.01 1',
                  group='3', rgba='.1 .4 .9 1', contype='1', conaffinity='1')
    if method == 'coacd':
        for i, part in enumerate(info['parts']):
            name = f'ring_part_{i}'
            ET.SubElement(asset, 'mesh', name=name, file=str((assets/part['file']).resolve()))
            ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name, **common)
    elif method == 'sdf':
        extension = root.find('extension')
        if extension is None:
            extension = ET.SubElement(root, 'extension')
        instance = ET.SubElement(ET.SubElement(extension, 'plugin', plugin='mujoco.sdf.torus'), 'instance', name='ring_sdf')
        ET.SubElement(instance, 'config', key='radius1', value=str(info['torus_major_radius_m']))
        ET.SubElement(instance, 'config', key='radius2', value=str(info['torus_minor_radius_m']))
        mesh = ET.SubElement(asset, 'mesh', name='ring_sdf_surface')
        ET.SubElement(mesh, 'plugin', instance='ring_sdf')
        geom = ET.SubElement(body, 'geom', name='ring_sdf', type='sdf', mesh='ring_sdf_surface', **common)
        ET.SubElement(geom, 'plugin', instance='ring_sdf')
    else:
        raise ValueError(method)


def isolated_xml(assets, info, method, case, support='plane'):
    root = ET.Element('mujoco', model=f'ring_{method}_{case}')
    ET.SubElement(root, 'compiler', angle='radian')
    option = ET.SubElement(root, 'option', timestep='.002', integrator='implicitfast', cone='elliptic',
                           noslip_iterations='0', sdf_iterations='10', sdf_initpoints='40')
    ET.SubElement(option, 'flag', multiccd='enable')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -.3 .5', dir='0 .3 -.5')
    if case != 'probe':
        ET.SubElement(world, 'geom', name='floor', type=support, size='.3 .3 .01',
                      pos='0 0 -.01' if support=='box' else '0 0 0', rgba='.55 .55 .55 1', friction='1 .005 .0001')
    else:
        option.set('gravity', '0 0 0')
    z0 = max(info['source_height_m']/2,info['torus_minor_radius_m'])+.001
    z = .10 if case == 'probe' else (.12 if case == 'drop' else z0)
    ring = ET.SubElement(world, 'body', name='blue_ring', pos=numbers([0,0,z]))
    if case == 'drop':
        ring.set('euler', '.261799 0 0')
    if case != 'probe':
        ET.SubElement(ring, 'freejoint', name='blue_ring_free')
    add_ring(root, ring, assets, info, method)
    if case == 'probe':
        probe = ET.SubElement(world, 'body', name='probe', pos='0 0 .1')
        ET.SubElement(probe, 'freejoint', name='probe_free')
        ET.SubElement(probe, 'geom', name='probe', type='sphere', size='.004', mass='.01')
    if case == 'pinch':
        # A real slide DOF supplies vertical velocity to the friction solver.
        # Teleporting a mocap carrier moves geometry but is not a valid lift test.
        carrier = ET.SubElement(world, 'body', name='carrier', pos=numbers([0,0,z0]), gravcomp='1')
        ET.SubElement(carrier, 'inertial', mass='.1', pos='0 0 0', diaginertia='.0001 .0001 .0001')
        ET.SubElement(carrier, 'joint', name='lift', type='slide', axis='0 0 1', range='0 .05', damping='10')
        act = ET.SubElement(root, 'actuator')
        R, r = info['torus_major_radius_m'], info['torus_minor_radius_m']
        for sign,name in [(-1,'inner'),(1,'outer')]:
            pad = ET.SubElement(carrier, 'body', name=name, pos=numbers([R+sign*(r+.008),0,0]))
            ET.SubElement(pad, 'joint', name=name, type='slide', axis=numbers([-sign,0,0]), range='0 .02', damping='1')
            ET.SubElement(pad, 'geom', name=name, type='box', size='.003 .008 .01', mass='.05',
                          friction='1 .005 .0001', condim='4', solref='.01 1', rgba='.2 .2 .2 1')
            ET.SubElement(act, 'position', name=name, joint=name, kp='2000', kv='5',
                          ctrlrange='0 .02', forcerange='-20 20')
        ET.SubElement(act, 'position', name='lift', joint='lift', kp='5000', kv='100',
                      ctrlrange='0 .05', forcerange='-100 100')
    return root, z0


def contact_stats(model, data, body_name='blue_ring'):
    bid = model.body(body_name).id
    contacts = [c for c in data.contact if bid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])]
    return len(contacts), min([0.]+[float(c.dist) for c in contacts])


def compile_xml(root, path, prepare_mesh_sdf=False):
    ET.indent(root)
    xml = ET.tostring(root, encoding='unicode')
    path.write_text(xml)
    start = time.perf_counter()
    if prepare_mesh_sdf:
        spec=mujoco.MjSpec.from_string(xml)
        names={g.meshname for g in spec.geoms if g.type==mujoco.mjtGeom.mjGEOM_MESH and (g.contype or g.conaffinity)}
        for name in names:
            spec.mesh(name).needsdf=True
        model=spec.compile()
        dump(path.with_suffix('.mesh-sdf.json'),{'meshes_with_needsdf':sorted(names),'note':'needsdf is a Python-only compiler setting; rerun this script to reproduce.'})
        mujoco.mj_saveModel(model,str(path.with_suffix('.mjb')),None)
    else:
        model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model), time.perf_counter()-start


def probe(model, data, info):
    adr = int(model.joint('probe_free').qposadr[0])
    hole, wall = [], []
    for z in np.linspace(.06,.14,17):
        data.qpos[adr:adr+3] = [0,0,z]
        mujoco.mj_forward(model,data)
        hole.append(contact_stats(model,data)[0])
    radius = info['torus_major_radius_m']+.004
    for angle in np.linspace(0,2*np.pi,16,endpoint=False):
        data.qpos[adr:adr+3] = [radius*np.cos(angle),radius*np.sin(angle),.102]
        mujoco.mj_forward(model,data)
        wall.append(contact_stats(model,data)[0])
    return {'hole_probe_contacts':hole,'wall_probe_contacts':wall,
            'hole_clear':max(hole)==0,'all_wall_probes_detected':min(wall)>0}


def dynamics(model, data, case, z0, max_seconds):
    duration = 4.0 if case == 'drop' else 3.5
    body = model.body('blue_ring').id
    joint = model.joint('blue_ring_free')
    veladr=int(joint.dofadr[0])
    samples=[];poses=[];start=time.perf_counter(); step_seconds=0.; minimum=0.; max_contacts=0
    mujoco.mj_forward(model,data)
    for i in range(round(duration/model.opt.timestep)):
        t=i*model.opt.timestep
        if case == 'pinch':
            closing=min(t/.4,1.)*.008 if t<2 else max(1-(t-2)/.3,0.)*.008
            data.ctrl[model.actuator('inner').id]=closing
            data.ctrl[model.actuator('outer').id]=closing
            data.ctrl[model.actuator('lift').id]=.04*np.clip((t-.7)/.6,0,1)
        t0=time.perf_counter();mujoco.mj_step(model,data);step_seconds+=time.perf_counter()-t0
        count,pen=contact_stats(model,data);minimum=min(minimum,pen);max_contacts=max(max_contacts,count)
        if i%25==0:
            samples.append([float(data.time),*data.xpos[body],*data.qvel[veladr:veladr+6],count,pen])
            poses.append(data.qpos.copy())
        if not np.isfinite(data.qpos).all() or time.perf_counter()-start>max_seconds:
            break
    samples=np.asarray(samples)
    tail=samples[samples[:,0]>duration-.5]
    settled=bool(len(tail)>0 and np.max(np.linalg.norm(tail[:,4:7],axis=1))<.005
                 and np.max(np.linalg.norm(tail[:,7:10],axis=1))<.5)
    output={'complete':bool(data.time>=duration-.003),'sim_seconds':float(data.time),
            'wall_seconds':time.perf_counter()-start,'physics_step_seconds':step_seconds,
            'min_contact_dist_mm':minimum*1000,'max_contacts':max_contacts,
            'final_position':data.xpos[body].tolist(),'final_speed_mm_s':float(np.linalg.norm(data.qvel[veladr:veladr+3])*1000),
            'final_spin_rad_s':float(np.linalg.norm(data.qvel[veladr+3:veladr+6])), 'settled_last_half_second':settled,
            'post_settle_drift_mm':float(np.linalg.norm(samples[-1,1:4]-samples[np.argmin(abs(samples[:,0]-.5)),1:4])*1000),
            'max_lift_mm':float((samples[:,3].max()-z0)*1000)}
    if case=='pinch':
        hold=samples[(samples[:,0]>=1.5)&(samples[:,0]<2)]
        output['held_during_lift']=bool(len(hold)>0 and np.min(hold[:,3]-z0)>.025)
        output['released_and_returned']=bool(settled and data.xpos[body,2]<z0+.005)
        output['fixture_note']='Two force-limited sliding pads (20 N per pad), carrier lift 40 mm; not Panda trajectory validation.'
    return output,{'samples':samples,'qpos':np.asarray(poses)}


def full_scene(assets, info, method, scene):
    spec=json.loads((scene/'stage0_spec/task_spec.json').read_text())
    _,_,source=S.build(spec,return_spec=True)
    root=ET.fromstring(source.to_xml())
    root.find('compiler').attrib.pop('meshdir',None)
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file'):
            family='robotiq_2f85' if mesh.get('name','').startswith('g_') else 'franka_emika_panda'
            mesh.set('file',str(Path(C.MENAGERIE)/family/'assets'/mesh.get('file')))
    option=root.find('option');option.set('sdf_iterations','10');option.set('sdf_initpoints','40')
    flag=option.find('flag')
    if flag is None:flag=ET.SubElement(option,'flag')
    flag.set('multiccd','enable')
    body=root.find(".//body[@name='blue_ring']")
    for child in list(body):
        if child.tag in ('geom','inertial'):body.remove(child)
    pose=np.linalg.inv(np.array(spec['robot']['T_world_base']))@np.load(assets/'mesh.npz')['world_pose']
    quat=Rotation.from_matrix(pose[:3,:3]).as_quat()
    body.set('pos',numbers(pose[:3,3]));body.set('quat',numbers(quat[[3,0,1,2]]))
    add_ring(root,body,assets,info,method)
    return root,spec


def replay(model,data,spec,demo,max_seconds):
    start=time.perf_counter();step_seconds=0.;minimum=0.;max_contacts=0;trace=[];contact_rows=[]
    arm=S.arm_actuators(model);grip=S.grip_actuator(model);qa=C.arm_qadr(model)
    body=model.body('blue_ring').id
    S.reset(model,data,spec,settle_s=0.)
    initial=data.xpos[body].copy()
    failed=None
    def step():
        nonlocal step_seconds,minimum,max_contacts
        t0=time.perf_counter()
        try:
            mujoco.mj_step(model,data)
        except mujoco.FatalError as e:
            raise RuntimeError(f'MuJoCo error at sim_time={data.time:.6f}: {e}') from e
        step_seconds+=time.perf_counter()-t0
        count,pen=contact_stats(model,data);minimum=min(minimum,pen);max_contacts=max(max_contacts,count)
    J,G=demo['q'],demo['g_cmd'];sub=round(1/float(demo['fps'])/model.opt.timestep)
    track=[];lift=[];completed=0
    phase='initial_settle';frame=None;k=None
    try:
        for _ in range(250):step()
        phase='source_replay'
        for frame in range(len(J)-1):
            for k in range(sub):
                a=k/sub;data.ctrl[arm]=(1-a)*J[frame]+a*J[frame+1];data.ctrl[grip]=255*((1-a)*G[frame]+a*G[frame+1]);step()
            trace.append(data.qpos.copy());track.append(float(np.max(np.abs(data.qpos[qa]-J[frame+1]))));lift.append(float(data.xpos[body,2]-initial[2]));completed=frame+1
            pairs=[]
            for c in data.contact:
                if body in (model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]):
                    pairs.append([model.geom(c.geom1).name,model.geom(c.geom2).name,float(c.dist)])
            if frame%15==0:contact_rows.append({'source_frame':frame+2,'contacts':pairs})
            if not np.isfinite(data.qpos).all():raise RuntimeError('Nonfinite qpos')
            if time.perf_counter()-start>max_seconds:raise RuntimeError('Replay wall-time budget exceeded')
        phase='final_settle'
        for _ in range(500):step()
    except RuntimeError as e:
        failed={'message':str(e),'phase':phase,'source_interval_zero_based':frame,'substep':k}
    complete=failed is None and completed==len(J)-1
    ok,detail=PR.evaluate(model,data,spec) if failed is None else (False,[])
    return {'complete':complete,'completed_source_intervals':completed,'success':bool(ok) if complete else None,
            'error':failed,'predicates':detail,'wall_seconds':time.perf_counter()-start,'physics_step_seconds':step_seconds,
            'min_ring_contact_dist_mm':minimum*1000,'max_ring_contacts':max_contacts,
            'max_lift_mm':max(lift,default=0.)*1000,'final_ring_position':data.xpos[body].tolist(),
            'joint_track_max_deg':float(np.degrees(max(track,default=0.))), 'contact_samples':contact_rows,
            'scene_scope':'Only blue-ring collision replaced; other objects/tilts/static boxes retain the input-stage approximations.'},np.asarray(trace)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--method',choices=['coacd','sdf'],required=True)
    ap.add_argument('--scene',type=Path,default=Path('runs/test4-input'))
    ap.add_argument('--full-scene',action='store_true');ap.add_argument('--max-seconds',type=float,default=180)
    ap.add_argument('--cases',nargs='*',choices=['probe','rest','drop','pinch'],default=['probe','rest','drop','pinch'])
    ap.add_argument('--support',choices=['plane','box'],default='plane')
    ap.add_argument('--prepare-mesh-sdf',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    info=json.loads((args.assets/'collision-assets.json').read_text())
    results={'method':args.method,'mujoco_version':mujoco.__version__,'mass_kg':info['mass_kg'],
             'common_parameters':{'timestep':.002,'multiccd':True,'support':args.support,'friction':[1,.005,.0001],'solref':[.01,1],
                                  'sdf_initpoints':40,'sdf_iterations':10},'cases':{}}
    for case in args.cases:
        root,z0=isolated_xml(args.assets,info,args.method,case,args.support)
        model,data,seconds=compile_xml(root,args.out/(case+'.xml'))
        if case=='probe':result=probe(model,data,info)
        else:
            result,samples=dynamics(model,data,case,z0,args.max_seconds)
            np.savez_compressed(args.out/(case+'-trace.npz'),**samples)
        result['compile_seconds']=seconds;results['cases'][case]=result
        dump(args.out/'results.json',results);print(case,json.dumps(result),flush=True)
    if args.full_scene:
        root,spec=full_scene(args.assets,info,args.method,args.scene)
        model,data,seconds=compile_xml(root,args.out/'replay.xml',args.prepare_mesh_sdf)
        with np.load(args.scene/'stage0_spec/demo.npz') as demo:
            result,trace=replay(model,data,spec,demo,args.max_seconds)
        result['compile_seconds']=seconds;results['cases']['replay']=result
        np.savez_compressed(args.out/'replay-trace.npz',qpos=trace,final_qpos=data.qpos.copy())
        dump(args.out/'results.json',results);print('replay',json.dumps({k:v for k,v in result.items() if k!='contact_samples'}),flush=True)


if __name__=='__main__':main()
