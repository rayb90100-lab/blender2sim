"""Build test5 source-pose geometry without changing DROID motion or source blend."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'b2s')]
import sim as S
import spec as SP
import droid_common as C
from common import Extract, quat_wxyz, rot_of
from build_spec import intrinsics_px
from repair_scene_collision import replace


def main():
    source = Path('runs/test5-input')
    out = Path('runs/test5-preview')
    out.mkdir(exist_ok=True)
    (out/'stage0_spec').mkdir(exist_ok=True)
    spec = SP.load(source/'stage0_spec/task_spec.json')
    spec['task_id'] = 'test5_droid_s00000_r00008_source_pose'
    spec['physics']['multiccd'] = True
    ex = Extract(source/'stage0_extract/extract.json')
    # Use the authored overview for a third selectable view, not a fake source camera.
    name = next(n for n in ex.E['cameras'] if n.startswith('总览'))
    cam = ex.E['cameras'][name]
    pose = np.linalg.inv(spec['robot']['T_world_base']) @ ex.M0(name)
    K = intrinsics_px(cam, spec['render']['width'], spec['render']['height'])
    spec['cameras']['orbit'] = {'mount':'world','pos':pose[:3,3].tolist(), 'quat_wxyz':quat_wxyz(rot_of(pose)),
        'fovy_deg':float(np.degrees(2*np.arctan(K['height']/2/K['fy']))),'intrinsics':K,
        'blend_camera':name,'source_camera':None}
    spec['blend_binding']['cameras']['orbit'] = name
    spec['blend_binding']['robot_visual_source'] = 'mujoco'
    spec['blend_binding']['hide'] = ['腕部 · 随动回路线束']
    spec['blend_binding']['objects']['metal_plate']['visual_companions'] = ['金属盘 · 线柄坐标']
    demo = source/'stage0_spec/demo.npz'
    spec['web_playback'] = {'type':'source_demo', 'label':'回放原始 DROID 轨迹', 'demo_sha256':hashlib.sha256(demo.read_bytes()).hexdigest()}
    _,_,ms = S.build(spec, return_spec=True)
    xml = ET.fromstring(ms.to_xml())
    for mesh in xml.findall('./asset/mesh'):
        file = Path(mesh.get('file'))
        if not file.is_absolute():
            robot_xml = C.GRIPPER_XML if mesh.get('name', '').startswith('g_') else C.PANDA_XML
            mesh.set('file', str((Path(robot_xml).parent/'assets'/file).resolve()))
    xml.find('compiler').attrib.pop('meshdir', None)
    inverse = np.linalg.inv(spec['robot']['T_world_base'])
    report={}
    for n in spec['objects']:
        assets=Path('outputs/test5-collision')/n
        report[n]=replace(xml,n,assets,inverse)
        spec['objects'][n]['center']=report[n]['pose_base'][0][3],report[n]['pose_base'][1][3],report[n]['pose_base'][2][3]
        spec['objects'][n]['density']=json.loads((assets/'collision-assets.json').read_text())['density_kg_m3']
        spec['objects'][n]['collision_note']='Full source mesh CoACD convex parts, explicit source pose and estimated mass; see collision report.'
    ET.indent(xml)
    (out/'model.xml').write_text(ET.tostring(xml,encoding='unicode')+'\n')
    spec['model_xml']={'path':'../model.xml','sha256':hashlib.sha256((out/'model.xml').read_bytes()).hexdigest()}
    (out/'stage0_spec/task_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(demo,out/'stage0_spec/demo.npz')
    shutil.copy2(source/'registration-report.json',out/'registration-report.json')
    (out/'collision-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    S.build(SP.load(out/'stage0_spec/task_spec.json'))
    print(json.dumps({'scene':str(out),'objects':report},ensure_ascii=False))

if __name__=='__main__':main()
