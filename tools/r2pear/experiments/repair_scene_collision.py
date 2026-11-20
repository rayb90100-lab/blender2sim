"""Replace named collision proxies with source-pose convex parts; never settle/bake poses.

Inputs are directories produced by extract_collision_mesh + prepare_ring_collision.
This is an explicit geometry repair, not a trajectory or contact-filter adjustment.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def numbers(values):
    return ' '.join(f'{float(x):.12g}' for x in np.asarray(values).ravel())


def set_pose(body, pose):
    for key in ('euler', 'axisangle', 'xyaxes', 'zaxis'):
        body.attrib.pop(key, None)
    body.set('pos', numbers(pose[:3, 3]))
    body.set('quat', numbers(Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]))


def replace(root, name, assets, base_inverse):
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f'Missing body: {name}')
    appearance = body.find('geom').get('rgba', '.5 .5 .5 1')
    old = next(g for g in body.findall('geom') if g.get('contype', '1') != '0')
    contact = {k: old.get(k) for k in ('condim', 'friction', 'solref', 'solimp', 'margin', 'gap') if old.get(k)}
    for child in list(body):
        if child.tag in ('geom', 'inertial'):
            body.remove(child)
    info = json.loads((assets/'collision-assets.json').read_text())
    pose = base_inverse @ np.load(assets/'mesh.npz')['world_pose']
    set_pose(body, pose)
    inertia = np.asarray(info['inertia_kg_m2'])
    ET.SubElement(body, 'inertial', mass=str(info['mass_kg']), pos=numbers(info['center_of_mass_m']),
                  fullinertia=numbers(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
    asset = root.find('asset')
    for suffix, filename, visual in [('visual', 'visual.obj', True)] + [
            (f'convex_{i}', p['file'], False) for i, p in enumerate(info['parts'])]:
        mesh_name = f'{name}_{suffix}'
        if asset.find(f"mesh[@name='{mesh_name}']") is not None:
            raise ValueError(f'Duplicate mesh: {mesh_name}')
        ET.SubElement(asset, 'mesh', name=mesh_name, file=str((assets/filename).resolve()))
        ET.SubElement(body, 'geom', name=mesh_name, type='mesh', mesh=mesh_name,
                      density='0', rgba=appearance, group='2' if visual else '3',
                      contype='0' if visual else '1', conaffinity='0' if visual else '1', **contact)
    return {'parts': len(info['parts']), 'pose_base': pose.tolist(), 'mass_kg': info['mass_kg'],
            'mass_note': f"Source volume at assumed density {info.get('density_kg_m3', 600):g} kg/m^3; not measured mass."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--objects', nargs='+', required=True)
    ap.add_argument('--table', action='store_true', help='Restore the extracted circular tabletop full pose and dimensions')
    ap.add_argument('--boxes', nargs='*', default=[], help='Restore full poses and metric bounds of known box-shaped source meshes')
    ap.add_argument('--attach-static', action='append', default=[], metavar='BODY:INDEX,INDEX',
                    help='Attach explicitly identified assembly components to their free body, preserving initial world poses')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root = ET.parse(a.model).getroot()
    spec = json.loads(a.spec.read_text())
    inverse = np.linalg.inv(spec['robot']['T_world_base'])
    report = {n: replace(root, n, a.assets/n, inverse) for n in a.objects}
    for name in a.boxes:
        mesh = np.load(a.assets/name/'mesh.npz')
        pose = inverse @ mesh['world_pose']
        body = root.find(f".//body[@name='{name}']")
        set_pose(body, pose)
        geom = body.find('geom')
        if geom.get('type') != 'box':
            raise ValueError(f'{name} is not a box')
        geom.set('size', numbers(np.ptp(mesh['vertices'], axis=0)/2))
        report[name] = {'pose_base': pose.tolist(), 'geometry': 'box from extracted source bounds'}
    if a.table:
        table = np.load(a.assets/'table/mesh.npz')
        pose = inverse @ table['world_pose']
        body = root.find(".//body[@name='table']")
        set_pose(body, pose)
        geom = body.find("geom[@name='table_top']")
        vertices = table['vertices']
        geom.set('type', 'cylinder')
        geom.set('size', numbers([np.linalg.norm(vertices[:, :2], axis=1).max(), np.ptp(vertices[:, 2])/2]))
        report['table'] = {'pose_base': pose.tolist(), 'geometry': 'circular tabletop from extracted source bounds'}
    for assignment in a.attach_static:
        name, indices = assignment.split(':')
        body = root.find(f".//body[@name='{name}']")
        pose = np.eye(4)
        pose[:3, 3] = np.fromstring(body.get('pos'), sep=' ')
        quat = np.fromstring(body.get('quat'), sep=' ')
        pose[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        components = []
        for index in map(int, indices.split(',')):
            world = root.find('worldbody')
            geom = world.find(f"geom[@name='static{index}']")
            if geom is None:
                raise ValueError(f'Missing static{index}')
            gp = np.eye(4)
            gp[:3, 3] = np.fromstring(geom.get('pos'), sep=' ')
            q = np.fromstring(geom.get('quat'), sep=' ')
            gp[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            set_pose(geom, np.linalg.inv(pose) @ gp)
            geom.set('density', str(spec['objects'][name]['density']))
            world.remove(geom)
            body.append(geom)
            components.append(spec['static'][index]['name'])
        report.setdefault(name, {})['rigid_components'] = components
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    a.out.write_text(ET.tostring(root, encoding='unicode')+'\n')
    a.out.with_suffix('.repair.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
