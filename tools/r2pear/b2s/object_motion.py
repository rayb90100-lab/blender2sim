"""Local-transform bindings for object-joint robots and keyed gripper visuals."""
import numpy as np
from scipy.spatial.transform import Rotation


def interpolate_pose(values, matrices, value):
    """Interpolate translation/scale and shortest-arc rotation; extend end segments.

    Outside the recorded gripper range this is a visual extrapolation, not a
    calibrated mechanism model. Physics still uses the MuJoCo gripper.
    """
    i = int(np.clip(np.searchsorted(values, value, side='right') - 1, 0, len(values) - 2))
    t = (value - values[i]) / (values[i+1] - values[i])
    a, b = np.asarray(matrices[i]), np.asarray(matrices[i+1])
    sa, sb = np.linalg.norm(a[:3, :3], axis=0), np.linalg.norm(b[:3, :3], axis=0)
    ra, rb = a[:3, :3]/sa, b[:3, :3]/sb
    rotation = ra @ Rotation.from_rotvec(Rotation.from_matrix(ra.T @ rb).as_rotvec()*t).as_matrix()
    result = np.eye(4)
    result[:3, :3] = rotation * (sa + t*(sb-sa))
    result[:3, 3] = a[:3, 3] + t*(b[:3, 3]-a[:3, 3])
    return result


def gripper_binding(ex, emb):
    descendants = set(ex.descendants(emb['base']))
    explicit = emb.get('gripper_property')
    candidates = [(n, key, values) for n, props in ex.E['props_anim'].items() if n in descendants
                  for key, values in props.items()
                  if ((n == explicit['object'] and key == explicit['property']) if explicit
                      else key == 'observation.gripper_position')]
    if not candidates:
        raise ValueError('Object robot needs an animated observation.gripper_position property for keyed gripper binding')
    if len(candidates) != 1:
        raise ValueError('Ambiguous object-robot gripper property')
    owner, key, g = candidates[0]
    g = np.asarray(g)
    if g.shape != (ex.T,) or not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
        raise ValueError('Invalid normalized gripper animation')
    values, indices = np.unique(g, return_index=True)
    if len(values) < 2:
        raise ValueError('Need at least two gripper openings to bind keyed geometry')
    channels = {}
    for name in sorted(descendants - set(emb['chain'])):
        matrices = np.asarray(ex.E.get('object_basis', {}).get(name, []), float)
        if not len(matrices) or np.allclose(matrices, matrices[0], atol=1e-7, rtol=0):
            continue
        if ex.O[name].get('constraints') or not np.isfinite(matrices).all():
            raise ValueError(f'Unsupported gripper transform: {name}')
        poses = matrices[indices]
        predicted = np.array([interpolate_pose(values, poses, v) for v in g])
        if np.max(np.abs(predicted - matrices)) > 2e-5:
            raise ValueError(f'Local animation is not a function of gripper opening: {name}')
        channels[name] = poses.tolist()
    if not channels:
        raise ValueError('No keyed gripper geometry found')
    return {'property_object': owner, 'property': key, 'values': values.tolist(), 'channels': channels,
            'outside_recorded_range': 'visual endpoint-segment extrapolation; not calibrated'}


class ObjectRobotBinding:
    """Drive the original hierarchy without changing parents or parent inverses."""
    def __init__(self, bpy, binding):
        self.bpy, self.binding = bpy, binding
        self.joints = [bpy.data.objects[n] for n in binding['objects']]
        self.initial = [np.array(ob.matrix_basis) for ob in self.joints]
        grip = binding['gripper']
        self.gripper_objects = {n: bpy.data.objects[n] for n in grip['channels']}
        self.property_object = bpy.data.objects[grip['property_object']]
        for ob in [*self.joints, *self.gripper_objects.values(), self.property_object]:
            if ob.animation_data:
                ob.animation_data.action = None

    def apply(self, q=None, gripper=None):
        from mathutils import Matrix
        binding = self.binding
        if q is not None:
            q = np.asarray(q, float)
            if q.shape != (len(self.joints),) or not np.isfinite(q).all():
                raise ValueError('Invalid object-robot joint state')
            for i, ob in enumerate(self.joints):
                angle = (q[i]-binding['offset'][i])/binding['sign'][i]
                local = self.initial[i].copy()
                local[:3, :3] = Rotation.from_rotvec(np.asarray(binding['axes'][i])*angle).as_matrix()
                ob.matrix_basis = Matrix(local.tolist())
        if gripper is not None:
            g = float(np.clip(gripper, 0, 1))
            grip = binding['gripper']
            self.property_object[grip['property']] = g
            self.property_object.update_tag()
            for name, ob in self.gripper_objects.items():
                ob.matrix_basis = Matrix(interpolate_pose(grip['values'], grip['channels'][name], g).tolist())

