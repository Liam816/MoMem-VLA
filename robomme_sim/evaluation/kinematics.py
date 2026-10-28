"""NumPy Panda URDF FK; returns robot-base to panda_hand_tcp pose."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

class PandaFK:

    def __init__(self, path=None, target='panda_hand_tcp'):
        if path is None:
            path = Path(__file__).resolve().parents[2] / 'third_party/ManiSkill/mani_skill/assets/robots/panda/panda_v2.urdf'
        tree = ET.parse(path).getroot()
        incoming = {j.find('child').get('link'): j for j in tree.findall('joint')}
        chain = []
        link = target
        while link in incoming:
            j = incoming[link]
            chain.append(j)
            link = j.find('parent').get('link')
        self.chain = list(reversed(chain))
        self.path = str(path)
        self.target = target
        variables = [j.get('name') for j in self.chain if j.get('type') != 'fixed']
        assert variables == [f'panda_joint{i}' for i in range(1, 8)], variables

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64)
        shape = q.shape[:-1]
        flat = q.reshape(-1, 7)
        T = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
        qi = 0

        def axis_rot(axis, theta):
            axis = np.asarray(axis, float)
            axis = axis / np.linalg.norm(axis)
            (x, y, z) = axis
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
            return np.eye(3)[None] + np.sin(theta)[:, None, None] * K[None] + (1 - np.cos(theta))[:, None, None] * (K @ K)[None]
        for j in self.chain:
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            R = np.eye(3)
            for (axis, theta) in zip([[0, 0, 1], [0, 1, 0], [1, 0, 0]], rpy[::-1]):
                R = R @ axis_rot(axis, np.array([theta]))[0]
            O = np.eye(4)
            O[:3, :3] = R
            O[:3, 3] = xyz
            T = T @ O
            if j.get('type') != 'fixed':
                M = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
                M[:, :3, :3] = axis_rot(np.fromstring(j.find('axis').get('xyz'), sep=' '), flat[:, qi])
                qi += 1
                T = T @ M
        return T.reshape(shape + (4, 4))
