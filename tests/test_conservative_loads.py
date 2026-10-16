"""Physical invariants of the vertical aerodynamic load transfer."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from conservative_loads import transfer


class ConservativeLoadsTests(unittest.TestCase):
    def test_generated_load_steps_have_independent_signed_resultants(self):
        from nagryzka_ansys import generate_combined_apdl_for_iteration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cdb = root/'mesh.cdb'
            cdb.write_text('N,1,0,0,1\nN,2,1,0,1\nN,3,1,1,1\nN,4,0,1,1\nEN,1,1,2,3,4\nEMODIF,1,TYPE,1\n')
            (root/'0-0.txt').write_text('X Y Z\n0.5 0.5 1\n')
            (root/'0_1.txt').write_text('0.2 0.3 -10\n')
            (root/'0_-1.txt').write_text('0.8 0.7 4\n')
            group = dict(iteration=0,cdb=str(cdb),cases=[('1',1,None),('-1',-1,None)])
            with patch('builtins.print'):
                self.assertTrue(generate_combined_apdl_for_iteration(folder,group))
            loads = {}
            resultants = []
            for line in (root/'simple_load_0.apdl').read_text(encoding='utf-8').splitlines():
                parts = line.split(',')
                if parts[0] == 'F':
                    self.assertEqual(parts[2], 'FZ')
                    loads[int(parts[1])] = float(parts[3])
                elif parts[0] == 'FDELE':
                    loads.clear()
                elif parts[0] == 'LSWRITE':
                    resultants.append(sum(loads.values()))
                self.assertNotEqual(parts[0], 'SFE')
            np.testing.assert_allclose(resultants,[10,-4],atol=1e-12)

    def test_signed_forces_and_moments_on_sloping_skin(self):
        xyz = np.array([[0,0,0], [3,0,1], [2,2,.8], [-1,2,.2]])
        raw = np.array([[.5,.4,-10], [1.2,1.1,3], [.2,1.5,-7]])
        fz, report = transfer(raw, [10,20,30,40], xyz)
        source_forces = np.column_stack((np.zeros((3,2)), -raw[:,2]))
        target_forces = np.column_stack((np.zeros((4,2)), fz))
        source_positions = np.column_stack((raw[:,:2], np.zeros(3)))
        np.testing.assert_allclose(target_forces.sum(axis=0), source_forces.sum(axis=0), atol=1e-12)
        np.testing.assert_allclose(np.cross(xyz,target_forces).sum(axis=0),
                                   np.cross(source_positions,source_forces).sum(axis=0), atol=1e-12)
        self.assertEqual(report['source_points'], 3)

    def test_outside_point_is_not_silently_clamped(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            transfer([[2,2,-1]], [1,2,3], [[0,0,0],[1,0,0],[0,1,0]])

    def test_coincident_projection_uses_upper_node(self):
        xyz = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
        fz, _ = transfer([[0,0,-2]], [1,2,3,4], xyz)
        np.testing.assert_allclose(fz, [0,0,0,2], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
