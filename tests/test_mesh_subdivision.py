import unittest
import numpy as np
from run_wing_mesh_check import subdivide


class SubdivisionTests(unittest.TestCase):
    def test_shared_edges_support_extent_and_oriented_area(self):
        nodes = {1:(0,0,0),2:(1,0,0),3:(1,1,0),4:(0,1,0),5:(2,0,0),6:(2,1,0)}
        elements = {1:[1,2,3,4],2:[2,5,6,3]}
        refined,children,fixed,parents = subdivide(nodes,elements,{1,2,5})
        self.assertEqual(len(children),8)
        self.assertEqual(len(refined),15)  # six originals, seven shared edges, two centers
        self.assertEqual(len(fixed),5)  # only the two bottom edge midpoints are added
        midpoint = [n for n,x in refined.items() if np.allclose(x,[1,.5,0])]
        self.assertEqual(len(midpoint),1)
        adjacent = {parents[e] for e,c in children.items() if midpoint[0] in c}
        self.assertEqual(adjacent,{1,2})
        oriented = np.zeros(3)
        for corners in children.values():
            xyz = np.array([refined[n] for n in corners])
            oriented += np.cross(xyz,np.roll(xyz,-1,axis=0)).sum(axis=0)/2
        np.testing.assert_allclose(oriented,[0,0,2],atol=1e-12)


if __name__=='__main__':
    unittest.main()
