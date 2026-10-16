"""Independent invariants and hand-calculated cases for stress averaging."""
import unittest
import numpy as np
from stress_filter import ConicalStressFilter, filter_stress_batched


class StressFilterTests(unittest.TestCase):
    def test_batched_matches_sparse_for_irregular_grouped_mesh(self):
        rng=np.random.default_rng(42);xyz=rng.random((57,3));areas=rng.uniform(.01,2,57)
        groups=np.arange(57)%3;stress=rng.uniform(0,300e6,57)
        reference=ConicalStressFilter(xyz,areas,groups,.6)
        for batch in (1,7,128):
            actual,counts=filter_stress_batched(xyz,areas,groups,.6,stress,batch)
            np.testing.assert_allclose(actual,reference.apply(stress),rtol=1e-14)
            np.testing.assert_array_equal(counts,reference.weights.getnnz(axis=1))

    def test_constant_with_boundaries_unequal_areas_and_isolated_point(self):
        p=np.array([[0,0],[.5,0],[1,0],[0,.5],[.5,.5],[10,10]])
        f=ConicalStressFilter(p,[1,2,3,4,5,6],[0]*6,.8)
        np.testing.assert_allclose(f.apply(np.full(6,296e6)),296e6,rtol=1e-14)

    def test_hand_calculated_area_and_cone_weights(self):
        f=ConicalStressFilter([[0,0],[1,0]],[1,3],[0,0],2)
        np.testing.assert_allclose(f.apply([10,30]),[22,190/7],rtol=1e-14)

    def test_coincident_different_surfaces_do_not_mix(self):
        f=ConicalStressFilter([[0,0],[0,0],[1,0],[1,0]],[1]*4,[0,1,0,1],2)
        np.testing.assert_allclose(f.apply([10,100,10,100]),[10,100,10,100])

    def test_outside_and_radius_boundary_have_no_influence(self):
        f=ConicalStressFilter([[0,0],[1,0],[3,0]],[1]*3,[0]*3,1)
        np.testing.assert_array_equal(f.apply([1,2,100]),[1,2,100])

    def test_area_partition_at_same_location_is_invariant(self):
        a=ConicalStressFilter([[0,0],[1,0]],[1,3],[0,0],2).apply([10,30])
        b=ConicalStressFilter([[0,0],[1,0],[1,0]],[1,1,2],[0]*3,2).apply([10,30,30])
        np.testing.assert_allclose(b,[a[0],a[1],a[1]])

    def test_permutation_bounds_and_input_preservation(self):
        p=np.array([[0,0],[.4,0],[.8,0],[1.2,0]])
        s=np.array([10.,20.,80.,40.]);before=s.copy();area=np.array([1,2,3,1])
        f=ConicalStressFilter(p,area,[0]*4,.9);a=f.apply(s)
        order=np.array([3,1,0,2])
        b=ConicalStressFilter(p[order],area[order],[0]*4,.9).apply(s[order])
        np.testing.assert_allclose(b,a[order]);np.testing.assert_array_equal(s,before)
        self.assertTrue(np.all((a>=s.min())&(a<=s.max())))

    def test_invalid_inputs(self):
        for radius in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):ConicalStressFilter([[0,0]],[1],[0],radius)
        for area in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):ConicalStressFilter([[0,0]],[area],[0],1)
        with self.assertRaises(ValueError):ConicalStressFilter([[np.nan,0]],[1],[0],1)
        with self.assertRaises(ValueError):ConicalStressFilter([[0,0]],[1],[0,1],1)
        f=ConicalStressFilter([[0,0]],[1],[0],1)
        for s in [[-1],[np.nan],[np.inf],[1,2]]:
            with self.assertRaises(ValueError):f.apply(s)

if __name__=='__main__':unittest.main()
