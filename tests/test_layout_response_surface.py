"""Analytical tests of bounded surrogate search and box handling; no ANSYS."""
import unittest
import tempfile, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from layout_response_surface import BOUNDS,points,fit_minimum,next_box


class ResponseSurfaceTests(unittest.TestCase):
    def test_seeded_three_stage_continuation(self):
        from run_layout_optimization import search
        calls=[]
        def evaluator(folder,ribs,spars,mesh_level):
            calls.append((ribs,tuple(spars)));folder.mkdir(parents=True)
            result=dict(status='feasible',feasible=True,objective_mass_kg=100+(spars[0]-.175)**2+(spars[1]-.7)**2,ribs=ribs,spars=spars)
            (folder/'summary.json').write_text(json.dumps(result));return result
        with tempfile.TemporaryDirectory() as tmp,patch('run_layout_optimization.evaluate',side_effect=evaluator),patch('run_layout_optimization.archive_candidate'):
            seed=Path(tmp)/'seed';out=Path(tmp)/'continued'
            search(seed,[8,16],max_evaluations=20,max_stages=1,mesh_level='extra-coarse')
            originals={str(p):p.read_bytes() for p in seed.rglob('*') if p.is_file()}
            initial=len(calls)
            search(out,[8,16],max_evaluations=60,max_stages=4,mass_tolerance=.001,mesh_level='extra-coarse',seed_from=seed)
            self.assertLessEqual(len(calls)-initial,54)
            self.assertEqual(len(calls),len(set(calls)))
            for ribs in [8,16]:
                state=json.loads((out/f'ribs_{ribs:02d}.json').read_text())
                self.assertEqual(len(state['stages']),4);self.assertEqual(state['status'],'converged')
            self.assertTrue(all(Path(p).read_bytes()==v for p,v in originals.items()))

    def test_one_stage_extra_coarse_batch(self):
        from run_layout_optimization import search
        calls=[]
        def evaluator(folder,ribs,spars,mesh_level):
            self.assertEqual(mesh_level,'extra-coarse');calls.append(ribs)
            folder.mkdir(parents=True)
            result=dict(status='feasible',feasible=True,objective_mass_kg=100+ribs+(spars[0]-.19)**2+(spars[1]-.67)**2,ribs=ribs,spars=spars)
            (folder/'summary.json').write_text(json.dumps(result));return result
        with tempfile.TemporaryDirectory() as tmp,patch('run_layout_optimization.evaluate',side_effect=evaluator),patch('run_layout_optimization.archive_candidate'):
            out=Path(tmp)
            search(out,[2,8,16,25],max_evaluations=40,max_stages=1,mesh_level='extra-coarse')
            self.assertEqual(len(calls),40)
            for n in [2,8,16,25]:
                state=json.loads((out/f'ribs_{n:02d}.json').read_text())
                self.assertEqual(len(state['stages']),1)
                self.assertEqual(state['status'],'stage_limit')
            settings=json.loads((out/'settings.json').read_text())
            self.assertEqual(settings['evaluator']['mesh_level'],'extra-coarse')
            self.assertEqual(settings['evaluator']['settings']['update_rule'],'direct')

    def test_interior(self):
        p=points(BOUNDS); optimum=np.array([.19,.67]);m=100+200*(p[:,0]-.19)**2+50*(p[:,1]-.67)**2
        x,r=fit_minimum(BOUNDS,p,m);np.testing.assert_allclose(x,optimum,atol=1e-12);self.assertAlmostEqual(r['predicted_mass_kg'],100)

    def test_boundary_and_indefinite(self):
        p=points(BOUNDS)
        for mass,expected in [(100+(p[:,0]-.3)**2+(p[:,1]-.7)**2,[.25,.7]),(100-p[:,0]**2+(p[:,1]-.7)**2,[.25,.7])]:
            x,_=fit_minimum(BOUNDS,p,mass);np.testing.assert_allclose(x,expected,atol=1e-11)

    def test_box_shifts_without_collapsing_ccd(self):
        b,move=next_box(BOUNDS,[.10,.90],True)
        self.assertFalse(move);np.testing.assert_allclose(np.diff(b,axis=1),np.diff(BOUNDS,axis=1)/2)
        self.assertEqual(len(np.unique(points(b),axis=0)),9)
        b2,move=next_box(b,[b[0,1],b[1,0]],True)
        self.assertTrue(move);np.testing.assert_allclose(np.diff(b2,axis=1),np.diff(b,axis=1))

    def test_degenerate_sampling_rejected(self):
        with self.assertRaises(ValueError):fit_minimum(BOUNDS,np.tile([.18,.7],(9,1)),np.ones(9)*100)

    def test_search_resume_and_feasibility_gate(self):
        from run_layout_optimization import search
        calls=[]
        def evaluator(folder,ribs,spars,mesh_level='coarse'):
            calls.append((ribs,spars));folder.mkdir(parents=True)
            feasible=not(spars[0]<.100001 and spars[1]<.500001)
            result=dict(status='feasible' if feasible else 'fsd_iteration_limit',feasible=feasible,objective_mass_kg=100+200*(spars[0]-.19)**2+50*(spars[1]-.67)**2 if feasible else None,ribs=ribs,spars=spars)
            (folder/'summary.json').write_text(json.dumps(result));return result
        with tempfile.TemporaryDirectory() as tmp,patch('run_layout_optimization.evaluate',side_effect=evaluator),patch('run_layout_optimization.archive_candidate'):
            out=Path(tmp)
            search(out,[20],max_evaluations=1)
            self.assertEqual(json.loads((out/'summary.json').read_text())['status'],'checkpoint')
            search(out,[20],max_evaluations=100)
            result=json.loads((out/'ribs_20.json').read_text())
            self.assertEqual(result['status'],'converged')
            self.assertAlmostEqual(result['best']['objective_mass_kg'],100)
            self.assertEqual(len(calls),len({(r,tuple(s)) for r,s in calls}))


if __name__=='__main__':unittest.main()
