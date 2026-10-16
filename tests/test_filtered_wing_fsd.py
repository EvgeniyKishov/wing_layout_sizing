import unittest
import numpy as np
from run_filtered_wing_fsd import replace_sections, update_total, active_variables, stopping_reason


class FilteredFSDTests(unittest.TestCase):
    def test_arithmetic_relaxation_and_only_physical_bounds(self):
        t=np.array([.01,.01,.01,.000625,.1])
        s=285e6*np.array([0.,1.,4.,0.,10.])
        np.testing.assert_allclose(update_total(t,s,285e6,'relaxed'),[.005,.01,.025,.000625,.15])

    def test_active_definition_strict_bounds_and_stress_band(self):
        t=[.000625,.15,.01,.01,.01,.01]
        s=np.array([200.,400.,285.,315.,284.,316.])*1e6
        r=active_variables(t,s)
        self.assertEqual(r['active_variables'],2)
        self.assertEqual(r['interior_variables'],4)
        self.assertEqual(r['active_below_band'],1)
        self.assertEqual(r['active_above_band'],1)
        self.assertAlmostEqual(r['active_fraction'],2/6)

    def test_stop_strict_five_percent_and_stress_guard(self):
        self.assertIsNone(stopping_reason(.04,299e6,.05,.05))
        self.assertEqual(stopping_reason(.04,299e6,.049,.05),'active_fraction')
        self.assertIsNone(stopping_reason(.04,301e6,0.,.05))
        self.assertIsNone(stopping_reason(.04,299e6,0.))
        self.assertEqual(stopping_reason(.01,300e6,.9,.05),'thickness_change')

    def test_direct_rule_has_no_internal_move_limits(self):
        total=np.array([.01,.01,.01,.001,.1])
        stress=285e6*np.array([.1,1.,4.,0.,10.])
        np.testing.assert_allclose(update_total(total,stress,target=285e6,rule='direct'),
                                   [.001,.01,.04,.000625,.15])

    def test_target_fixed_point_and_physical_bounds(self):
        total=np.array([.000625,.01,.15])
        np.testing.assert_allclose(update_total(total,np.full(3,294e6)),total)
        np.testing.assert_allclose(update_total(total,np.zeros(3)),[.000625,.008,.12])
        np.testing.assert_allclose(update_total(total,np.full(3,1176e6)),[.00078125,.0125,.15])

    def test_section_ids_not_text_order_and_offsets_preserved(self):
        code=''.join(f'SECTYPE,{i},SHELL\nSECDATA,.1,1,0,0\nSECDATA,.2,2,0,0\nSECDATA,.1,1,0,1\nSECOFFSET,{offset}\n' for i,offset in [(2,'MID'),(1,'BOT')])
        result=replace_sections(code,np.array([.01,.02]))
        rows=result.splitlines()
        self.assertAlmostEqual(float(rows[1].split(',')[1]),.008)
        self.assertAlmostEqual(float(rows[6].split(',')[1]),.004)
        self.assertEqual(rows[4],'SECOFFSET,MID')
        self.assertEqual(rows[9],'SECOFFSET,BOT')
        with self.assertRaises(ValueError):replace_sections(code,np.array([.01]))

    def test_bad_solver_fields_rejected(self):
        for stress in ([np.nan],[-1],[1,2]):
            with self.assertRaises(ValueError):update_total([.01],stress)


if __name__=='__main__':unittest.main()
