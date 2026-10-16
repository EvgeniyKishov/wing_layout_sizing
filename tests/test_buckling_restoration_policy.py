import unittest
from buckling_restoration_policy import restoration_decision


def rows():
    return [dict(phase='buckling',iteration=i,BLF=.995+i*.001,mass_kg=426+i*.1,filtered_peak_MPa=297,rule='original_energy') for i in range(4)]


class RestorationTests(unittest.TestCase):
    def test_plateau_and_resume(self):
        h=rows();self.assertEqual(restoration_decision(h),(True,'near_feasible_stagnation'))
        self.assertEqual(restoration_decision(h[:3]),(False,None))
        h[-1]['rule']='uniform_restoration';h[-1]['filtered_peak_MPa']=305
        self.assertEqual(restoration_decision(h),(True,'restoration_latched'))

    def test_unsafe_or_changing_states_do_not_trigger(self):
        for key,value in [('BLF',.95),('filtered_peak_MPa',301),('mass_kg',430),('BLF',1.01)]:
            h=rows();h[-1][key]=value;self.assertEqual(restoration_decision(h),(False,None))

    def test_legacy_cap_and_policy(self):
        h=rows();self.assertEqual(restoration_decision(h,'fixed40'),(False,None))
        h[-1]['iteration']=40
        self.assertEqual(restoration_decision(h,'fixed40'),(True,'energy_iteration_limit'))

    def test_tiny_changes_far_from_feasible_do_not_trigger(self):
        h=rows()
        for r in h:r['BLF']-=.5
        self.assertEqual(restoration_decision(h),(False,None))


if __name__=='__main__':unittest.main()
