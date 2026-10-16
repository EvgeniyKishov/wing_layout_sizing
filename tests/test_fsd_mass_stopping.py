import unittest
from fsd_mass_stopping import mass_stagnation

def rows():
    return [dict(phase='fsd',iteration=i,mass_kg=m,filtered_peak_MPa=300.)
            for i,m in enumerate([100.,100.05,100.,100.05])]

class MassStoppingTests(unittest.TestCase):
    def test_three_changes_and_both_signs(self):
        self.assertTrue(mass_stagnation(rows())[0])
        self.assertFalse(mass_stagnation(rows()[:3])[0])

    def test_any_violation_invalidates_window(self):
        for i in range(4):
            h=rows(); h[i]['filtered_peak_MPa']=300.01
            self.assertFalse(mass_stagnation(h)[0])

    def test_large_change_not_cancelled_by_return(self):
        h=rows();h[1]['mass_kg']=101
        self.assertFalse(mass_stagnation(h)[0])

    def test_invalid_or_nonconsecutive(self):
        for key,value in [('mass_kg',float('nan')),('filtered_peak_MPa',float('nan')),('phase','buckling'),('iteration',9)]:
            h=rows();h[1][key]=value
            self.assertFalse(mass_stagnation(h)[0])
        with self.assertRaises(ValueError):mass_stagnation(rows(),0)

if __name__=='__main__':unittest.main()
