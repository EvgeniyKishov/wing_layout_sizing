"""Feasible mass stagnation: three changes require four solved FSD states."""
import math

def mass_stagnation(history, tolerance=.001):
    if not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Mass tolerance must be positive and finite')
    rows=history[-4:]
    if len(rows)!=4 or any(r['phase']!='fsd' for r in rows):
        return False, []
    if any(b['iteration']!=a['iteration']+1 for a,b in zip(rows,rows[1:])):
        return False, []
    masses=[r['mass_kg'] for r in rows]
    stresses=[r['filtered_peak_MPa'] for r in rows]
    if any(not math.isfinite(m) or m<=0 for m in masses):
        return False, []
    changes=[abs(b-a)/a for a,b in zip(masses,masses[1:])]
    feasible=all(math.isfinite(s) and 0<=s<=300 for s in stresses)
    return feasible and all(c<tolerance for c in changes), changes
