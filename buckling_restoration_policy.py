"""Switch the original energy heuristic to its existing restoration near a plateau."""
import math

SETTINGS=dict(window=3,maximum_absolute_blf_change=.002,maximum_relative_mass_change=.001,
              minimum_blf=.98,stress_limit_MPa=300.,maximum_energy_corrections=40,
              maximum_restoration_corrections=20)


def restoration_decision(history,policy='stagnation'):
    """Decide the NEXT correction from committed buckling states; latch restoration."""
    if policy not in ('stagnation','fixed40'):raise ValueError('Unknown restoration policy')
    rows=[r for r in history if r['phase']=='buckling']
    if not rows:return False,None
    if any(r.get('rule')=='uniform_restoration' for r in rows):return True,'restoration_latched'
    if rows[-1]['iteration']>=SETTINGS['maximum_energy_corrections']:return True,'energy_iteration_limit'
    n=SETTINGS['window']
    if policy=='fixed40' or len(rows)<n+1:return False,None
    window=rows[-n-1:]
    if any(not all(math.isfinite(r[k]) for k in ('BLF','mass_kg','filtered_peak_MPa')) or r['mass_kg']<=0 for r in window):
        raise ValueError('Invalid stagnation history')
    if any(b['iteration']!=a['iteration']+1 for a,b in zip(window,window[1:])):raise ValueError('Nonconsecutive history')
    # All three corrections must be near feasible, not merely the last one.
    if any(not SETTINGS['minimum_blf']<=r['BLF']<1 or r['filtered_peak_MPa']>SETTINGS['stress_limit_MPa'] for r in window):return False,None
    stable=all(abs(b['BLF']-a['BLF'])<SETTINGS['maximum_absolute_blf_change'] and
               abs(b['mass_kg']/a['mass_kg']-1)<SETTINGS['maximum_relative_mass_change']
               for a,b in zip(window,window[1:]))
    return (True,'near_feasible_stagnation') if stable else (False,None)
