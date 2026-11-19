"""Shared APDL core-growth factor used by the wing and benchmark.

The caller supplies effective_energy from its modal-energy weighting, the
current minimum buckling factor and existing damping/cap parameters.
This is a heuristic, not an eigenvalue derivative.
"""


def core_growth_factor_commands():
    return """
req_ratio = TARGET_BLF/min_blf
*if,req_ratio,gt,MAX_REQ_RATIO,then
    req_ratio = MAX_REQ_RATIO
*endif
*if,buck_iter,le,5,and,min_blf,lt,0.2,then
    factor = sqrt(req_ratio)
*else
    factor = 1+(req_ratio-1)*effective_energy*DAMP_FACTOR
*endif
factor = factor**DAMP_EXP
*if,factor,gt,MAX_FACTOR,then
    factor = MAX_FACTOR
*endif
*if,factor,gt,MAX_INCREASE_RATIO,then
    factor = MAX_INCREASE_RATIO
*endif
"""
