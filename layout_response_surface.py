"""Bounded two-variable CCD search primitives, independent of the FE solver."""
import numpy as np

BOUNDS = np.array([[.10, .25], [.50, .90]])
CCD = np.array([[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1],[0,0]], float)


def points(box):
    box = np.asarray(box, float)
    if box.shape != (2,2) or not np.isfinite(box).all() or np.any(box[:,1] <= box[:,0]):
        raise ValueError('Invalid local box')
    return box.mean(1) + CCD * np.diff(box, axis=1).ravel()/2


def basis(x):
    x = np.atleast_2d(x)
    a,b = x.T
    return np.column_stack([np.ones(len(x)),a,b,a*a,a*b,b*b])


def fit_minimum(box, xy, mass):
    """Fit feasible samples; enumerate every possible bounded quadratic minimum."""
    box = np.asarray(box); mass = np.asarray(mass)
    z = (np.asarray(xy)-box.mean(1))/(np.diff(box,axis=1).ravel()/2)
    A = basis(z)
    if len(mass)<6 or not np.isfinite(mass).all() or np.any(mass<=0):
        raise ValueError('At least six finite feasible masses required')
    coeff,_,rank,_ = np.linalg.lstsq(A,mass,rcond=None)
    condition = np.linalg.cond(A)
    if rank!=6 or condition>1e8: raise ValueError('Insufficient sampling geometry')
    H=np.array([[2*coeff[3],coeff[4]],[coeff[4],2*coeff[5]]])
    candidates=[p for p in CCD[:4]]
    if abs(np.linalg.det(H))>1e-14:
        p=np.linalg.solve(H,-coeff[1:3])
        if np.all(np.abs(p)<=1): candidates.append(p)
    for fixed in (-1.,1.):
        if coeff[5]>0: candidates.append(np.array([fixed,np.clip(-(coeff[2]+coeff[4]*fixed)/(2*coeff[5]),-1,1)]))
        if coeff[3]>0: candidates.append(np.array([np.clip(-(coeff[1]+coeff[4]*fixed)/(2*coeff[3]),-1,1),fixed]))
    values=basis(candidates)@coeff; k=int(np.argmin(values)); p=candidates[k]
    xy=box.mean(1)+p*np.diff(box,axis=1).ravel()/2
    return xy,dict(coefficients=coeff.tolist(),condition=float(condition),predicted_mass_kg=float(values[k]),rms_fit_kg=float(np.sqrt(np.mean((A@coeff-mass)**2))))


def next_box(box, best, improved):
    """Translate toward an improving artificial edge; otherwise halve both widths."""
    box=np.asarray(box,float); best=np.asarray(best,float)
    artificial=((np.isclose(best,box[:,0],atol=1e-9,rtol=0)&(box[:,0]>BOUNDS[:,0]+1e-9)) |
                (np.isclose(best,box[:,1],atol=1e-9,rtol=0)&(box[:,1]<BOUNDS[:,1]-1e-9)))
    move=bool(improved and artificial.any())
    width=np.diff(box,axis=1).ravel()*(1 if move else .5)
    low=np.clip(best-width/2,BOUNDS[:,0],BOUNDS[:,1]-width)
    return np.column_stack([low,low+width]),move
