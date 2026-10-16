"""Standalone area-weighted conical filter for scalar shell stress measures.

No solver or optimization hooks. Distances are Euclidean between supplied
centroids, in the same physical units as radius. Groups must identify separate
continuous surfaces/fields, not merely element types. On folded surfaces,
Euclidean distance is not a geodesic distance: callers must partition them.
This changes the reported stress measure, not the equilibrium stress tensor.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


class ConicalStressFilter:
    """Build once per mesh/radius; apply to finite, nonnegative scalar stresses.

    W_ij = A_j max(0, 1-distance(i,j)/radius), normalized by row sum.
    Self contribution is retained; truncated boundary neighbourhoods are
    renormalized. Constant fields are preserved, but the global area integral
    of a nonconstant field need not be. Neither equilibrium nor local strength
    feasibility follows from filtering. Do not use this scalar field for Kg.
    """

    def __init__(self, centers, areas, groups, radius):
        xyz=np.asarray(centers,dtype=float)
        area=np.asarray(areas,dtype=float)
        group=np.asarray(groups)
        radius=float(radius)
        if xyz.ndim!=2 or xyz.shape[1] not in (2,3) or len(xyz)==0 or not np.isfinite(xyz).all():
            raise ValueError('centers must be a nonempty finite N x 2 or N x 3 array')
        n=len(xyz)
        if area.shape!=(n,) or not np.isfinite(area).all() or (area<=0).any():
            raise ValueError('areas must be finite, positive and have length N')
        if group.shape!=(n,) or group.dtype.kind not in 'iu':
            raise ValueError('groups must be an integer surface-group array of length N')
        if not np.isfinite(radius) or radius<=0:
            raise ValueError('radius must be finite and positive')
        rows=[];cols=[];values=[]
        for label in np.unique(group):
            ids=np.flatnonzero(group==label);points=xyz[ids];tree=cKDTree(points)
            for local,near in enumerate(tree.query_ball_point(points,radius)):
                near=np.asarray(sorted(near),dtype=int)
                distance=np.linalg.norm(points[near]-points[local],axis=1)
                weights=area[ids[near]]*np.maximum(0.,1.-distance/radius)
                keep=weights>0;near=near[keep];weights=weights[keep]
                total=weights.sum()
                if not np.isfinite(total) or total<=0:
                    raise ValueError('invalid weight sum; check geometry/area scales')
                rows.extend([ids[local]]*len(near));cols.extend(ids[near]);values.extend(weights/total)
        self.weights=csr_matrix((values,(rows,cols)),shape=(n,n))
        self.radius=radius

    def apply(self, stress):
        stress=np.asarray(stress,dtype=float)
        if stress.shape!=(self.weights.shape[0],) or not np.isfinite(stress).all() or (stress<0).any():
            raise ValueError('stress must be a finite, nonnegative scalar array of length N')
        return np.asarray(self.weights@stress)


def filter_stress_batched(centers, areas, groups, radius, stress, batch_size=128):
    """Same scalar operator without retaining a whole-wing sparse matrix.

    Returns filtered values and positive-weight neighbour counts (including self).
    Batching bounds neighbour-list memory; physical radius and groups are unchanged.
    """
    xyz=np.asarray(centers,dtype=float);area=np.asarray(areas,dtype=float)
    group=np.asarray(groups);s=np.asarray(stress,dtype=float);radius=float(radius)
    if xyz.ndim!=2 or xyz.shape[1] not in (2,3) or len(xyz)==0 or not np.isfinite(xyz).all():
        raise ValueError('centers must be a nonempty finite N x 2 or N x 3 array')
    n=len(xyz)
    if area.shape!=(n,) or not np.isfinite(area).all() or (area<=0).any():raise ValueError('invalid areas')
    if group.shape!=(n,) or group.dtype.kind not in 'iu':raise ValueError('invalid groups')
    if s.shape!=(n,) or not np.isfinite(s).all() or (s<0).any():raise ValueError('invalid scalar stress')
    if not np.isfinite(radius) or radius<=0:raise ValueError('invalid radius')
    if not isinstance(batch_size,int) or batch_size<1:raise ValueError('invalid batch size')
    result=np.empty(n);counts=np.empty(n,dtype=int)
    for label in np.unique(group):
        ids=np.flatnonzero(group==label);points=xyz[ids];tree=cKDTree(points)
        for start in range(0,len(ids),batch_size):
            neighbours=tree.query_ball_point(points[start:start+batch_size],radius)
            for k,near in enumerate(neighbours,start):
                near=np.asarray(sorted(near),dtype=int)
                d=np.linalg.norm(points[near]-points[k],axis=1)
                w=area[ids[near]]*np.maximum(0.,1.-d/radius);positive=w>0
                total=w[positive].sum()
                if not np.isfinite(total) or total<=0:raise ValueError('invalid weight sum')
                result[ids[k]]=np.dot(w[positive]/total,s[ids[near[positive]]])
                counts[ids[k]]=positive.sum()
    return result,counts
