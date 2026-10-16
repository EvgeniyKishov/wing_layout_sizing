"""Conservative transfer of the MATLAB export's vertical forces to FE nodes.

The legacy three-column file stores (x, y, -delta_Y_z), not a complete
aerodynamic pressure/traction field. Barycentric weights in projected x-y
coordinates preserve vertical force and its two moments without creating
horizontal force. This does not supply absent drag or pitching-moment data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from nagryzka_ansys import parse_nodes_from_cdb, parse_elements_from_cdb, element_center


def transfer(raw, node_ids, xyz):
    raw=np.asarray(raw,dtype=float)
    xyz=np.asarray(xyz,dtype=float)
    node_ids=np.asarray(node_ids,dtype=int)
    if raw.ndim!=2 or raw.shape[1]!=3 or xyz.shape!=(len(node_ids),3):
        raise ValueError('Expected source (x,y,minus_Fz) and target (x,y,z) arrays')
    if not np.isfinite(raw).all() or not np.isfinite(xyz).all():
        raise ValueError('Nonfinite coordinates or forces')
    # At a coincident projected location choose the highest connected skin node.
    unique={}
    for i,point in enumerate(xyz):
        key=tuple(point[:2])
        if key not in unique or point[2]>xyz[unique[key],2]:
            unique[key]=i
    selected=np.array(list(unique.values()))
    points=xyz[selected,:2]
    triangulation=Delaunay(points)
    simplex=triangulation.find_simplex(raw[:,:2],tol=1e-10)
    if (simplex<0).any():
        bad=np.flatnonzero(simplex<0)
        raise ValueError(f'{len(bad)} aerodynamic points outside projected upper-skin mesh; first rows {bad[:8].tolist()}')
    transform=triangulation.transform[simplex]
    delta=raw[:,:2]-transform[:,2,:]
    first=np.einsum('nij,nj->ni',transform[:,:2,:],delta)
    weights=np.column_stack((first,1-first.sum(axis=1)))
    if weights.min() < -1e-9:
        raise ValueError('Invalid barycentric weights')
    fz=np.zeros(len(node_ids))
    vertices=selected[triangulation.simplices[simplex]]
    np.add.at(fz,vertices.ravel(),(-raw[:,2,None]*weights).ravel())
    source=np.array([-raw[:,2].sum(),np.dot(raw[:,1],-raw[:,2]),np.dot(raw[:,0],raw[:,2])])
    target=np.array([fz.sum(),np.dot(xyz[:,1],fz),-np.dot(xyz[:,0],fz)])
    scale=np.maximum(np.abs(source),np.sum(np.abs(raw[:,2]))*1e-6)
    if np.any(np.abs(target-source)>scale*1e-10):
        raise ValueError('Force or moment conservation failed')
    return fz,dict(source_Fz_Mx_My=source.tolist(),target_Fz_Mx_My=target.tolist(),
                   residual_Fz_Mx_My=(target-source).tolist(),source_points=len(raw),
                   target_nodes=len(node_ids),loaded_nodes=int(np.count_nonzero(fz)))


def from_files(cdb,centers,forces):
    nodes=parse_nodes_from_cdb(cdb)
    elements,types=parse_elements_from_cdb(cdb)
    skin_ids=[eid for eid in elements if types.get(eid,1)==1]
    centroids=np.array([element_center(nodes,elements[eid]) for eid in skin_ids])
    # Use the generator's upper-surface classification, not z>0 on a cambered airfoil.
    with Path(centers).open(encoding='utf-8-sig') as stream:
        if stream.readline().split() != ['X', 'Y', 'Z']:
            raise ValueError('Expected upper-skin center file with X Y Z header')
    targets=np.loadtxt(centers,skiprows=1,ndmin=2)
    distances,indices=cKDTree(centroids).query(targets[:,:3])
    if distances.max()>2e-6 or len(set(indices.tolist()))!=len(indices):
        raise ValueError('Upper-skin centers do not match unique current mesh elements')
    upper=[skin_ids[i] for i in indices]
    ids=sorted({nid for eid in upper for nid in elements[eid]})
    xyz=np.array([nodes[nid] for nid in ids])
    raw=np.loadtxt(forces,ndmin=2)
    fz,report=transfer(raw,ids,xyz)
    report.update(method='Projected barycentric nodal vertical-force transfer',
                  sign_convention='FZ = -column 3 of legacy MATLAB export (delta_Y_z)',
                  source_scope='Vertical aerodynamic force only; no inferred horizontal traction',
                  upper_elements=len(upper),max_center_matching_distance_m=float(distances.max()),
                  sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (cdb,centers,forces)})
    return ids,fz,report


def write_loads(cdb,centers,forces,apdl_path,report_path):
    ids,fz,report=from_files(cdb,centers,forces)
    commands=['! Global vertical forces; conservative transfer of -legacy force column\n']
    commands += [f'F,{nid},FZ,{value:.16e}\n' for nid,value in zip(ids,fz) if value!=0]
    Path(apdl_path).write_text(''.join(commands),encoding='ascii')
    Path(report_path).write_text(json.dumps(report,indent=2))
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir')
    p.add_argument('--case',default='12.72')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    run=Path(args.run_dir).resolve()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=write_loads(run/'wing_mesh_config_0.cdb',run/'0-0.txt',run/f'0_{args.case}.txt',
                      out/'nodal_loads.apdl',out/'transfer.json')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
