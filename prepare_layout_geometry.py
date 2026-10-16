"""Fresh Fokker layout, canonical sandwich sections and conservative load transfer."""
from pathlib import Path
import argparse, contextlib, os, shutil, json, sys
import numpy as np
from scipy.spatial import cKDTree
from nagryzka_ansys import parse_nodes_from_cdb, parse_elements_from_cdb
from conservative_loads import write_loads
from diagnose_wing_mesh import audit
from final_restoration import face_stress_commands

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'examples/fokker/for_ansys_1.npy'
FORCES=ROOT/'examples/fokker/aerodynamic_forces.txt'


def prepare(out,ribs,spars,mesh_settings=None,refine=False):
    if not 2<=ribs<=25 or not .10<=spars[0]<=.25 or not .50<=spars[1]<=.90:
        raise ValueError('Layout outside agreed bounds')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    row=np.load(CONFIG)[0].copy();row[10:12]=spars;row[15]=ribs
    np.save(out/'configuration.npy',row)
    shutil.copy2(ROOT/'examples/fokker/naca2412-il.csv',out/'naca2412-il.csv')
    os.environ['MPLBACKEND']='Agg'
    import geom_ansys as geom
    geom.nodes.clear();geom.elements.clear();geom.node_dict.clear()
    previous=Path.cwd()
    try:
        os.chdir(out)
        with (out/'geometry.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
            generator=geom.WingMeshGenerator(row)
            if mesh_settings is not None:
                limits={'num_sections':2,'interpolation_density':12,'spar_points_multiplier':3}
                if any(k not in limits or not isinstance(v,int) or v<limits[k] for k,v in mesh_settings.items()):
                    raise ValueError('Invalid mesh resolution override')
                generator.params.update(mesh_settings)
            generator.generate_mesh('geometry.cdb')
        centers_file=list(out.glob('upper_centers_*.txt'))
        if len(centers_file)!=1:raise ValueError('Missing unique upper surface export')
        shutil.copy2(centers_file[0],out/'upper_centers.txt')
    finally:os.chdir(previous)
    nodes=parse_nodes_from_cdb(out/'geometry.cdb');els,types=parse_elements_from_cdb(out/'geometry.cdb')
    assert list(nodes)==list(range(1,len(nodes)+1)) and list(els)==list(range(1,len(els)+1))
    xyz=np.array(list(nodes.values()));corners=np.array(list(els.values()));types=np.array([types[i] for i in els])
    c=xyz[corners-1].mean(1);groups=np.full(len(c),-1,int)
    skin=np.flatnonzero(types==1);dist,idx=cKDTree(c[skin]).query(np.loadtxt(out/'upper_centers.txt',skiprows=1))
    assert dist.max()<2e-6 and len(set(idx))==len(idx)
    groups[skin]=1;groups[skin[idx]]=0
    # Connected components distinguish the two spars without assuming element counts.
    parent=np.arange(len(c));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in np.flatnonzero(types==2):
        for n in corners[i]:
            if n in owner:parent[find(i)]=find(owner[n])
            else:owner[n]=i
    components={}
    for i in np.flatnonzero(types==2):components.setdefault(find(i),[]).append(i)
    assert len(components)==2
    for g,ids in enumerate(sorted(components.values(),key=lambda ids:c[ids,0].mean()),2):groups[ids]=g
    stations=np.unique(c[types==3,1]);assert len(stations)==ribs
    np.testing.assert_allclose(stations,np.linspace(0,stations[-1],ribs),atol=1e-6)
    for g,y in enumerate(stations,4):groups[(types==3)&(c[:,1]==y)]=g
    assert (groups>=0).all(), 'Unexpected component requires explicit surface group'
    parent_ids=np.arange(1,len(c)+1)
    if refine:
        from run_wing_mesh_check import subdivide
        original=(out/'geometry.cdb').read_text(encoding='utf-8',errors='replace')
        fixed=set()
        for line in original.splitlines():
            if line.startswith('D,'):
                parts=line.split(',')
                if parts[2:]!=['ALL','0']:raise ValueError('Expected clamped supports')
                fixed.add(int(parts[1]))
        shutil.copy2(out/'geometry.cdb',out/'parent_geometry.cdb')
        np.savez_compressed(out/'parent_mesh.npz',xyz=xyz,corners=corners,types=types,groups=groups)
        nodes,els,fixed,parents=subdivide(nodes,els,fixed)
        parent_ids=np.array(list(parents.values()));types=types[parent_ids-1];groups=groups[parent_ids-1]
        xyz=np.array(list(nodes.values()));corners=np.array(list(els.values()));c=xyz[corners-1].mean(1)
        refined=[line for line in original.splitlines() if line.startswith('MP,')]
        refined += [f'N,{i},'+','.join(f'{v:.16e}' for v in point) for i,point in nodes.items()]
        for i,ns in els.items():refined += [f'EN,{i},'+','.join(map(str,ns)),f'EMODIF,{i},TYPE,{types[i-1]}']
        refined += [f'D,{i},ALL,0' for i in sorted(fixed)]
        (out/'geometry.cdb').write_text('\n'.join(refined)+'\n',encoding='ascii')
        np.savetxt(out/'upper_centers.txt',c[groups==0],header='X Y Z',comments='',fmt='%.16e')
    p=xyz[corners-1]
    areas=(np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)+np.linalg.norm(np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]),axis=1))/2
    assert (areas>0).all()
    transfer=write_loads(out/'geometry.cdb',out/'upper_centers.txt',FORCES,out/'loads.apdl',out/'load_transfer.json')
    lines=['/CLEAR','/FILNAME,diagnostic,1','/PREP7','SHPP,ON']
    lines += [s for s in (out/'geometry.cdb').read_text(encoding='utf-8',errors='replace').splitlines() if s.startswith(('N,','MP,'))]
    for k in range(1,5):lines += [f'ET,{k},SHELL181',f'KEYOPT,{k},3,2',f'KEYOPT,{k},8,2']
    for i,t in enumerate(types,1):
        offset={1:'BOT',2:'MID',3:'MID',4:'TOP'}[int(t)]
        lines += [f'SECTYPE,{i},SHELL','SECDATA,0.004,1,0,0','SECDATA,0.002,2,0,0','SECDATA,0.004,1,0,1',f'SECOFFSET,{offset}',f'TYPE,{t}','MAT,1',f'SECNUM,{i}',f'EN,{i},'+','.join(map(str,corners[i-1]))]
    fixed=[s for s in (out/'geometry.cdb').read_text(encoding='utf-8',errors='replace').splitlines() if s.startswith('D,')]
    lines+=fixed+(out/'loads.apdl').read_text().splitlines()
    lines += [f'ne={len(c)}',f'nn={len(xyz)}','*DIM,rr_stress,ARRAY,ne','*DIM,rr_tmp,ARRAY,ne','FINISH','/SOLU','ANTYPE,STATIC','PSTRES,ON','SOLVE','FINISH',face_stress_commands(),'/POST1','SET,LAST','ALLSEL,ALL','*CFOPEN,stress,txt','*VWRITE,rr_stress(1)','(E24.16)','*CFCLOS','*DIM,du,ARRAY,nn,3']
    for j,v in enumerate('XYZ',1):lines += [f'*VGET,du(1,{j}),NODE,1,U,{v}']
    lines += ['*CFOPEN,static_u,txt','*VWRITE,du(1,1),du(1,2),du(1,3)','(3E24.16)','*CFCLOS','*CFOPEN,reactions,txt']
    for line in fixed:
        n=int(line.split(',')[1]);lines+=[f'nodeid={n}']
        for j,v in enumerate(('FX','FY','FZ','MX','MY','MZ')):lines += [f'*GET,r{j},NODE,{n},RF,{v}']
        lines+=['*VWRITE,nodeid,r0,r1,r2,r3,r4,r5','(7E24.16)']
    lines+=['*CFCLOS','FINISH','/PREP7','ALLSEL,ALL','*DIM,fsd_area,ARRAY,ne','*DO,fsd_i,1,ne','*GET,fsd_area(fsd_i),ELEM,fsd_i,AREA','*ENDDO','*CFOPEN,actual_areas,txt','*VWRITE,fsd_area(1)','(E24.16)','*CFCLOS','SAVE,diagnostic,db','FINISH','/EXIT,NOSAVE']
    (out/'template.apdl').write_text('\n'.join(lines)+'\n',encoding='ascii')
    report=audit(out/'template.apdl')
    assert not report['fixed_off_root'] and report['fixed_nodes']>0
    assert report['shared_skin_spar_nodes']>0 and report['shared_skin_rib_nodes']>0
    np.savez_compressed(out/'mesh.npz',xyz=xyz,corners=corners,types=types,groups=groups,centers=c,geometric_areas=areas,parent_ids=parent_ids)
    report.update(ribs=ribs,spars=list(spars),surface_groups=int(len(set(groups))),load_transfer=transfer,root_support='Root caisson follows the moving spars',mesh_policy=generator.params)
    report['refinement']='Each standard coarse quad subdivided into four conforming children' if refine else 'none'
    (out/'geometry_summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--ribs',type=int,default=20);p.add_argument('--spars',nargs=2,type=float,default=[.18,.70])
    a=p.parse_args();r=prepare(a.output,a.ribs,a.spars);print(json.dumps({k:r[k] for k in ('nodes','elements','ribs','spars','surface_groups')}))
