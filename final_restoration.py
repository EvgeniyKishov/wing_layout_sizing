"""APDL helpers for final face-stress and positive-load buckling checks.

Uses element-averaged von Mises stress at BOT/MID/TOP of both face layers.
The reinforced core is retained. Every update is rechecked. This is feasibility restoration,
not a minimum-mass optimization or a sandwich-specific failure assessment.
"""
import math


def read_feasibility(path):
    values = [float(v.replace('D', 'E')) for v in path.read_text().split()]
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid final feasibility report')
    flag, stress_ratio, buckling_factor, iteration = values
    if flag != 1 or not 0 < stress_ratio <= 1 or buckling_factor < 1:
        raise ValueError(f'stress ratio={stress_ratio}, buckling factor={buckling_factor}')
    return dict(stress_ratio=stress_ratio, buckling_factor=buckling_factor,
                correction_iterations=int(iteration))


def face_stress_commands():
    code = """
/POST1
SET,LAST
ALLSEL,ALL
"""
    for layer in (1, 3):
        for surface in ('BOT', 'MID', 'TOP'):
            code += f"""
LAYER,{layer}
SHELL,{surface}
ETABLE,ERAS
ETABLE,RR_EQV,S,EQV
*VGET,rr_tmp(1),ELEM,1,ETAB,RR_EQV
*VOPER,rr_stress(1),rr_stress(1),MAX,rr_tmp(1)
"""
    return code + "\nLAYER,0\nSHELL,TOP\nETABLE,ERAS\nFINISH\n"


def positive_buckling_commands():
    # Positive eigenvalues correspond to increasing the current signed load case.
    return """
/SOLU
ANTYPE,BUCKLE
BUCOPT,LANB,5,1e-6,1000,RANGE
MXPAND,5,,,YES
SOLVE
FINISH
/POST1
*DO,rr_mode,1,5
    rr_eig=0
    *GET,rr_eig,MODE,rr_mode,FREQ
    *IF,rr_eig,LE,0,THEN
        rr_valid=0
    *ENDIF
    rr_blf=MIN(rr_blf,rr_eig)
    *CFOPEN,restoration_eigenvalues,txt,,APPEND
    *VWRITE,rr_iter,rr_ls,rr_mode,rr_eig
    (4E24.16)
    *CFCLOS
*ENDDO
FINISH
"""


def restoration_commands(max_iterations=20):
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    code = f"""
! Final feasibility restoration: retain reinforced core, grow faces as needed.
rr_ok=0
*DIM,rr_stress,ARRAY,ne
*DIM,rr_tmp,ARRAY,ne
*CFOPEN,restoration_history,txt
*CFCLOS
*CFOPEN,restoration_eigenvalues,txt
*CFCLOS
*DO,rr_iter,0,{max_iterations}
    *VFILL,rr_stress(1),RAMP,0,0
    rr_blf=1e30
    rr_valid=1
    *DO,rr_ls,1,LS_COUNT
        /SOLU
        LSREAD,rr_ls
        ANTYPE,STATIC
        PSTRES,ON
        SOLVE
        FINISH
{face_stress_commands()}
{positive_buckling_commands()}
    *ENDDO
    *VSCFUN,rr_peak,MAX,rr_stress(1)
    rr_ratio=rr_peak/s_lim
    rr_mass=0
    /PREP7
    *DO,i,1,ne
        *GET,rr_area,ELEM,i,AREA
        rr_mass=rr_mass+rr_area*(2*rho_face*t_face(i)+rho_core*t_core(i))
    *ENDDO
    FINISH
    *CFOPEN,restoration_history,txt,,APPEND
    *VWRITE,rr_iter,rr_mass,rr_ratio,rr_blf
    (4E24.16)
    *CFCLOS
    *IF,rr_valid,NE,1,THEN
        *EXIT
    *ENDIF
    *IF,rr_ratio,LE,1,AND,rr_blf,GE,1,THEN
        rr_ok=1
        *EXIT
    *ENDIF
    *IF,rr_iter,EQ,{max_iterations},THEN
        *EXIT
    *ENDIF
    rr_changed=0
    /PREP7
    *DO,i,1,ne
        rr_oldface=t_face(i)
        rr_oldcore=t_core(i)
        *GET,rr_type,ELEM,i,ATTR,TYPE
        rr_cap=max_total_1
        *IF,rr_type,EQ,2,THEN
            rr_cap=max_total_2
        *ELSEIF,rr_type,EQ,3,THEN
            rr_cap=max_total_3
        *ELSEIF,rr_type,EQ,4,THEN
            rr_cap=max_total_4
        *ENDIF
        ! Only a failing stress check triggers face growth; target 98% allowable.
        *IF,rr_ratio,GT,1,THEN
            rr_scale=MAX(1,MIN(1.5,rr_stress(i)/(0.98*s_lim)))
            t_face(i)=MAX(rr_oldface,MIN(rr_oldface*rr_scale,(rr_cap-t_core(i))/2))
        *ENDIF
        ! Uniform small core growth is a fallback for a failing final buckling check.
        *IF,rr_blf,LT,1,THEN
            rr_scale=MAX(1.02,MIN(1.25,SQRT(1.02/MAX(rr_blf,1e-6))))
            t_core(i)=MAX(t_core(i),MIN(t_core(i)*rr_scale,MIN(BUCK_T_MAX,rr_cap-2*t_face(i))))
        *ENDIF
        total_thick(i)=2*t_face(i)+t_core(i)
        *IF,t_face(i),GT,rr_oldface,OR,t_core(i),GT,rr_oldcore,THEN
            rr_changed=rr_changed+1
        *ENDIF
        SECTYPE,i,SHELL
        SECDATA,t_face(i),1,0,0
        SECDATA,t_core(i),2,0,0
        SECDATA,t_face(i),1,0,1
        *IF,rr_type,EQ,1,THEN
            SECOFFSET,BOT
        *ELSEIF,rr_type,EQ,4,THEN
            SECOFFSET,TOP
        *ELSE
            SECOFFSET,MID
        *ENDIF
        EMODIF,i,SECNUM,i
    *ENDDO
    FINISH
    *IF,rr_changed,EQ,0,THEN
        *EXIT
    *ENDIF
*ENDDO
*CFOPEN,final_feasibility,txt
*VWRITE,rr_ok,rr_ratio,rr_blf,rr_iter
(4E24.16)
*CFCLOS
! Restore static results for ordinary final postprocessing and mass export.
/SOLU
ANTYPE,STATIC
PSTRES,ON
LSSOLVE,1,LS_COUNT
FINISH
"""
    return code
