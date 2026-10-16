"""Resolve external executables without machine-specific installation paths."""
import os
from pathlib import Path
import shutil

def executable(explicit, variable, command):
    value = explicit or os.environ.get(variable) or shutil.which(command)
    if value and Path(value).is_file():
        return str(Path(value).resolve())
    raise FileNotFoundError('Set '+variable+' to the full executable path.')

def ansys_executable():
    return executable(None, 'ANSYS_EXE', 'ansys')

def matlab_executable(explicit=None):
    return executable(explicit, 'MATLAB_EXE', 'matlab')
