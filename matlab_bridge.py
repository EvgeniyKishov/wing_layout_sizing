"""File-based MATLAB bridge; no MATLAB Engine or Python packages needed."""

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


MATLAB_DIR = Path(__file__).resolve().parent / "MATLAB"
PARAMETERS = (
    "wing_area_m2", "aspect_ratio", "taper_ratio", "sweep_angle_deg",
    "thickness", "flight_speed_ms", "air_density_kgm3",
)


def matlab_string(value):
    """Quote a path/string for MATLAB, including paths containing apostrophes."""
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def find_matlab(executable=None):
    from runtime_config import matlab_executable
    return matlab_executable(executable)


def read_scalars(path):
    values = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, value = line.split("\t")
        number = float(value)
        if not math.isfinite(number) or key in values:
            raise ValueError("Invalid or duplicate MATLAB result: " + key)
        values[key] = number
    return values


def run_script(body, work_dir, executable=None, timeout=600, label="run"):
    """Run in an isolated log directory; require explicit success from MATLAB."""
    if timeout <= 0:
        raise ValueError("MATLAB timeout must be positive")
    executable = find_matlab(executable)
    work_dir = Path(work_dir).resolve()
    runs = work_dir / "matlab_runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=label + "_", dir=str(runs)))
    wrapper = run_dir / "bridge_run.m"
    # MATLAB has no jsonencode or -batch. Use a script, text files and -r.
    wrapper.write_text(
        "try\n"
        + "cd(" + matlab_string(work_dir) + ");\n"
        + "bridge_dir = " + matlab_string(run_dir) + ";\n"
        + body + "\n"
        + "fid = fopen(fullfile(bridge_dir, 'SUCCESS'), 'w');\n"
        + "assert(fid >= 0, 'Cannot write success marker');\n"
        + "fprintf(fid, 'OK'); fclose(fid);\nexit(0);\n"
        + "catch err\n"
        + "report = getReport(err, 'extended', 'hyperlinks', 'off');\n"
        + "fprintf(2, '%s\\n', report);\n"
        + "fid = fopen(" + matlab_string(run_dir / "error.txt") + ", 'w');\n"
        + "if fid >= 0, fprintf(fid, '%s', report); fclose(fid); end\n"
        + "exit(1);\nend\n",
        encoding="utf-8",
    )
    command = [executable, "-nosplash", "-nodesktop"]
    options = {}
    if os.name == "nt":
        command += ["-wait", "-minimize"]
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options["startupinfo"] = startup
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    command += ["-logfile", str(run_dir / "matlab.log"),
                "-r", "run(" + matlab_string(wrapper) + ")"]
    print("MATLAB logs: " + str(run_dir), flush=True)
    with (run_dir / "launcher.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(work_dir), stdout=log,
                                   stderr=subprocess.STDOUT, **options)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.kill()
            process.wait()
            raise TimeoutError("MATLAB timed out; logs: " + str(run_dir)) from exc
    if code != 0 or not (run_dir / "SUCCESS").is_file():
        error = run_dir / "error.txt"
        detail = error.read_text(encoding="utf-8", errors="replace") if error.exists() else "No success marker"
        raise RuntimeError("MATLAB failed (exit %s); logs: %s\n%s" % (code, run_dir, detail))
    return run_dir


def run_aero(config_idx, params, target_cy, output_force_file,
             executable=None, timeout=3600, work_dir=None):
    work_dir = Path(work_dir or Path.cwd()).resolve()
    if not (work_dir / "A_prof.txt").is_file():
        raise FileNotFoundError("Missing aerodynamic profile: " + str(work_dir / "A_prof.txt"))
    numbers = {key: float(params[key]) for key in PARAMETERS}
    numbers["target_cy"] = float(target_cy)
    if not all(math.isfinite(v) for v in numbers.values()):
        raise ValueError("Aerodynamic inputs must be finite")
    for key in ("wing_area_m2", "aspect_ratio", "taper_ratio", "thickness",
                "flight_speed_ms", "air_density_kgm3"):
        if numbers[key] <= 0:
            raise ValueError(key + " must be positive")
    body = "addpath(" + matlab_string(MATLAB_DIR) + ");\nparams = struct();\n"
    body += "\n".join("params.%s = %.17g;" % (key, numbers[key]) for key in PARAMETERS)
    body += "\nresults = aero_solver(params, %.17g, fullfile(bridge_dir, 'forces.txt'));\n" % numbers["target_cy"]
    body += r"""
fid = fopen(fullfile(bridge_dir, 'results.tsv'), 'w');
assert(fid >= 0, 'Cannot write aerodynamic results');
names = fieldnames(results);
for k = 1:numel(names)
    val = results.(names{k});
    if isnumeric(val) && isscalar(val)
        fprintf(fid, '%s\t%.17g\n', names{k}, val);
    end
end
fclose(fid);
"""
    run_dir = run_script(body, work_dir, executable, timeout, "config_%s" % config_idx)
    results = read_scalars(run_dir / "results.tsv")
    for key in ("alpha_target", "Cy", "Cx"):
        if key not in results:
            raise RuntimeError("Missing result %s; logs: %s" % (key, run_dir))
    forces = run_dir / "forces.txt"
    count = 0
    for line in forces.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3 or not all(math.isfinite(float(v)) for v in fields):
            raise RuntimeError("Invalid force row; logs: " + str(run_dir))
        count += 1
    if not count:
        raise RuntimeError("MATLAB exported no force rows; logs: " + str(run_dir))
    output = Path(output_force_file)
    if not output.is_absolute():
        output = work_dir / output
    shutil.copy2(forces, output)
    return results


def smoke_test(executable=None, timeout=120, work_dir=None):
    body = r"""
A = [3 1; 1 2]; b = [9; 8]; x = A\b;
assert(norm(A*x-b) < 1e-12, 'Linear solve check failed');
fid = fopen(fullfile(bridge_dir, 'results.tsv'), 'w');
fprintf(fid, 'x1\t%.17g\nx2\t%.17g\n', x(1), x(2)); fclose(fid);
fid = fopen(fullfile(bridge_dir, 'version.txt'), 'w');
fprintf(fid, '%s', version); fclose(fid);
"""
    directory = run_script(body, work_dir or Path.cwd(), executable, timeout, "smoke")
    values = read_scalars(directory / "results.tsv")
    if abs(values["x1"] - 2) > 1e-12 or abs(values["x2"] - 3) > 1e-12:
        raise RuntimeError("Unexpected smoke-test result")
    print("Bridge smoke test passed: " + (directory / "version.txt").read_text())
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", required=True)
    parser.add_argument("--matlab-exe")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    smoke_test(args.matlab_exe, args.timeout)
