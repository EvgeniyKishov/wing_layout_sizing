"""Bridge checks. Set RUN_MATLAB_TESTS=1 for the installed-MATLAB checks.

The integration fixture is deliberately artificial and is not wing validation.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matlab_bridge as bridge


class FileChecks(unittest.TestCase):
    def test_rejects_nonfinite_results(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / "results.tsv"
            result.write_text("Cy\tNaN\n", encoding="ascii")
            with self.assertRaises(ValueError):
                bridge.read_scalars(result)

    def test_reads_exponents_and_signed_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / "results.tsv"
            result.write_text("Cy\t-1.2e-3\nalpha_target\t2.5\n", encoding="ascii")
            self.assertEqual(bridge.read_scalars(result)["Cy"], -0.0012)

    def test_missing_profile_does_not_launch_matlab(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(bridge, "run_script") as launch:
            with self.assertRaises(FileNotFoundError):
                bridge.run_aero(0, {}, 1, "forces.txt", work_dir=folder)
            launch.assert_not_called()

    def test_zero_exit_without_success_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(bridge, "find_matlab", return_value="matlab.exe"):
                with patch.object(bridge.subprocess, "Popen") as process:
                    process.return_value.wait.return_value = 0
                    with self.assertRaisesRegex(RuntimeError, "No success marker"):
                        bridge.run_script("", folder)


@unittest.skipUnless(os.environ.get("RUN_MATLAB_TESTS") == "1", "Installed MATLAB integration checks are opt-in")
class MatlabChecks(unittest.TestCase):
    def test_aerodynamic_interface_fixture(self):
        root = Path(__file__).resolve().parents[1] / "matlab_runs"
        root.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="interface_fixture_", dir=str(root)))
        # Keep logs and fixture files for diagnosis. Exercise spaces and apostrophes.
        folder = folder / "input's folder"
        folder.mkdir()
        (folder / "A_prof.txt").write_text("TEST FIXTURE ONLY", encoding="ascii")
        (folder / "aero_solver.m").write_text(r"""
function results = aero_solver(params, target_cy, output_file)
assert(abs(params.wing_area_m2 - 16.2) < 1e-12);
assert(abs(params.sweep_angle_deg + 2.5) < 1e-12);
results = struct('alpha_target', -1.25, 'Cy', target_cy, 'Cx', 1.2e-3);
fid = fopen(output_file, 'w');
fprintf(fid, '# TEST FIXTURE ONLY\n0.1 0.2 -123.45\n'); fclose(fid);
end
""", encoding="ascii")
        params = dict(zip(bridge.PARAMETERS, [16.2, 7.66, 1.4, -2.5, 0.12, 83.3, 0.9]))
        with patch.object(bridge, "MATLAB_DIR", folder):
            result = bridge.run_aero(0, params, -0.4, "output.txt", timeout=120, work_dir=folder)
        self.assertEqual(result, {"alpha_target": -1.25, "Cy": -0.4, "Cx": 0.0012})
        self.assertIn("-123.45", (folder / "output.txt").read_text())

    def test_matlab_error_reaches_python(self):
        with self.assertRaisesRegex(RuntimeError, "intentional_bridge_test_failure"):
            bridge.run_script("error('intentional_bridge_test_failure');",
                              Path(__file__).resolve().parents[1], timeout=120,
                              label="expected_failure")


if __name__ == "__main__":
    unittest.main()
