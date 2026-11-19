try
cd('C:/Users/Evgeniy/Desktop/AvtomatRep-main/runs/fokker_aero_reproduction_20261013');
bridge_dir = 'C:/Users/Evgeniy/Desktop/AvtomatRep-main/runs/fokker_aero_reproduction_20261013/matlab_runs/config_0_fo9ls1ly';
addpath('C:/Users/Evgeniy/Desktop/AvtomatRep-main/MATLAB');
params = struct();
params.wing_area_m2 = 73;
params.aspect_ratio = 12;
params.taper_ratio = 2.5;
params.sweep_angle_deg = 2;
params.thickness = 0.12;
params.flight_speed_ms = 128;
params.air_density_kgm3 = 0.66000000000000003;
results = aero_solver(params, 1.25, fullfile(bridge_dir, 'forces.txt'));

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

fid = fopen(fullfile(bridge_dir, 'SUCCESS'), 'w');
assert(fid >= 0, 'Cannot write success marker');
fprintf(fid, 'OK'); fclose(fid);
exit(0);
catch err
report = getReport(err, 'extended', 'hyperlinks', 'off');
fprintf(2, '%s\n', report);
fid = fopen('C:/Users/Evgeniy/Desktop/AvtomatRep-main/runs/fokker_aero_reproduction_20261013/matlab_runs/config_0_fo9ls1ly/error.txt', 'w');
if fid >= 0, fprintf(fid, '%s', report); fclose(fid); end
exit(1);
end
