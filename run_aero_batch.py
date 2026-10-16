import argparse
import numpy as np
import os
import sys
import time
from pathlib import Path
from matlab_bridge import run_aero

def run_matlab_aero(config_idx, params_dict, target_cy, output_force_file,
                    executable=None, timeout=3600):
    """
    Запускает MATLAB функцию aero_solver для одной конфигурации.
    Возвращает словарь с результатами.
    """
    return run_aero(config_idx, params_dict, target_cy, output_force_file,
                    executable=executable, timeout=timeout)

def main():
    parser = argparse.ArgumentParser(description="Run MATLAB aerodynamics without MATLAB Engine")
    parser.add_argument("file_number", nargs="?", type=int, default=1)
    parser.add_argument("--matlab-exe")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    file_number = args.file_number

    data_file = f"for_ansys_{file_number}.npy"
    if not os.path.exists(data_file):
        print(f"❌ Ошибка: Файл {data_file} не найден.")
        sys.exit(1)

    all_data = np.load(data_file, allow_pickle=False)
    if all_data.ndim != 2 or all_data.shape[0] == 0 or all_data.shape[1] < 15:
        raise ValueError("Configuration array must have rows and at least 15 columns")
    if not np.isfinite(all_data).all():
        raise ValueError("Configuration array contains non-finite values")
    if not Path("A_prof.txt").is_file():
        raise FileNotFoundError("Missing A_prof.txt in the working directory")
    # Never mix a fresh aerodynamic batch with older angle-named load files.
    # Preserve existing results; use a fresh calculation folder for a rerun.
    conflicts = []
    for idx in range(len(all_data)):
        conflicts.extend(Path.cwd().glob(f"{idx}_*.txt"))
        conflicts.extend(Path.cwd().glob(f"config_{idx}_alpha_CL.txt"))
    if Path("aerodata_target.npy").exists():
        conflicts.append(Path("aerodata_target.npy"))
    if conflicts:
        raise FileExistsError("Existing aerodynamic results: use a fresh calculation folder. "
                              + ", ".join(p.name for p in conflicts))
    print(f"✅ Загружено {len(all_data)} конфигураций из {data_file}")

    aerodata_target_list = []

    for idx, row in enumerate(all_data):
        wing_area = float(row[0])
        aspect_ratio = float(row[1])
        taper_ratio = float(row[2])
        sweep_angle = float(row[3])
        thickness_raw = float(row[4])
        flight_speed = float(row[6])
        target_cy = float(row[12])
        air_density = float(row[14])

        # Если нужно округлить входные параметры до 3 знаков, раскомментируйте:
        # wing_area = round(wing_area, 3)
        # aspect_ratio = round(aspect_ratio, 3)
        # taper_ratio = round(taper_ratio, 3)
        # sweep_angle = round(sweep_angle, 3)

        # Округляем толщину профиля до целого процента
        thickness_percent = round(thickness_raw * 100)  # целое число процентов
        thickness = thickness_percent / 100.0           # доля, переданная в расчёт

        print(f"\n{'#'*60}")
        print(f"ОБРАБОТКА КОНФИГУРАЦИИ #{idx}")
        print(f"{'#'*60}")
        print(f"Площадь: {wing_area:.3f} м² | Удлинение: {aspect_ratio:.3f} | Сужение: {taper_ratio:.3f}")
        print(f"Стреловидность: {sweep_angle:.3f}° | Толщина: {thickness_percent}%")
        print(f"Скорость: {flight_speed:.2f} м/с | Плотность: {air_density:.3f} кг/м³")
        print(f"Целевой Cy: {target_cy:.4f}")

        # Формируем словарь параметров для MATLAB (все в СИ)
        params = {
            'wing_area_m2': wing_area,
            'aspect_ratio': aspect_ratio,
            'taper_ratio': taper_ratio,
            'sweep_angle_deg': sweep_angle,
            'thickness': thickness,               # передаём округлённую долю
            'flight_speed_ms': flight_speed,
            'air_density_kgm3': air_density
        }

        temp_force_file = f"temp_forces_{idx}.txt"

        t0 = time.time()
        results = run_matlab_aero(idx, params, target_cy, temp_force_file,
                                  args.matlab_exe, args.timeout)
        elapsed = time.time() - t0

        if results is None:
            print(f"❌ Ошибка при расчёте конфигурации #{idx}")
            continue

        alpha_target = results['alpha_target']
        CL = results['Cy']
        CD = results.get('Cx', 0.0)
        Lift = 0.5 * air_density * flight_speed**2 * wing_area * CL

        print(f"\n--> Результаты (время: {elapsed:.2f} с):")
        print(f"    alpha_target = {alpha_target:.3f}°")
        print(f"    CL = {CL:.4f}, CD = {CD:.4f}, Lift = {Lift:.2f} Н")

        # Сохраняем округлённые до 3 знаков результаты
        aerodata_target_list.append([
            round(alpha_target, 3),
            round(CL, 3),
            round(CD, 3),
            round(Lift, 3)
        ])

        # Переименовываем файл с силами
        final_force = f"{idx}_{alpha_target:.2f}.txt"
        if os.path.exists(temp_force_file):
            os.rename(temp_force_file, final_force)
            print(f"    ✅ Файл сил сохранён как: {final_force}")
        else:
            print(f"    ⚠ Файл {temp_force_file} не найден")

        # Записываем alpha и CL с округлением до 3 знаков
        with open(f"config_{idx}_alpha_CL.txt", 'w') as f:
            f.write(f"{round(alpha_target, 3):.3f} {round(CL, 3):.3f}\n")

    if aerodata_target_list:
        aerodata_target = np.array(aerodata_target_list)
        np.save('aerodata_target.npy', aerodata_target)
        print(f"\n✅ Аэродинамические данные сохранены: aerodata_target.npy (форма {aerodata_target.shape})")
        print("   Столбцы: [alpha_target, CL, CD, Lift] (все округлены до 3 знаков)")

if __name__ == "__main__":
    main()
