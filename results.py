def write_results_to_file(calibrated_params, file_path="results.py"):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Cherche la définition de la classe Results
    updated_lines = []
    in_results_class = False
    for line in lines:
        if line.strip().startswith("class Results"):
            in_results_class = True
        if in_results_class and "calibration_results" in line:
            # Remplace la valeur de calibration_results
            updated_lines.append(f"    calibration_results = {calibrated_params}\n")
            in_results_class = False  # Fin de la modification
        else:
            updated_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

class Results:
    calibration_results = {'kappa': 1.997593209388519, 'v0': 0.09799082270480705, 'theta': 0.11170597551131412, 'sigma': 0.2064711617898025, 'rho': -0.5164318949237379}
