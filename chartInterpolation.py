import numpy as np
from math import pi, sqrt, atan, atan2, tan, cos, sin, radians, degrees

# === Find_min_ypa_for_angle_inAM1stChart ===

def find_min_ypa_for_angle_inAM1stChart(filename, beta2_target):
    """
    Extracts the minimum Yp,a and corresponding s/c from Ainley-Mathieson chart for a given beta2.
    Returns: (closest_beta2, s_c_min, yp_min)
    """
    beta2_blocks = {}
    current_beta = None
    s_c = []
    yp = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('Beta2'):
            if current_beta is not None and s_c:
                beta2_blocks[current_beta] = list(zip(s_c, yp))
            current_beta = int(line.split()[1])
            s_c = []
            yp = []
        elif line and not any(x in line for x in ['s/c', 'Yp,a']):
            try:
                parts = list(map(float, line.split()))
                if len(parts) == 2:
                    s_c.append(parts[0])
                    yp.append(parts[1])
            except ValueError:
                continue

    if current_beta is not None and s_c:
        beta2_blocks[current_beta] = list(zip(s_c, yp))

    closest_beta2 = min(beta2_blocks.keys(), key=lambda x: abs(x - beta2_target))
    data = beta2_blocks[closest_beta2]
    s_values, yp_values = zip(*data)
    min_index = np.argmin(yp_values)
    return closest_beta2, s_values[min_index], yp_values[min_index]


# === Robust AM_1stChartInterpolation Using Parsed Chart ===

def AM_1stChartInterpolation(beta2abs, s2c_ratio, filename):
    import numpy as np
    import re

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {}
    beta = None
    x_vals, y_vals = [], []

    for line in lines:
        if line.startswith('Beta2'):
            if beta is not None and x_vals:
                data[beta] = (np.array(x_vals), np.array(y_vals))
            match = re.search(r'Beta2\s+(\d+)', line)
            beta = int(match.group(1)) if match else None
            x_vals, y_vals = [], []
        elif not any(h in line for h in ['s/c', 'Yp,a']) and beta is not None:
            try:
                parts = list(map(float, line.split()))
                if len(parts) == 2:
                    x_vals.append(parts[0])
                    y_vals.append(parts[1])
            except:
                continue

    if beta is not None and x_vals:
        data[beta] = (np.array(x_vals), np.array(y_vals))

    betas = sorted(data.keys())
    if not betas:
        raise ValueError("No valid data blocks found in Ainley-Mathieson file.")

    if beta2abs <= betas[0]:
        beta_lo, beta_hi = betas[0], betas[1]
    elif beta2abs >= betas[-1]:
        beta_lo, beta_hi = betas[-2], betas[-1]
    else:
        for i in range(len(betas) - 1):
            if betas[i] <= beta2abs <= betas[i + 1]:
                beta_lo, beta_hi = betas[i], betas[i + 1]
                break

    def interp_y(beta_val):
        x, y = data[beta_val]
        return float(np.interp(s2c_ratio, x, y))

    yp_lo = interp_y(beta_lo)
    yp_hi = interp_y(beta_hi)

    return float(np.interp(beta2abs, [beta_lo, beta_hi], [yp_lo, yp_hi]))


# === Interpolator for Ainley-Mathieson 3rd Graph (c/s from beta1, beta2) ===

def ainley_mathieson_3rd_graph(beta1_query, beta2_query, filepath):
    import re
    import numpy as np

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    sections = {}
    beta1_list = []
    current_beta1 = None
    beta2_data = []

    for line in lines:
        if line.lower().startswith("beta1"):
            if current_beta1 is not None and beta2_data:
                beta2_data = np.array(beta2_data)
                sections[current_beta1] = {
                    'beta2': beta2_data[:, 0],
                    'cs': beta2_data[:, 1]
                }
                beta2_data = []
            match = re.search(r'Beta1\s+(\d+)', line, re.IGNORECASE)
            if match:
                current_beta1 = int(match.group(1))
                beta1_list.append(current_beta1)
        elif line.lower().startswith("beta2"):
            continue  # header row, skip
        elif current_beta1 is not None:
            try:
                parts = list(map(float, line.split()))
                if len(parts) == 2:
                    beta2_data.append(parts)
            except ValueError:
                continue  # skip any malformed line

    if current_beta1 is not None and beta2_data:
        beta2_data = np.array(beta2_data)
        sections[current_beta1] = {
            'beta2': beta2_data[:, 0],
            'cs': beta2_data[:, 1]
        }

    beta1_list = sorted(beta1_list)

    if beta1_query < beta1_list[0] or beta1_query > beta1_list[-1]:
        raise ValueError(f"beta1={beta1_query:.2f} out of bounds [{beta1_list[0]}, {beta1_list[-1]}]")

    low_idx = max(i for i in range(len(beta1_list)) if beta1_list[i] <= beta1_query)
    high_idx = min(i for i in range(len(beta1_list)) if beta1_list[i] >= beta1_query)

    b1_low = beta1_list[low_idx]
    b1_high = beta1_list[high_idx]

    def interp_beta2(beta1_val):
        b2_vals = sections[beta1_val]['beta2']
        cs_vals = sections[beta1_val]['cs']
        if beta2_query < min(b2_vals) or beta2_query > max(b2_vals):
            raise ValueError(f"beta2={beta2_query:.2f} out of range for beta1={beta1_val}")
        return np.interp(beta2_query, b2_vals, cs_vals)

    if b1_low == b1_high:
        return interp_beta2(b1_low)
    else:
        cs_low = interp_beta2(b1_low)
        cs_high = interp_beta2(b1_high)
        return np.interp(beta1_query, [b1_low, b1_high], [cs_low, cs_high])


# === Interpolates TEEC values for stator and rotor from a given t2o ratio file ===

def t2o_vs_teec(filename, t2o_ratio):
    """
    Interpolates TEEC values for stator and rotor from a given t2o ratio file.

    Parameters:
        filename (str): Path to the input .txt file with TEEC data.
        t2o_ratio (float): Target t/o ratio to interpolate.

    Returns:
        (TEEC_stator, TEEC_rotor): tuple of interpolated TEEC values.
    """
    stator_t, stator_teec = [], []
    rotor_t, rotor_teec = [], []
    reading_rotor = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if 'stator' in line.lower():
                reading_rotor = False
                continue
            if 'rotor' in line.lower():
                reading_rotor = True
                continue
            if line.lower().startswith('t/o'):
                continue
            try:
                vals = list(map(float, line.split()))
                if len(vals) != 2:
                    continue
                if reading_rotor:
                    rotor_t.append(vals[0])
                    rotor_teec.append(vals[1])
                else:
                    stator_t.append(vals[0])
                    stator_teec.append(vals[1])
            except ValueError:
                continue

    import numpy as np

    def interp_with_check(x_data, y_data, target):
        if not x_data:
            return float('nan')
        if target < min(x_data) or target > max(x_data):
            return float('nan')
        return float(np.interp(target, x_data, y_data))

    TEEC_stator = interp_with_check(stator_t, stator_teec, t2o_ratio)
    TEEC_rotor = interp_with_check(rotor_t, rotor_teec, t2o_ratio)
    return TEEC_stator, TEEC_rotor

#=== Interpolates M1hub2M1 value from file based on given rH2rT_ratio ===

def rh2rtVSM1hub2M1Interpolation(rH2rT_ratio, filename):
    """
    Interpolates M1hub2M1 value from file based on given rH2rT_ratio.

    Parameters:
        rH2rT_ratio (float): Ratio of hub to tip radius.
        filename (str): Path to the text file.

    Returns:
        float: Interpolated M1hub2M1 value.
    """
    x_data = []
    y_data = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or 'rh2rt' in line.lower() or 'M1hub2M1' in line:
                continue
            try:
                vals = list(map(float, line.split()))
                if len(vals) == 2:
                    x_data.append(vals[0])
                    y_data.append(vals[1])
            except ValueError:
                continue

    # Remove duplicates while maintaining order
    from collections import OrderedDict
    filtered = OrderedDict()
    for x, y in zip(x_data, y_data):
        if x not in filtered:
            filtered[x] = y
    x_unique = list(filtered.keys())
    y_unique = list(filtered.values())

    if rH2rT_ratio < min(x_unique) or rH2rT_ratio > max(x_unique):
        raise ValueError(f"rH2rT_ratio ({rH2rT_ratio:.4f}) is out of bounds [{min(x_unique):.4f}, {max(x_unique):.4f}]")

    import numpy as np
    return float(np.interp(rH2rT_ratio, x_unique, y_unique))




# === dual_target_residual ===

def dual_target_residual(V2, p):
    from math import pi, sqrt, tan, atan, atan2, cos, radians, degrees
    import pandas as pd
    import os

    # Velocity
    V2ax = V2
    V2tang = tan(radians(p['alfa0'])) * V2ax
    W2ax = V2ax
    W2tang = V2tang - p['U_mean']
    W2 = sqrt(W2ax**2 + W2tang**2)
    beta2 = degrees(atan2(W2tang, W2ax))
    beta1 = p['beta1']

    # Thermodynamics
    T2t = p['T1t'] - p['L_euler'] / p['cp_mixture']
    T2 = T2t - V2**2 / (2 * p['cp_mixture'] * 1000)
    M2 = V2 / sqrt(p['gamma_mixture'] * p['R_mix'] * 1000 * T2)
    M2rel = W2 / sqrt(p['gamma_mixture'] * p['R_mix'] * 1000 * T2)
    T2trel = T2 * (1 + (p['gamma_mixture'] - 1)/2 * M2rel**2)
    P2 = p['P2t'] / (1 + (p['gamma_mixture'] - 1)/2 * M2**2)**(p['gamma_mixture'] / (p['gamma_mixture'] - 1))

    T2rel = T2t - W2**2 / (2 * p['cp_mixture'] * 1000)
    P2trel = P2 * (1 + (p['gamma_mixture'] - 1)/2 * M2rel**2)**(p['gamma_mixture'] / (p['gamma_mixture'] - 1))
    P2rel = P2trel * (T2rel / T2trel)**(p['gamma_mixture'] / (p['gamma_mixture'] - 1))

    # Ainley-Mathison
    beta2abs = abs(beta2)
    cs_ratio = ainley_mathieson_3rd_graph(beta1, beta2, p['filename_AM3'])
    pitch = p['chord_rotor'] / cs_ratio
    NumBlades = round((2 * pi * p['r_mean']) / pitch)
    pitch_new = 2 * pi * p['r_mean'] / NumBlades
    s2c = pitch_new / p['chord_rotor']

    Yp_a = AM_1stChartInterpolation(beta2abs, s2c, p['filename_AM1'])
    Yp_b = AM_1stChartInterpolation(beta2abs, s2c, p['filename_AM2'])
    m_beta_ratio = abs(beta1 / beta2)
    Yp = (Yp_a + m_beta_ratio**2 * (Yp_b - Yp_a)) * \
         (p['maxBladeThickness_rotor'] / p['chord_rotor'] / 0.2)**m_beta_ratio

    # Mach loss correction
    K2 = (p['M1rel'] / M2rel)**2
    K1 = 1
    if p['M1'] > 0.2:
        K1 = (1 - 1.25*abs(M2rel - 0.2))
    KP = 1 - K2 * (1 - K1)

    # Shock Losses
    rH2rT = p['r_hub'] / (p['r_hub'] + p['b1_new'])
    M1hubrel2M1rel = rh2rtVSM1hub2M1Interpolation(rH2rT, p['filename_rotor_shock_losses'])
    M1hubrel = M1hubrel2M1rel * p['M1rel']
    #print(M1hubrel2M1rel, M1hubrel)
    DP_q1_hub = 0.75 * (M1hubrel - 0.4)**1.75
    Dp_q1_shock = rH2rT * DP_q1_hub

    gamma_power = p['gamma_mixture'] / (p['gamma_mixture'] - 1)
    mach1_func = 1 + (p['gamma_mixture'] - 1)/2 * p['M1rel']**2
    mach2_func = 1 + (p['gamma_mixture'] - 1)/2 * M2rel**2

    Y_shock = Dp_q1_shock * (p['P1rel'] / P2rel) * ((1 - mach1_func**gamma_power) / (1 - mach2_func**gamma_power))

    Y_profile_total = 0.914 * (2/3 * Yp * KP + Y_shock)

    # Reynolds correction
    rho2 = P2 * 1e5 / (p['R_mix'] * 1000 * T2)
    Re = rho2 * V2 * p['chord_rotor'] / p['mu_mix']
    #print(p['chord_rotor'], p['mu_mix'])
    if Re <= 2e5:
        X_Re = (Re / 2e5)**(-0.4)
    elif Re < 1e6:
        X_Re = 1
    else:
        X_Re = (Re / 1e6)**(-0.2)

    # Secondary Losses

    beta1rad = radians(beta1)
    beta2rad = radians(beta2)
    beta_mean = atan(0.5 * (tan(radians(beta1)) + tan(radians(beta2))))
    Cl = 2 * (beta1rad - beta2rad) * cos(beta_mean)
    #print(beta1rad, beta2rad, beta_mean, Cl)

    AR = p['b1_new'] / p['chord_rotor']
    if AR <= 2:
        Func_AR = 1 - 0.25 * sqrt(2 - AR)
    else:
        Func_AR = 1 / AR

    K3 = 1 / AR**2
    print(K3)
    KS = 1 - K3 * (1 - KP)

    Y_secondary = 1.2 * 0.0334 * Func_AR * (cos(radians(beta2)) / cos(radians(beta1))) * \
                  Cl**2 * cos(radians(beta2))**2 / cos(radians(beta_mean))**3 * KS

    t2o = p['te_thickness_rotor'] / p['throat_op_rotor']
    TEEC_stator, TEEC_rotor = t2o_vs_teec(p['filename_tet'], t2o)
    DTEEC = TEEC_stator + (beta1 / beta2)**2 * (TEEC_rotor - TEEC_stator)

    Y_TET = ((1 - (p['gamma_mixture'] - 1)/2 * M2rel**2 * (1 / (1 - DTEEC) - 1))
             **(-gamma_power) - 1) / \
             (1 - (1 + (p['gamma_mixture'] - 1)/2 * M2rel**2)
             **(-gamma_power))

    k_prime = p['tipClearance'] / (p['numSeals'])**0.42
    Y_TC = 0.37 * (p['chord_rotor'] / p['b1_new']) * (k_prime / p['chord_rotor'])**0.78 * \
           Cl**2 * cos(radians(beta2))**2 / cos(radians(beta_mean))**3

    Y_total_rotor = X_Re * Y_profile_total + Y_secondary + Y_TET + Y_TC

    mach2rel = 1 + (p['gamma_mixture'] - 1)/2 * M2rel**2
    mach2 = 1 + (p['gamma_mixture'] - 1)/2 * M2**2

    P2t_calc = p['P1trel'] / (
        (mach2rel / mach2)**gamma_power +
        Y_total_rotor * ((mach2rel / mach2)**gamma_power - mach2**(-gamma_power)))

    b2 = p['mass_flow_total'] / (2 * pi * p['r_mean'] * V2ax * rho2)
    error_P2t = abs(P2t_calc - p['P2t']) / p['P2t']
    error_b2 = abs(b2 - p['b1_new']) / p['b1_new']

    export_dict = {
        'V2': V2,
        'V2ax': V2ax,
        'V2tang': V2tang,
        'W2ax': W2ax,
        'W2tang': W2tang,
        'W2': W2,
        'beta2': beta2,
        'T2t': T2t,
        'T2': T2,
        'M2': M2,
        'M2rel': M2rel,
        'T2trel': T2trel,
        'P2': P2,
        'T2rel': T2rel,
        'P2trel': P2trel,
        'P2rel': P2rel,
        'Yp': Yp,
        'KP': KP,
        'Y_shock': Y_shock,
        'Y_profile_total': Y_profile_total,
        'X_Re': X_Re,
        'Re': Re,
        'Cl': Cl,
        'Y_secondary': Y_secondary,
        'Y_TET': Y_TET,
        'Y_TC': Y_TC,
        'Y_total_rotor': Y_total_rotor,
        'rho2': rho2,
        'error_P2t': error_P2t,
        'error_b2': error_b2,
        'b2': b2,
        'P2t_calculated': P2t_calc,
        'NumBlades_rotor': NumBlades
    }

    pd.Series(export_dict).to_csv(os.path.join('data', 'TM_station2_rotor_losses.csv'))

    residual = 100 * error_P2t**6 + 10 * error_b2**2
    return residual