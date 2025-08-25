# turbine_meanline_calc.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import fminbound
from math import pi, sqrt, atan, atan2, tan, cos, sin, radians, degrees

from chartInterpolation import dual_target_residual, AM_1stChartInterpolation, find_min_ypa_for_angle_inAM1stChart
from meanlinePlot import stator_2D_meanline_plot, rotor_2D_meanline_plot

cmap = plt.get_cmap('cool', 3)


# === Section 1: Initialization and Inputs ===

data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Load gas properties from CSV
gas_props = pd.read_csv(os.path.join(data_dir, 'FG_flue_gas_properties_1700K.csv'))
props = {col: gas_props[col].values[5] for col in gas_props.columns}

cp_mixture = props['cp [kJ/kg.K]']            # [kJ/kg-K]
gamma_mixture = props['gamma']                # [-]
R_mix = props['R [kJ/kg.K]']                  # [kJ/kg-K]

# You may need to add these if not in the CSV
mu_mix = gas_props.loc[gas_props['Gas'] == 'Viscosity [Pa.s]', 'cp [kJ/kg.K]'].values[0]
mass_flow_air = gas_props.loc[gas_props['Gas'] == 'm_dot_core [kg/s]', 'cp [kJ/kg.K]'].values[0]
mass_flow_fuel = gas_props.loc[gas_props['Gas'] == 'm_dot_fuel [kg/s]', 'cp [kJ/kg.K]'].values[0]
T3 = 1700         # [K]

# Input parameters
cooling_air = 0.10
T0t = T3
P0t = 40  # bar
ExpRatio_tot = 2
P2t = P0t / ExpRatio_tot

mass_flow_total = mass_flow_air / (1 - cooling_air) + mass_flow_fuel

# Geometry assumptions
clearance = 0.5e-3  # m
chord_stator = 0.035  # m
chord_rotor = 0.035   # m
alfa0 = 0  # deg
omega = 25000  # RPM

# Non-dimensional parameters
work_coeff = 1.3   # [Psi]
flow_coeff = 0.7   # [phi]
eta_tt = 93.3 / 100 #
alfa2 = 0.0         # deg (no swirl at rotor exit) -> V2tang = 0

# Save key inputs
input_dict = {
    'T0t': T0t, 'P0t': P0t, 'P2t': P2t,
    'mass_flow_total': mass_flow_total,
    'omega': omega,
    'work_coeff': work_coeff, 'flow_coeff': flow_coeff, 'eta_tt': eta_tt,
    'cp_mixture': cp_mixture, 'gamma_mixture': gamma_mixture, 'R_mix': R_mix
}
pd.Series(input_dict).to_csv(os.path.join(data_dir, 'TM_turbine_inputs.csv'))


# === Section 2: Station 1 (Rotor Inlet) ===

# Isentropic and actual work
L_is = cp_mixture * T0t * (1 - (P2t / P0t)**((gamma_mixture - 1) / gamma_mixture))
L_euler = eta_tt * L_is

# Mean blade speed and radius
U_mean = sqrt(L_euler * 1000 / work_coeff)
omega_rad = 2 * pi * omega / 60
r_mean = U_mean / omega_rad

# Velocity triangles
V1ax = flow_coeff * U_mean
alfa1 = degrees(atan(work_coeff / flow_coeff))
V1tang = tan(radians(alfa1)) * V1ax
V1 = sqrt(V1ax**2 + V1tang**2)
W1ax = V1ax
W1tang = V1tang - U_mean
W1 = sqrt(W1ax**2 + W1tang**2)
beta1 = degrees(atan(W1tang / W1ax))

# Temperatures and Mach numbers
T1t = T0t
T1 = T1t - (V1**2 / (2 * cp_mixture * 1000))
M1 = V1 / sqrt(gamma_mixture * R_mix * 1000 * T1)
M1rel = W1 / sqrt(gamma_mixture * R_mix * 1000 * T1)
T1trel = T1 * (1 + (gamma_mixture - 1) / 2 * M1rel**2)

# Save station 1 results
station1_dict = {
    'L_is': L_is,
    'L_euler': L_euler,
    'U_mean': U_mean,
    'r_mean': r_mean,
    'omega_rad': omega_rad,
    'V1ax': V1ax,
    'V1tang': V1tang,
    'V1': V1,
    'W1ax': W1ax,
    'W1tang': W1tang,
    'W1': W1,
    'alfa1': alfa1,
    'beta1': beta1,
    'T1t': T1t,
    'T1': T1,
    'M1': M1,
    'M1rel': M1rel,
    'T1trel': T1trel
}
pd.Series(station1_dict).to_csv(os.path.join(data_dir, 'TM_station1_results.csv'))


# === Section 3: Station 1 Losses with Interpolation ===

def station1_losses_with_interpolation(params):
    import numpy as np
    import pandas as pd
    import os

    rho1_guess = params['P0t'] * 1e5 / (params['R_mix'] * 1000 * params['T0t'])
    b1_old = params['mass_flow_total'] / (rho1_guess * params['V1'] * 2 * np.pi * params['r_mean'])

    tol = 1e-10
    error = 1

    results = []

    while error > tol:
        # === Yp from Ainley-Mathieson Min Point ===
        closest_beta, s_c_min, yp_min = find_min_ypa_for_angle_inAM1stChart(
            params['filename_AM1'], params['alfa1']
        )
        pitch = s_c_min * params['chord_stator']
        num_blades = round((2 * np.pi * params['r_mean']) / pitch)
        pitch_new = 2 * np.pi * params['r_mean'] / num_blades
        s2c = pitch_new / params['chord_stator']

        Yp_ref = AM_1stChartInterpolation(params['alfa1'], s2c, params['filename_AM1'])
        Xi = 1.0
        M1 = params['M1']
        Y_profile = (1 + 60 * (M1 - 1)**2) * Xi * Yp_ref

        alfa_mean = np.degrees(np.arctan((np.tan(np.radians(params['alfa0'])) + np.tan(np.radians(params['alfa1']))) / 2))
        C_lift = 2 * s2c * abs(np.tan(np.radians(params['alfa0']) - np.radians(params['alfa1']))) * np.cos(np.radians(alfa_mean))
        Y_secondary = (params['chord_stator'] / b1_old) * 0.0334 * (np.cos(np.radians(params['alfa1'])) / np.cos(np.radians(params['alfa0']))) * \
                      (C_lift / s2c)**2 * (np.cos(np.radians(params['alfa1'])))**2 / (np.cos(np.radians(alfa_mean)))**3

        Re = rho1_guess * params['V1'] * params['chord_stator'] / params['mu_mix']
        X_Re = (Re / 2e5)**(-0.2)

        t_trailing = 0.5e-3
        t2s_ref = 0.02
        X_TE = 1 + 7 * ((t_trailing / pitch_new) - t2s_ref)

        Y_total = Y_profile * X_Re * X_TE + Y_secondary

        gamma = params['gamma_mixture']
        t2s_ratio = (1 + (gamma - 1)/2 * M1**2)**(gamma / (gamma - 1))
        P1t = params['P0t'] / (1 + Y_total - (Y_total / t2s_ratio))
        P1 = P1t / t2s_ratio

        rho1_new = P1 * 1e5 / (params['R_mix'] * 1000 * params['T1'])
        b1_new = params['mass_flow_total'] / (rho1_new * params['V1ax'] * 2 * np.pi * params['r_mean'])

        error = abs(b1_new - b1_old) / b1_old
        b1_old = b1_new
        rho1_guess = rho1_new

        results.append([Y_total, rho1_new, b1_new, error])

    print(f's/c: {s_c_min},   Yp_min: {yp_min},    beta_closest: {closest_beta}')
    final = {
        'Y_total_stator': Y_total,
        'Y_secondary_calculated': Y_secondary,
        'P1t': P1t,
        'P1': P1,
        'rho1': rho1_new,
        'b1': b1_new
    }
    pd.DataFrame(results, columns=['Y_total', 'rho1', 'b1', 'error']).to_csv(os.path.join('data', 'TM_station1_iterations.csv'), index=False)
    return final



# Prepare inputs for interpolation function
station1_params = {
    'P0t': P0t,
    'T0t': T0t,
    'T1': T1,
    'P0': P0t,
    'alfa0': alfa0,
    'alfa1': alfa1,
    'V1': V1,
    'V1ax': V1ax,
    'M1': M1,
    'mass_flow_total': mass_flow_total,
    'r_mean': r_mean,
    'R_mix': R_mix,
    'cp_mixture': cp_mixture,
    'gamma_mixture': gamma_mixture,
    'mu_mix': mu_mix,
    'chord_stator': chord_stator,
    'filename_AM1': os.path.join('data', 'AM_1st_chart_interpolation_Points.txt')
}

# Run iterative station 1 loss calculation with interpolation
station1_losses = station1_losses_with_interpolation(station1_params)

# Save final results and history
pd.Series(station1_losses).to_csv(os.path.join(data_dir, 'TM_station1_losses.csv'))

print("Station 1 loss loop complete.")
print(f"Final b1 = {station1_losses['b1']:.5f} m, P1 = {station1_losses['P1']:.3f} bar")

# Store b1 and rho1 for downstream calculations
b1_new = station1_losses['b1']
Rho1_new = station1_losses['rho1']
P1t = station1_losses['P1t']
P1 = station1_losses['P1']


# === Section 4: Station 0 (Stator Inlet) ===

# Initial guess and reused variables
rho0 = Rho1_new  # from Station 1 final result
b0 = b1_new             # assume constant blade height
V0_old = mass_flow_total / (rho0 * 2 * pi * r_mean * b0)

tolerance = 1e-6
error = 1.0
error_rho = 1
iteration = 0

station0_log = []

while error_rho > tolerance:
    iteration += 1

    # Temperature from energy balance
    T0 = T0t - V0_old**2 / (2 * cp_mixture * 1000)
    M0 = V0_old / sqrt(gamma_mixture * R_mix * 1000 * T0)
    P0 = P0t / (1 + (gamma_mixture - 1) * 0.5 * M0**2)**(gamma_mixture / (gamma_mixture - 1))
    rho0_new = P0 * 1e5 / (R_mix * 1000 * T0)

    V0_new = mass_flow_total / (rho0_new * 2 * pi * r_mean * b0)
    error = abs((V0_new - V0_old) / V0_old)
    error_rho = abs((rho0_new - rho0) / rho0)

    station0_log.append([iteration, V0_new, T0, M0, P0, rho0_new, error])

    V0_old = V0_new
    rho0 = rho0_new

# Save final Station 0 values
station0_data = {
    'V0': V0_new,
    'T0': T0,
    'M0': M0,
    'P0': P0,
    'rho0': rho0_new,
    'T0t': T0t,
    'P0t': P0t
}
pd.Series(station0_data).to_csv(os.path.join(data_dir, 'TM_station0_results.csv'))

# Save iteration history
station0_df = pd.DataFrame(station0_log, columns=[
    'Iteration', 'V0', 'T0', 'M0', 'P0', 'rho0', 'Error'])
station0_df.to_csv(os.path.join(data_dir, 'TM_station0_iterations.csv'), index=False)



# === Section 5: Station 2 (Rotor Outlet Optimization) Refactored ===

# Compute relative temperature and pressures at station 1
T2t = T1t - L_euler / cp_mixture
T1rel = T1t - W1**2 / (2 * cp_mixture * 1000)
P1trel = P1 * (1 + ((gamma_mixture - 1) / 2) * M1rel**2) ** (gamma_mixture / (gamma_mixture - 1))
P1rel = P1trel * (T1rel / T1trel) ** (gamma_mixture / (gamma_mixture - 1))

# Assemble all necessary parameters for rotor outlet loss model
params = {
    'T1t': T1t,
    'L_euler': L_euler,
    'cp_mixture': cp_mixture,
    'gamma_mixture': gamma_mixture,
    'R_mix': R_mix,
    'P2t': P2t,
    'r_mean': r_mean,
    'b1_new': b1_new,
    'alfa0': alfa0,
    'U_mean': U_mean,
    'mu_mix': mu_mix,
    'mass_flow_total': mass_flow_total,
    'chord_rotor': chord_rotor,
    'beta1': beta1,
    'M1': M1,
    'M1rel': M1rel,
    'P1trel': P1trel,
    'P1rel': P1rel,
    'te_thickness_rotor': chord_rotor * 0.025,
    'throat_op_rotor': chord_rotor * 0.2,
    'numSeals': 3,
    'tipClearance': 0.5e-3,
    'maxBladeThickness_rotor': 0.01,
    'filename_AM1': os.path.join(data_dir, 'AM_1st_chart_interpolation_Points.txt'),
    'filename_AM2': os.path.join(data_dir, 'AM_2nd_chart_interpolation_Points.txt'),
    'filename_AM3': os.path.join(data_dir, 'ainley mathieson 3rd graph interpolation points.txt'),
    'filename_rotor_shock_losses': os.path.join(data_dir, 'rh2rtVSM1hub2M1_KackerOcapuu_Rotor.txt'),
    'filename_tet': os.path.join(data_dir, 't2oVsTEEC.txt'),
    'r_hub': 0.120
}



# === Section 6: Solve Rotor Outlet Velocity with Dual Target Residual ===

print("\nStarting optimization for rotor outlet axial velocity V2...")
V2_opt = fminbound(lambda V2: dual_target_residual(V2, params), 160.9, 526.7)

# Load computed losses and velocities from CSV output generated by residual function
loss_file = os.path.join(data_dir, 'TM_station2_rotor_losses.csv')
station2_rotor = pd.read_csv(loss_file, index_col=0).squeeze()

# Final values to export and report
station2_data = {
    'V2_opt': V2_opt,
    'T2t': T2t,
    'V2ax': station2_rotor['V2ax'],
    'V2tang': station2_rotor['V2tang'],
    'b2': station2_rotor['b2'],
    'Y_total_rotor': station2_rotor['Y_total_rotor'],
    'Y_secondary': station2_rotor['Y_secondary'],
    'Y_TC': station2_rotor['Y_TC'],
    'rho2': station2_rotor['rho2']
}

pd.Series(station2_data).to_csv(os.path.join(data_dir, 'TM_station2_results.csv'))

print("Rotor outlet optimization complete.")
print(f"Optimal axial velocity V2 = {V2_opt:.3f} m/s")



# === Section 7: Blade Geometry Generation ===

import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from meanlinePlot import stator_2D_meanline_plot, rotor_2D_meanline_plot

# Geometry profile parameters
x_perc = np.array([0, 1.25, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100])
t_perc = np.array([0, 1.124, 1.571, 2.222, 2.709, 3.111, 3.746, 4.218,
                   4.824, 5.057, 4.87, 4.151, 3.038, 1.847,
                   0.749, 0.354, 0.20, 0.15])

# Stator Geometry
max_thickness = 0.01
TE_target = 0.001
chord_target = 0.035

stator_error_func = lambda axial_chord: stator_2D_meanline_plot(
    axial_chord, max_thickness, TE_target, alfa0, alfa1,
    x_perc, t_perc, chord_target)[0]

axial_chord_stator = fminbound(stator_error_func, 0.01, 0.2, xtol=1e-10)
_, x_camber_s, y_camber_s, x_ss_s, y_ss_s, x_ps_s, y_ps_s, t_s = stator_2D_meanline_plot(axial_chord_stator, max_thickness, TE_target, alfa0, alfa1, x_perc, t_perc, chord_target)

# Rotor Geometry
beta2 = station2_rotor['beta2']
rotor_error_func = lambda axial_chord: rotor_2D_meanline_plot(
    axial_chord, max_thickness, TE_target, beta1, beta2, x_perc, t_perc, chord_target)[0]

axial_chord_rotor = fminbound(rotor_error_func, 0.01, 0.2, xtol=1e-10)
_, x_camber_r, y_camber_r, x_ss_r, y_ss_r, x_ps_r, y_ps_r, t_r = rotor_2D_meanline_plot(
    axial_chord_rotor, max_thickness, TE_target, beta1, beta2, x_perc, t_perc, chord_target)

# === Plotting Profiles ===
os.makedirs('plots', exist_ok=True)

# Stator plot
plt.figure(figsize=(8,4))
plt.plot(x_ss_s, y_ss_s, color=cmap(2), label='Suction Side')
plt.plot(x_ps_s, y_ps_s, color=cmap(0), label='Pressure Side')
plt.plot(x_camber_s, y_camber_s, 'k--', label='Camber')
plt.axis('equal')
plt.title('Stator Blade Geometry')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid(True)
plt.savefig('plots/TM_stator_blade_geometry.png')
plt.close()

# Rotor plot
plt.figure(figsize=(8,4))
plt.plot(x_ss_r, y_ss_r, color=cmap(0), label='Suction Side')
plt.plot(x_ps_r, y_ps_r, color=cmap(2), label='Pressure Side')
plt.plot(x_camber_r, y_camber_r, 'k--', label='Camber')
plt.axis('equal')
plt.title('Rotor Blade Geometry')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid(True)
plt.savefig('plots/TM_rotor_blade_geometry.png')
plt.close()

print("Stator and rotor geometries generated and saved to 'plots' folder.")




# === Section 8: Final Data Export and Performance Summary ===

summary = {
    'omega (RPM)': omega,
    'L_is (kJ/kg)': station1_dict['L_is'],
    'L_euler (kJ/kg)': station1_dict['L_euler'],
    'U_mean (m/s)': station1_dict['U_mean'],
    'r_mean (m)': station1_dict['r_mean'],
    'b1 (m)': b1_new,
    'V0 (m/s)': V0_new,
    'T0 (K)': T0,
    'P0 (bar)': P0,
    'rho0 (kg/m^3)': rho0_new,
    'V1 (m/s)': station1_dict['V1'],
    'V1ax (m/s)': station1_dict['V1ax'],
    'V1tang (m/s)': station1_dict['V1tang'],
    'T1 (K)': station1_dict['T1'],
    'T1t (K)': station1_dict['T1t'],
    'P1 (bar)': station1_losses['P1'],
    'rho1 (kg/m^3)': station1_losses['rho1'],
    'Y_total_stator': station1_losses['Y_total_stator'],
    'beta1 (deg)': station1_dict['beta1'],
    'alfa0 (deg)': alfa0,
    'V2opt (m/s)': station2_data['V2_opt'],
    'V2ax (m/s)': station2_data['V2ax'],
    'V2tang (m/s)': station2_data['V2tang'],
    'b2 (m)': station2_data['b2'],
    'T2 (K)': station2_data['T2'],
    'T2t (K)': station2_data['T2t'],
    'rho2 (kg/m^3)': station2_data['rho2'],
    'Y_total_rotor': station2_data['Y_total_rotor'],
    'Y_secondary_rotor': station2_data['Y_secondary'],
    'Y_Secondary_calculated': station1_losses['Y_secondary_calculated'],
    'Y_TC': station2_data['Y_TC'],
    'T0t (K)': T0t,
    'P0t (bar)': P0t,
    'P2t (bar)': P2t,
    'Expansion Ratio': P0t / P2t,
    'Mass Flow Total (kg/s)': mass_flow_total,
    'Work Coeff (ψ)': work_coeff,
    'Flow Coeff (φ)': flow_coeff,
    'Efficiency (η_tt)': eta_tt
}

# Export to CSV
summary_path = os.path.join(data_dir, 'TM_performance_summary.csv')
pd.Series(summary).to_csv(summary_path)

print(f"\nPerformance summary written to: {summary_path}")
print("\nGenerated files:")
for file in sorted(os.listdir(data_dir)):
    print("-", file)
for file in sorted(os.listdir('plots')):
    print("- plots/", file)