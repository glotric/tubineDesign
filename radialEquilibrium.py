import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from chartInterpolation import dual_target_residual, AM_1stChartInterpolation, find_min_ypa_for_angle_inAM1stChart
from meanlinePlot import stator_2D_meanline_plot, rotor_2D_meanline_plot
from lossProfiles import secondaryLossProfile, scaleLossProfile, tipClearanceLossProfile

cmap = plt.get_cmap('cool', 8)


# Radial Equilibrium - Part 1: Initialization and Station 0

# Ensure 'data' folder exists
os.makedirs('data', exist_ok=True)
data_dir = 'data'
plot_dir = 'plots'

# Load initial performance summary
perf_df = pd.read_csv('data/TM_performance_summary.csv', header=None)
perf_df[0] = perf_df[0].astype(str).str.strip()  # Ensure keys are strings
perf_dict = dict(zip(perf_df[0], perf_df[1]))

# Helper to access keys flexibly
def get_value(key_hint):
    for key in perf_dict:
        if key_hint.lower() in str(key).lower():
            return float(perf_dict[key])
    raise KeyError(f"'{key_hint}' not found in TM_performance_summary.csv")

# Extract necessary variables
r_mean = get_value('r_mean')
b1_new = get_value('b1')
omega = get_value('omega') * 2 * np.pi / 60
#print('omega = ',omega)
T0 = get_value('T0')
P0 = get_value('P0')
P0t = get_value('P0t')
alfa0 = get_value('alfa0')


# Load gas properties from CSV
gas_props = pd.read_csv(os.path.join(data_dir, 'FG_flue_gas_properties_1700K.csv'))
props = {col: gas_props[col].values[5] for col in gas_props.columns}

cp_mixture = props['cp [kJ/kg.K]']             # [kJ/kg-K]
gamma_mixture = props['gamma']                # [-]
R_mix = props['R [kJ/kg.K]']                  # [kJ/kg-K]

# Discretization
dn = 1000
span = np.linspace(0, 1, dn)
r_prime = np.linspace(r_mean - b1_new / 2, r_mean + b1_new / 2, dn)
U_R_1 = omega * r_prime
alpha0_re = np.full(dn, alfa0)

# Reference conditions
T_ref = np.full(dn, 298.15)
P_ref = np.full(dn, 1.0)  # bar

# Station 0 properties
T0_re = np.full(dn, T0)
P0_re = np.full(dn, P0)
P0t_re = np.full(dn, P0t)

# Entropy at station 0 (kJ/kg-K)
s0_re = cp_mixture * np.log(T0_re / T_ref) - R_mix * np.log(P0_re / P_ref)

# Plot entropy
plt.figure()
plt.plot(s0_re, span, color=cmap(0), linewidth=1.5)
plt.grid(True)
plt.xlabel('Entropy $s_0$ (kJ/kg·K)')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Entropy Distribution at Station 0')
plt.xlim([min(s0_re)*0.98, max(s0_re)*1.02])
plt.savefig('plots/RE1_station0_entropy.png', dpi=300)
plt.close()

# Save initial profiles
pd.DataFrame({
    'r_prime': r_prime,
    'span': span,
    'U_R_1': U_R_1,
    's0_re': s0_re
}).to_csv('data/RE_station0_profiles.csv', index=False)

print("Part 1 complete: Initialization and Station 0 processed and saved.")


# Radial Equilibrium - Part 2: Station 1 Setup and Secondary Losses

# Load previous results
# station0_df = pd.read_csv('data/RE_station0_profiles.csv')
# r_prime = station0_df['r_prime'].values
# span = station0_df['span'].values
# U_R_1 = station0_df['U_R_1'].values
# s0_re = station0_df['s0_re'].values

# Retrieve needed values
V1tang = get_value('V1tang')
V1ax = get_value('V1ax')
T1t = get_value('T1t')
Y_total_stator = get_value('Y_total_stator')
Rho1_new = get_value('rho1')

Y_secondary = get_value('Y_secondary_calculated')

# Blended vortex assumption (a = 0.5)
a = 0.5
V1tang_re = V1tang * (a * r_prime / r_mean + (1 - a) * r_mean / r_prime)

# Mass flow rate at mean radius
mass_flow_in_1 = b1_new * Rho1_new * 2 * np.pi * r_mean * V1ax

# Initial guesses for profiles
V1ax_re = np.full_like(r_prime, V1ax)
T1t_re = np.full_like(r_prime, T1t)
V1_re = np.sqrt(V1ax_re**2 + V1tang_re**2)
alpha1_re = np.degrees(np.arctan2(V1tang_re, V1ax_re))
T1_re = T1t_re - (V1_re**2 / (2 * cp_mixture * 1000))
P1_re = P0 * (T1_re / T0) ** (gamma_mixture / (gamma_mixture - 1))
rho1_re = (P1_re * 1e5) / (R_mix * 1000 * T1_re)
M1_re = V1_re / np.sqrt(gamma_mixture * R_mix * 1000 * T1_re)

# Initial elemental mass flow
elemental_mass_flow_1_init = rho1_re * V1ax_re * 2 * np.pi * r_prime * b1_new

# Secondary loss profile
delta_sh = 0.35
delta_st = 0.35
epsilon_mid = 0.10
normalized_profile_secondary = secondaryLossProfile(span, delta_sh, delta_st, epsilon_mid)

# Plot secondary loss
plt.figure()
plt.plot(normalized_profile_secondary, span, color=cmap(7), linewidth=1.8)
plt.grid(True)
plt.xlabel('Secondary Loss $Y_{secondary}$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Secondary Loss Distribution (Stator)')
plt.xlim([0, 1.05])
plt.ylim([0, 1])
plt.savefig('plots/RE2_secondary_loss_profile_stator.png', dpi=300)
plt.close()

# Distribute Y_other using elemental mass flow
Y_other = Y_total_stator - Y_secondary
distributed_Y_other_stator, _, _ = scaleLossProfile(Y_other, np.ones_like(span), r_prime, elemental_mass_flow_1_init)

# Plot other loss
plt.figure()
plt.plot(distributed_Y_other_stator, span, color=cmap(6), linewidth=1.8)
plt.grid(True)
plt.xlabel('Other Loss $Y_{other}$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Distribution of Other Losses (Stator)')
plt.xlim([0.98*distributed_Y_other_stator[0], 1.02*distributed_Y_other_stator[0]])
plt.ylim([0, 1])
plt.savefig('plots/RE3_other_loss_profile_stator.png', dpi=300)
plt.close()

# Save to CSV
pd.DataFrame({
    'span': span,
    'V1tang_re': V1tang_re,
    'V1ax_re': V1ax_re,
    'T1t_re': T1t_re,
    'T1_re': T1_re,
    'P1_re': P1_re,
    'rho1_re': rho1_re,
    'secondary_loss': normalized_profile_secondary,
    'distributed_Y_other_stator': distributed_Y_other_stator
}).to_csv('data/RE_station1_initial_profiles.csv', index=False)

print("Part 2 complete: Station 1 setup and secondary loss distribution done.")



# Radial Equilibrium - Part 3: Radial equilibrium solver for Station 1 with Loss scaling

# # Load required data
# station0_df = pd.read_csv('data/RE_station0_profiles.csv')
# station1_df = pd.read_csv('data/RE_station1_initial_profiles.csv')

# r_prime = station0_df['r_prime'].values
# span = station0_df['span'].values
# U_R_1 = station0_df['U_R_1'].values
# s0_re = station0_df['s0_re'].values

# # Load station 1 values
# V1ax_re = station1_df['V1ax_re'].values
# V1tang_re = station1_df['V1tang_re'].values
# T1t_re = station1_df['T1t_re'].values
# T1_re = station1_df['T1_re'].values
# P1_re = station1_df['P1_re'].values
# rho1_re = station1_df['rho1_re'].values
# normalized_profile_secondary = station1_df['secondary_loss'].values
# distributed_Y_other_stator = station1_df['distributed_Y_other_stator'].values

# Start iteration
err_mass = np.inf
iteration = 0
max_iter = 50
dn = len(r_prime)

while err_mass > 1e-8 and iteration < max_iter:
    iteration += 1

    elemental_mass_flow = rho1_re * V1ax_re * 2 * np.pi * r_prime * b1_new

    distributed_secondary_loss, _, _ = scaleLossProfile(
        Y_secondary * np.ones(dn), normalized_profile_secondary, r_prime, elemental_mass_flow
    )

    Y_tot_stator_re = distributed_Y_other_stator + distributed_secondary_loss
    P1t_re = (P0t + P1_re * Y_tot_stator_re) / (1 + Y_tot_stator_re)
    #P0t_re = np.full_like(P1t_re, P0t)

    s1_re = s0_re - R_mix * np.log(P1t_re / P0t_re)
    ds_dr = np.gradient(s1_re, r_prime)

    def interp(r, array):
        return np.interp(r, r_prime, array)

    def dVax_dr(r, Vax):
        return -(interp(r, T1_re) / Vax) * interp(r, ds_dr)

    sol = solve_ivp(dVax_dr, [r_prime[0], r_prime[-1]], [V1ax_re[0]], t_eval=r_prime)
    V1ax_re = sol.y[0]

    V1_re = np.sqrt(V1ax_re**2 + V1tang_re**2)
    alpha1_re = np.degrees(np.arctan(V1tang_re / V1ax_re))
    T1_re = T1t_re - V1_re**2 / (2 * cp_mixture * 1000)
    P1_re = P1t_re / (T1t_re / T1_re) ** (gamma_mixture / (gamma_mixture - 1))
    rho1_re = (P1_re * 1e5) / (R_mix * 1000 * T1_re)
    M1_re = V1_re / np.sqrt(gamma_mixture * R_mix * 1000 * T1_re)
    W1ax_re = V1ax_re
    W1tang_re = V1tang_re - U_R_1
    W1_re = np.sqrt(W1ax_re**2 + W1tang_re**2)
    M1rel_re = W1_re / np.sqrt(gamma_mixture * R_mix * 1000 * T1_re)
    P1trel_re = P1_re * (1 + (gamma_mixture-1)/2 * M1rel_re**2)**(gamma_mixture / (gamma_mixture - 1))
    beta1_re = np.degrees(np.arctan(W1tang_re / W1ax_re))

    d_r = r_prime[1] - r_prime[0]
    mass_flow_radial = rho1_re * V1ax_re * 2 * np.pi * r_prime * d_r
    mass_flow_sum = np.sum(mass_flow_radial)

    scale_factor = np.sqrt(mass_flow_in_1 / mass_flow_sum)
    V1ax_re *= scale_factor
    err_mass = abs(mass_flow_sum - mass_flow_in_1) / mass_flow_in_1

# Final output
pd.DataFrame({
    'span': span,
    'V1ax_re': V1ax_re,
    'V1tang_re': V1tang_re,
    'T1_re': T1_re,
    'P1_re': P1_re,
    'rho1_re': rho1_re,
    's1_re': s1_re,
    'Y_tot_stator_re': Y_tot_stator_re
}).to_csv('data/RE_station1_radial_equilibrium.csv', index=False)

print("Part 3 complete: Radial equilibrium solver for Station 1 converged in", iteration, "iterations.")




# Radial Equilibrium - Part 4: Mass-Averaged Quantities at Station 1

# # Load required profile data
# df = pd.read_csv('data/RE_station1_radial_equilibrium.csv')
# rho1_re = df['rho1_re'].values
# V1ax_re = df['V1ax_re'].values
# V1tang_re = df['V1tang_re'].values
# T1_re = df['T1_re'].values
# P1_re = df['P1_re'].values
# s1_re = df['s1_re'].values
# r_prime = pd.read_csv('data/RE_station0_profiles.csv')['r_prime'].values

d_r = r_prime[1] - r_prime[0]
mass_flow_elemental = rho1_re * V1ax_re * 2 * np.pi * r_prime * d_r
mass_flow_total = np.sum(mass_flow_elemental)

# Helper function
def mass_avg(var):
    return np.sum(var * mass_flow_elemental) / mass_flow_total

# Compute mass-averaged values
V1ax_avg = mass_avg(V1ax_re)
V1tang_avg = mass_avg(V1tang_re)
V1_avg = mass_avg(np.sqrt(V1ax_re**2 + V1tang_re**2))
alpha1_avg = mass_avg(np.degrees(np.arctan2(V1tang_re, V1ax_re)))
T1_avg = mass_avg(T1_re)
T1t_avg = mass_avg(T1t_re)
P1_avg = mass_avg(P1_re)
P1t_avg = mass_avg(P1t_re)
rho1_avg = mass_avg(rho1_re)
M1_avg = mass_avg(M1_re)
s1_avg = mass_avg(s1_re)

# Save to CSV
avg_data = {
    'V1ax_avg': V1ax_avg,
    'V1tang_avg': V1tang_avg,
    'V1_avg': V1_avg,
    'alpha1_avg': alpha1_avg,
    'T1_avg': T1_avg,
    'T1t_avg': T1t_avg,
    'P1_avg': P1_avg,
    's1_avg': s1_avg,
    'mass_flow_total': mass_flow_total
}
pd.DataFrame([avg_data]).to_csv('data/RE_station1_mass_averaged.csv', index=False)


# Plot V1tang
plt.figure()
plt.plot(V1tang_re, span, color=cmap(3), linewidth=1.5)
plt.grid(True)
plt.xlabel(r'Tangential Velocity $V_{\theta, 1}(r)$ [m/s]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Blended Vortex Tangential Velocity Profile at Station 1')
plt.xlim([min(V1tang_re)*0.98, max(V1tang_re)*1.02])
plt.savefig('plots/RE4_V1tang_station1.png', dpi=300)
plt.close()

# Plot V1ax
plt.figure()
plt.plot(V1ax_re, span, color=cmap(3), linewidth=1.5)
plt.grid(True)
plt.xlabel('Axial Velocity $V_{ax, 1}(r)$ [m/s]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Axial Velocity Profile at Station 1 $(V_{ax,1})$')
# Force x-axis to show actual values, no offset or scientific notation
ax = plt.gca()
formatter = ScalarFormatter(useMathText=False, useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
plt.savefig('plots/RE7_V1ax_station1.png', dpi=300)
plt.close()


# Plot: Distributed Secondary Loss
plt.figure()
plt.plot(distributed_secondary_loss, span, color=cmap(6), linewidth=1.5)
plt.grid(True)
plt.xlabel('Distributed Secondary Loss $Y_{secondary}$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Final Spanwise Secondary Loss Distribution')
plt.xlim([0, 0.15])
plt.savefig('plots/RE5_secondary_loss_station1.png', dpi=300)
plt.close()

# Plot: Total Loss Profile at Station 1
plt.figure()
plt.plot(Y_tot_stator_re, span, color=cmap(6), linewidth=1.5)
plt.grid(True)
plt.xlabel('Total Loss $Y_{total}$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Final Total Loss Profile Across Span at Station 1')
plt.xlim([0, 0.15])
plt.savefig('plots/RE6_total_loss_station1.png', dpi=300)
plt.close()

# PLot: Entropy Profile
plt.figure()
plt.plot(s1_re, span, color=cmap(0), linewidth=1.5)
plt.grid(True)
plt.xlabel('Entropy $s_1(r)$ [KJ/Kg K]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Entropy distribution at Station 1')
#plt.xlim([min(s1_re)*0.98, max(s1_re)*1.02])
plt.savefig('plots/RE8_entropy_station1.png', dpi=300)
plt.close()


print("Part 4 complete: Mass-averaged quantities for Station 1 saved.")




# Radial Equilibrium - Part 5: Rotor Outlet (Station 2) Initialization and Loss Setup

# # Load Station 1 data
# station1_df = pd.read_csv('data/RE_station1_radial_equilibrium.csv')
# rho1_re = station1_df['rho1_re'].values
# V1tang_re = station1_df['V1tang_re'].values
# T1_re = station1_df['T1_re'].values
# P1_re = station1_df['P1_re'].values
# s1_re = station1_df['s1_re'].values

# # Load geometry and span
# r_prime = pd.read_csv('data/RE_station0_profiles.csv')['r_prime'].values
# span = pd.read_csv('data/RE_station0_profiles.csv')['span'].values
# U_R_1 = pd.read_csv('data/RE_station0_profiles.csv')['U_R_1'].values
# dn = len(span)

# Load station2 results from turbine meanline export
station2_df = pd.read_csv('data/TM_station2_results.csv', index_col=0).squeeze()
station2_losses = pd.read_csv('data/TM_station2_rotor_losses.csv', index_col=0).squeeze()

V2tang = station2_df['V2tang']
V2ax = station2_df['V2ax']
T2t = station2_df['T2t']
b2_final = station2_losses['b2']
rho2_final = station2_losses['rho2']
Y_total_rotor = station2_losses['Y_total_rotor']
Y_secondary_rotor = station2_losses['Y_secondary']
Y_TC = station2_losses['Y_TC']

# Blended vortex for rotor outlet
a = 0.5
V2tang_re = V2tang * (a * r_prime / r_mean + (1 - a) * r_mean / r_prime)
mass_flow_in_2 = b2_final * rho2_final * 2 * np.pi * r_mean * V2ax

V2ax_re = np.full_like(r_prime, V2ax)
T2t_re = np.full_like(r_prime, T2t)
T2trel_re = T2t_re + U_R_1**2 / (2 * cp_mixture * 1000)
V2_re = np.sqrt(V2ax_re**2 + V2tang_re**2)
T2_re = T2t_re - V2_re**2 / (2 * cp_mixture * 1000)
P2_re = P1_re * (T2_re / T1_re)**(gamma_mixture / (gamma_mixture - 1))
rho2_re = (P2_re * 1e5) / (R_mix * 1000 * T2_re)
M2_re = V2_re / np.sqrt(gamma_mixture * R_mix * 1000 * T2_re)
alpha2_re = np.degrees(np.arctan(V2tang_re / V2ax_re))

# Initial elemental mass flow
elemental_mass_flow_2_init = rho2_re * V2ax_re * 2 * np.pi * r_prime * b2_final

# Loss profiles
normalized_profile_secondary_2 = secondaryLossProfile(span, 0.35, 0.35, 0.1)
normalized_profile_Tip_Clearance = tipClearanceLossProfile(span, 0.25, 'gaussian')
Y_other_2 = Y_total_rotor - Y_secondary_rotor - Y_TC
distributed_Y_other_2, _, _ = scaleLossProfile(Y_other_2, np.ones(dn), r_prime, elemental_mass_flow_2_init)

# Save initial Station 2 data
pd.DataFrame({
    'span': span,
    'r_prime': r_prime,
    'V2ax_re': V2ax_re,
    'V2tang_re': V2tang_re,
    'T2t_re': T2t_re,
    'T2trel_re': T2trel_re,
    'V2_re': V2_re,
    'T2_re': T2_re,
    'P2_re': P2_re,
    'rho2_re': rho2_re,
    'secondary_profile': normalized_profile_secondary_2,
    'tip_clearance_profile': normalized_profile_Tip_Clearance,
    'distributed_Y_other_2': distributed_Y_other_2
}).to_csv('data/RE_station2_initial_profiles.csv', index=False)

print("Part 5 complete: Rotor outlet (Station 2) initialized and loss profiles saved.")




# Radial Equilibrium - Part 6: Station 2 Iterative Radial Equilibrium Solver

# # Load Station 2 initial profile
# df2 = pd.read_csv('data/RE_station2_initial_profiles.csv')
# span = df2['span'].values
# r_prime = df2['r_prime'].values
# U_R_1 = pd.read_csv('data/RE_station0_profiles.csv')['U_R_1'].values

# # Inputs from station 2
# V2ax_re = df2['V2ax_re'].values
# V2tang_re = df2['V2tang_re'].values
# T2t_re = df2['T2t_re'].values
# T2trel_re = df2['T2trel_re'].values
# V2_re = df2['V2_re'].values
# T2_re = df2['T2_re'].values
# P2_re = df2['P2_re'].values
# rho2_re = df2['rho2_re'].values
# normalized_profile_secondary_2 = df2['secondary_profile'].values
# normalized_profile_Tip_Clearance = df2['tip_clearance_profile'].values
# distributed_Y_other_2 = df2['distributed_Y_other_2'].values

# # Load values
# P1trel_re = P1t_re * (1 + Y_secondary_rotor)
# #P1t_re = P1trel_re / (1 + Y_secondary_rotor)
# T1t_re = np.full_like(P1trel_re, get_value('T1t'))
# s1_re = pd.read_csv('data/RE_station1_radial_equilibrium.csv')['s1_re'].values

# Iteration loop for station 2
dn = len(r_prime)
err_mass_2 = np.inf
iteration_2 = 0
max_iter = 1000
print('T2', np.average(T2_re))
print('T1t', np.average(T1t_re))
print('T2t', np.average(T2t_re))
print('P1t', np.average(P1t_re))

plt.figure()

while err_mass_2 > 1e-8 and iteration_2 < max_iter:
    iteration_2 += 1
    plt.plot(V2ax_re, span, 'b-', linewidth=1.5, label=f'{iteration_2}', color=cmap(3))
    elemental_mass_flow_2 = rho2_re * V2ax_re * 2 * np.pi * r_prime * b2_final
    #print(sum(elemental_mass_flow_2))

    distributed_secondary_loss_2, _, _ = scaleLossProfile(Y_secondary_rotor * np.ones(dn), normalized_profile_secondary_2, r_prime, elemental_mass_flow_2)

    distributed_tip_loss_2, _, _ = scaleLossProfile(Y_TC * np.ones(dn), normalized_profile_Tip_Clearance, r_prime, elemental_mass_flow_2)

    Y_tot_rotor_re = distributed_Y_other_2 + distributed_secondary_loss_2 + distributed_tip_loss_2

    P2trel_re = (P1trel_re + P2_re * Y_tot_rotor_re) / (1 + Y_tot_rotor_re)
    P2t_re = P2trel_re * (T2t_re / T2trel_re) ** (gamma_mixture / (gamma_mixture - 1))
    #print('P2t', np.average(P2t_re))

    s2_re = s1_re + cp_mixture * np.log(T2t_re / T1t_re) - R_mix * np.log(P2t_re / P1t_re)
    #print('cp:', np.mean(cp_mixture * np.log(T2t_re / T1t_re)))
    #print('R:', np.mean(- R_mix * np.log(P2t_re / P1t_re)))
    ds_dr_2 = np.gradient(s2_re, r_prime)

    # Precompute gradients and interpolators
    Euler_work = U_R_1 * (V1tang_re - V2tang_re) / dn

    # Gradient of Euler work across span
    dEuler_dr = np.gradient(Euler_work, r_prime)

    # Create interpolator functions (with extrapolation enabled)
    interp_T2 = interp1d(r_prime, T2_re, kind='linear', fill_value='extrapolate', bounds_error=False)
    interp_dsdr_2 = interp1d(r_prime, ds_dr_2, kind='linear', fill_value='extrapolate', bounds_error=False)
    interp_dEulerdr = interp1d(r_prime, dEuler_dr, kind='linear', fill_value='extrapolate', bounds_error=False)

    # Final NISRE differential equation
    def dVax_dr_2(r, Vax):
        return (1 / (Vax * cp_mixture * 1000)) * interp_dEulerdr(r) - (interp_T2(r) / Vax) * interp_dsdr_2(r)

    # Integrate with solve_ivp
    #print('V2ax', len(V2ax_re))
    Vax_initial_2 = V2ax_re[0]
    sol2 = solve_ivp(dVax_dr_2, [r_prime[0], r_prime[-1]], [Vax_initial_2], t_eval=r_prime, method='RK45')
    V2ax_re = sol2.y[0]
    #print('V2ax', len(V2ax_re))

    # Euler work
    Euler_local = U_R_1 * (V1tang_re - V2tang_re)
    Euler_energy_flow = Euler_local * elemental_mass_flow_2
    total_Euler_energy = np.sum(Euler_energy_flow)
    total_mass_flow = np.sum(elemental_mass_flow_2)
    delta_h_Euler = total_Euler_energy / total_mass_flow / 1000

    # Physical Model Update
    # s_factor = np.exp((s2_re - s1_re) / cp_mixture)
    # print(np.mean(s_factor))
    # e_factor = np.exp(-delta_h_Euler / (cp_mixture * T1t_re))
    # print(np.mean(e_factor))
    T2t_re = T1t_re * np.exp((s2_re - s1_re) / cp_mixture) * np.exp(-delta_h_Euler / (cp_mixture * T1t_re))
    #print('T2t', np.average(T2t_re))

    # Update properties
    V2_re = np.sqrt(V2ax_re**2 + V2tang_re**2)
    alpha2_re = np.degrees(np.arctan(V2tang_re / V2ax_re))
    T2_re = T2t_re - V2_re**2 / (2 * cp_mixture * 1000)
    #print('T2', np.average(T2_re))
    P2_re = P2t_re / (T2t_re / T2_re) ** (gamma_mixture / (gamma_mixture - 1))
    rho2_re = (P2_re * 1e5) / (R_mix * 1000 * T2_re)
    #print('rho2', np.average(rho2_re))

    M2_re = V2_re / np.sqrt(gamma_mixture * R_mix * 1000 * T2_re)
    W2ax_re = V2ax_re
    W2tang_re = V2tang_re - U_R_1
    W2_re = np.sqrt(W2ax_re**2 + W2tang_re**2)
    beta2_re = np.degrees(np.arctan(W2tang_re / W2ax_re))

    # Mass flow correction
    d_r_2 = r_prime[1] - r_prime[0]
    mass_flow_radial_2 = rho2_re * V2ax_re * 2 * np.pi * r_prime * d_r_2
    mass_flow_sum_2 = np.sum(mass_flow_radial_2)

    scale_factor_2 = np.sqrt(mass_flow_in_2 / mass_flow_sum_2)
    V2ax_re *= scale_factor_2
    err_mass_2 = np.absolute(mass_flow_sum_2 - mass_flow_in_2) / mass_flow_in_2

    print(f"Iteration {iteration_2} | V2ax_avg: {np.mean(V2ax_re):.2f} | T2_re_avg: {np.mean(T2_re):.2f} | T2t_re_avg: {np.mean(T2t_re):.2f} | rho2_min: {np.min(rho2_re):.5f}")

    plt.plot(V2ax_re, span, 'b-', linewidth=1.5, label=f'{iteration_2}', color=cmap(3))

plt.xlabel('Axial Velocity $V_{ax, 2}(r)$ [m/s]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Axial Velocity Profile at Rotor Outlet $(V_{ax, 2})$')
# Force x-axis to show actual values, no offset or scientific notation
ax = plt.gca()
formatter = ScalarFormatter(useMathText=False, useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
#plt.xlim([0.98*np.mean(V2ax_re), 1.02*np.mean(V2ax_re)])
plt.savefig('plots/RE_V2ax_Iterations.png', dpi=300)
plt.close()

# Save results
pd.DataFrame({
    'span': span,
    'V2ax_re': V2ax_re,
    'V2tang_re': V2tang_re,
    'T2_re': T2_re,
    'P2_re': P2_re,
    'rho2_re': rho2_re,
    's2_re': s2_re,
    'Y_tot_rotor_re': Y_tot_rotor_re
}).to_csv('data/RE_station2_radial_equilibrium.csv', index=False)

print("Part 6 complete: Station 2 radial equilibrium converged in", iteration_2, "iterations.")



# Radial Equilibrium - Part 7: Mass-Averaged Quantities at Station 2

# # Load required profile data
# df2 = pd.read_csv('data/RE_station2_radial_equilibrium.csv')
# rho2_re = df2['rho2_re'].values
# V2ax_re = df2['V2ax_re'].values
# V2tang_re = df2['V2tang_re'].values
# T2_re = df2['T2_re'].values
# P2_re = df2['P2_re'].values
# s2_re = df2['s2_re'].values
# r_prime = pd.read_csv('data/RE_station0_profiles.csv')['r_prime'].values

d_r = r_prime[1] - r_prime[0]
mass_flow_elemental = rho2_re * V2ax_re * 2 * np.pi * r_prime * d_r_2
mass_flow_total = np.sum(mass_flow_elemental)

# Helper function
def mass_avg2(var):
    return np.sum(var * mass_flow_elemental) / mass_flow_total

# Compute mass-averaged values
V2ax_avg = mass_avg2(V2ax_re)
V2tang_avg = mass_avg2(V2tang_re)
V2_avg = mass_avg2(np.sqrt(V2ax_re**2 + V2tang_re**2))
alpha2_avg = mass_avg2(np.degrees(np.arctan2(V2tang_re, V2ax_re)))
T2_avg = mass_avg2(T2_re)
P2_avg = mass_avg2(P2_re)
s2_avg = mass_avg2(s2_re)

# Save to CSV
avg_data = {
    'V2ax_avg': V2ax_avg,
    'V2tang_avg': V2tang_avg,
    'V2_avg': V2_avg,
    'alpha2_avg': alpha2_avg,
    'T2_avg': T2_avg,
    'P2_avg': P2_avg,
    's2_avg': s2_avg,
    'mass_flow_total': mass_flow_total
}
pd.DataFrame([avg_data]).to_csv('data/RE_station2_mass_averaged.csv', index=False)

# LOSSES PLOTS
# Plot: Normalized Secondary loss Profile at Rotor Outlet
plt.figure()
plt.plot(normalized_profile_secondary_2, span, color=cmap(7), linewidth=1.5)
plt.grid(True)
plt.xlabel('Normalized Secondary Loss Profile')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Normalized Secondary Loss Profile')
plt.savefig('plots/RE9_NormalizedSecondaryLoss_RotorOutlet.png', dpi=300)
plt.close()

# Plot: Normalized Tip Clearance Loss Profile at Rotor Outlet
plt.figure()
plt.plot(normalized_profile_Tip_Clearance, span, color=cmap(7), linewidth=1.5)
plt.grid(True)
plt.xlabel('Normalized Tip Clearance Loss Profile')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Normalized Tip Clearance Loss Profile')
plt.savefig('plots/RE10_NormalizedTipClearanceLoss_RotorOutlet.png', dpi=300)
plt.close()

# Plot: Distributed Other Losses at Rotor Outlet
plt.figure()
plt.plot(distributed_Y_other_2, span, color=cmap(6), linewidth=1.5)
plt.grid(True)
plt.xlabel('Distributed $Y_{other} (r)$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Distributed Other Losses')
plt.savefig('plots/RE11_DistributedOtherLosses_RotorOutlet.png', dpi=300)
plt.close()

# Plot: Distributed Secondary Losses at Rotor Outlet
plt.figure()
plt.plot(distributed_secondary_loss_2, span, color=cmap(6), linewidth=1.5)
plt.grid(True)
plt.xlabel('Distributed Secondary Loss $Y_{secondary} (r)$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Distributed Secondary Losses')
plt.savefig('plots/RE12_DistributedSecondaryLosses_RotorOutlet.png', dpi=300)
plt.close()

# Plot: Distributed Tip Clearance Losses at Rotor Outlet
plt.figure()
plt.plot(distributed_tip_loss_2, span, color=cmap(6), linewidth=1.5)
plt.grid(True)
plt.xlabel('Distributed Tip Clearance Loss $Y_{TC} (r)$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Distributed Tip Clearance Losses')
plt.savefig('plots/RE13_DistributedTipLosses_RotorOutlet.png', dpi=300)
plt.close()

# Plot: Distributed TOTAL Losses at Rotor Outlet
plt.figure()
plt.plot(Y_tot_rotor_re, span, color=cmap(6), linewidth=1.5)
plt.grid(True)
plt.xlabel('Total Loss $Y_{Total} (r)$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Total Losses Distribution')
plt.savefig('plots/RE14_DistributedTOTAL_RotorOutlet.png', dpi=300)
plt.close()


# OTHER PLOTS
# Plot: Entropy distribution at Rotor Outlet
plt.figure()
plt.plot(s2_re, span, color=cmap(0), linewidth=1.5)
plt.grid(True)
plt.xlabel('Entropy $s_2(r)$ [kJ/kg K]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Entropy Distribution')
plt.savefig('plots/RE15_Entropy_RotorOutlet.png', dpi=300)
plt.close()

# Plot: V2ax distribution at Rotor Outlet
plt.figure()
plt.plot(V2ax_re, span, color=cmap(3), linewidth=1.5)
plt.xlabel('Axial Velocity $V_{ax, 2}(r)$ [m/s]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Axial Velocity Profile at Rotor Outlet $(V_{ax, 2})$')
# Force x-axis to show actual values, no offset or scientific notation
ax = plt.gca()
formatter = ScalarFormatter(useMathText=False, useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.xlim([0.99*np.mean(V2ax_re), 1.01*np.mean(V2ax_re)])
plt.savefig('plots/RE16_V2ax_RotorOutlet.png', dpi=300)
plt.close()

# Plot: T2t distribution at Rotor Outlet
plt.figure()
plt.plot(T2t_re, span, color=cmap(1), linewidth=1.5)
plt.grid(True)
plt.xlabel('Total Temperature $T_{0,2}(r)$ [K]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - Total Temperature Distribution')
plt.savefig('plots/RE17_T2_RotorOutlet.png', dpi=300)
plt.close()

# Plot: P2
plt.figure()
plt.plot(P2_re, span, color=cmap(4), linewidth=1.5)
plt.grid(True)
plt.xlabel('$P_2$ [Pa]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Rotor Outlet - $P_2(r)$')
plt.savefig('plots/RE23_P2_RotorOutlet.png', dpi=300)
plt.close()

# Plot: Pressures
plt.figure()
plt.plot(P2_re, span, color=cmap(4), linewidth=1.5, label='$P_2$')
plt.plot(P1_re, span, color=cmap(3), linewidth=1.5, label='$P_1$')
plt.plot(P0_re, span, color=cmap(2), linewidth=1.5, label='$P_0$')
plt.grid(True)
plt.legend()
plt.xlabel('Pressure [Pa]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Pressure Distribution')
plt.savefig('plots/RE26_Pressures.png', dpi=300)
plt.close()

plt.figure()
plt.plot(P2t_re, span, color=cmap(4), linewidth=1.5, label='$P_{2t}$')
plt.plot(P1t_re, span, color=cmap(3), linewidth=1.5, label='$P_{1t}$')
plt.plot(P0t_re, span, color=cmap(2), linewidth=1.5, label='$P_{0t}$')
plt.grid(True)
plt.legend()
plt.xlabel('Pressure [Pa]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Total Pressure Distribution')
plt.savefig('plots/RE27_PressuresTotal.png', dpi=300)
plt.close()

# Plot: Temperatures
T0t = get_value('T0t')
T0t_re = np.ones_like(span)*T0t
plt.figure()
plt.plot(T2t_re, span, color=cmap(1), linewidth=1.5, label='$T_{2t}$')
plt.plot(T1t_re, span, color=cmap(2), linewidth=1.5, label='$T_{1t}$')
plt.plot(T0t_re, span, color=cmap(3), linewidth=1.5, label='$T_{0t}$')
plt.grid(True)
plt.legend()
plt.xlabel('Temperature [K]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Total Temperature Distribution')
plt.savefig('plots/RE28_TempsTotal.png', dpi=300)
plt.close()

plt.figure()
plt.plot(T2_re, span, color=cmap(1), linewidth=1.5, label='$T_2$')
plt.plot(T1_re, span, color=cmap(2), linewidth=1.5, label='$T_1$')
plt.plot(T0_re, span, color=cmap(3), linewidth=1.5, label='$T_0$')
plt.grid(True)
plt.legend()
plt.xlabel('Temperature [K]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Temperature Distribution')
plt.savefig('plots/RE29_Temps.png', dpi=300)
plt.close()

# Plot: Velocities
V0 = get_value('V0')
V0_re = np.ones_like(span)*V0
plt.figure()
plt.plot(V2_re, span, color=cmap(3), linewidth=1.5, label='$V_2$')
plt.plot(V1_re, span, color=cmap(4), linewidth=1.5, label='$V_1$')
plt.plot(V0_re, span, color=cmap(5), linewidth=1.5, label='$V_0$')
plt.grid(True)
plt.legend()
plt.xlabel('Velocity [m/s]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Absolute Velocity Distribution')
plt.savefig('plots/RE30_V.png', dpi=300)
plt.close()

plt.figure()
plt.plot(W2_re, span, color=cmap(3), linewidth=1.5, label='$W_2$')
plt.plot(W1_re, span, color=cmap(4), linewidth=1.5, label='$W_1$')
plt.grid(True)
plt.legend()
plt.xlabel('Velocity [m/s]')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Relative Velocity Distribution')
plt.savefig('plots/RE31_W.png', dpi=300)
plt.close()

# Plot: Mach number
plt.figure()
plt.plot(M2_re, span, color=cmap(5), linewidth=1.5, label='$M_2$')
plt.plot(M1_re, span, color=cmap(4), linewidth=1.5, label='$M_1$')
plt.grid(True)
plt.legend()
plt.xlabel('mach number')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Absolute Mach number')
plt.savefig('plots/RE32_M.png', dpi=300)
plt.close()


print("Part 7 complete: Mass-averaged quantities for Station 2 saved.")




# Radial Equilibrium - Part 8: Reaction Degree and Stage Efficiency Calculation

# Load required profiles
# df0 = pd.read_csv('data/RE_station0_profiles.csv')
# df2 = pd.read_csv('data/RE_station2_radial_equilibrium.csv')
# r_prime = df0['r_prime'].values
# span = df0['span'].values

total_minus_static = P2t_re - P2_re
entropy_diff = s2_re - s1_re

elemental_mass_flow_2 = rho2_re * V2ax_re * 2 * np.pi * r_prime * b2_final
weighted_T2t = T2t_re * elemental_mass_flow_2
mass_flow_total_2 = sum(elemental_mass_flow_2)
T2t_massavg = sum(weighted_T2t) / mass_flow_total_2

RD1 = 1 - (V1tang_re - V2tang_re) / (2 * U_R_1)
RD2 = (W1tang_re-W2tang_re) / (2 * U_R_1)

# Plot: Reaction Degree at Stator
plt.figure()
plt.plot(RD1, span, color=cmap(2), linewidth=1.5)
plt.grid(True)
plt.xlabel('Stator Reaction Degree $\chi_1$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Distribution of Stator Reaction Degree')
plt.savefig('plots/RE18_RD_Stator.png', dpi=300)
plt.close()

# Plot: Reaction degree at Rotor
plt.figure()
plt.plot(RD2, span, color=cmap(2), linewidth=1.5)
plt.grid(True)
plt.xlabel('Rotor Reaction Degree $\chi_2$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Distribution of Rotor Reaction Degree')
plt.savefig('plots/RE19_RD_Rotor.png', dpi=300)
plt.close()


T0t = get_value('T0t')
P0t = get_value('P0t')

# Isentropic outlet temperature using ideal gas law
T0t_re = np.full_like(span, T0t)
T2t_re_is = T0t_re * (P2t_re / P0t_re) ** ((gamma_mixture - 1) / gamma_mixture)

eta_tt_re = (T2t_re - T0t_re) / (T2t_re_is - T0t_re)

#mass_flow_radial_2 = rho2_re * V2ax_re * 2 * np.pi * r_prime * d_r_2

eta_tt_mass_avg = np.sum(mass_flow_radial_2 * eta_tt_re) / np.sum(mass_flow_radial_2)

# Plot: TT Efficiency
plt.figure()
plt.plot(eta_tt_re, span, color=cmap(5), linewidth=1.5)
plt.grid(True)
plt.xlabel('$\eta_{tt}$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title('Spanwise Distribution of Rotor Total-to-Total Efficiency')
plt.savefig('plots/RE20_etaTT_rotor.png', dpi=300)
plt.close()

# Plot: beta2
plt.figure()
plt.plot(beta2_re, span, color=cmap(5), linewidth=1.5)
plt.grid(True)
plt.xlabel(r'$\beta_2$')
plt.ylabel('Spanwise Location (0 = Hub, 1 = Tip)')
plt.title(r'Spanwise Distribution of $\beta_2$')
plt.savefig('plots/RE33_beta2_rotor.png', dpi=300)
plt.close()

# Save results
pd.DataFrame({
    'span': span,
    'eta_tt_re': eta_tt_re
}).to_csv('data/RE_stage_efficiency_spanwise.csv', index=False)

pd.DataFrame([{'eta_tt_mass_avg': eta_tt_mass_avg}]).to_csv('data/RE_stage_efficiency_mass_averaged.csv', index=False)

print("Part 8 complete: Stage efficiency computed and saved.")



# Radial Equilibrium - Part 9: Stator Blade Geometry Generation (3D Coordinates)

max_thickness = 0.01         # [m]
t_TE_target = 0.001          # [m]
target_LE_TE_chord = 0.04    # [m]
dn = len(span)  # assuming span is defined from station0_profiles.csv
cmap3d = plt.get_cmap('cool', 2*dn)
section_stator = []

x_perc = np.array([0, 1.25, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100])
t_perc = np.array([0, 1.124, 1.571, 2.222, 2.709, 3.111, 3.746, 4.218, 4.824, 5.057, 4.87, 4.151, 3.038, 1.847, 0.749, 0.354, 0.20, 0.15])

fig = plt.figure(figsize=(6,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Line View of Stator Blade Profiles')
ax.set_xlabel('Axial Direction [m]')
ax.set_ylabel('Tangential Direction [m]')
ax.set_zlabel('Spanwise Direction [m]')

for i in range(dn):
    alfa0_i = alpha0_re[i]
    alfa1_i = alpha1_re[i]

    res = minimize_scalar(
        lambda axial_chord: stator_2D_meanline_plot(
            axial_chord, max_thickness, t_TE_target, alfa0_i, alfa1_i,
            x_perc, t_perc, target_LE_TE_chord)[0],
        bounds=(0.01, 0.3), method='bounded', options={'xatol': 1e-6}
    )

    _, _, _, x_ss, y_ss, x_ps, y_ps, _ = stator_2D_meanline_plot(
        res.x, max_thickness, t_TE_target, alfa0_i, alfa1_i,
        x_perc, t_perc, target_LE_TE_chord
    )

    z_val = i * b1_new / (dn - 1)
    ax.plot(x_ss, y_ss, z_val * np.ones_like(x_ss), color=cmap3d(dn+i), alpha=0.5)
    ax.plot(x_ps, y_ps, z_val * np.ones_like(x_ps), color=cmap3d(i), alpha=0.5)

    if i == 0 or i == dn//2 or i == dn-1:
        section_stator.append([x_ss, y_ss, x_ps, y_ps])

ax.set_aspect('equal')
plt.tight_layout()
ax.view_init(elev=30, azim=60)
plt.savefig('plots/RE21A_3D_stator_geometry.png', dpi=300)
ax.view_init(elev=30, azim=-30)
plt.savefig('plots/RE21B_3D_stator_geometry.png', dpi=300)
plt.close()

print("Stator blade geometry plotted in 3D.")

lbls = ['hub', 'meanline', 'tip']
cmapSec = plt.get_cmap('cool', len(section_stator))
plt.figure()
for i in range(len(section_stator)):
    plt.plot(section_stator[i][0], section_stator[i][1], label=lbls[i], color=cmapSec(i), alpha=0.7)
    plt.plot(section_stator[i][2], section_stator[i][3], color=cmapSec(i), alpha=0.7)

plt.axis('equal')
plt.title('Stator Blade Geometry')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid(True)
plt.savefig('plots/RE24_stator_sections.png', dpi=300)
#plt.close()

print("Part 9 complete: Stator Blade geometries Generated.")


# Radial Equilibrium - Part 10: Rotor Blade Geometry Generation (3D Coordinates)

max_thickness_rot = 0.01         # [m]
t_TE_target_rot = 0.001          # [m]
target_LE_TE_chord_rot = 0.04    # [m]
section_rotor = []

x_perc_rot = np.array([0, 1.25, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100])
t_perc_rot = np.array([0, 1.124, 1.571, 2.222, 2.709, 3.111, 3.746, 4.218, 4.824, 5.057, 4.87, 4.151, 3.038, 1.847, 0.749, 0.354, 0.20, 0.15])

fig = plt.figure(figsize=(6,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Line View of Rotor Blade Profiles')
ax.set_xlabel('Axial Direction [m]')
ax.set_ylabel('Tangential Direction [m]')
ax.set_zlabel('Spanwise Direction [m]')

for i in range(dn):
    beta1_i = beta1_re[i]
    beta2_i = beta2_re[i]

    res = minimize_scalar(
        lambda axial_chord: rotor_2D_meanline_plot(
            axial_chord, max_thickness_rot, t_TE_target_rot,
            beta1_i, beta2_i, x_perc_rot, t_perc_rot, target_LE_TE_chord_rot)[0],
        bounds=(0.01, 0.3), method='bounded', options={'xatol': 1e-6}
    )

    _, _, _, x_ss, y_ss, x_ps, y_ps, _ = rotor_2D_meanline_plot(
        res.x, max_thickness_rot, t_TE_target_rot, beta1_i, beta2_i,
        x_perc_rot, t_perc_rot, target_LE_TE_chord_rot
    )

    z_val_rot = i * b2_final / (dn - 1)
    ax.plot(x_ss, y_ss, z_val_rot * np.ones_like(x_ss), color=cmap3d(i), alpha=0.5)
    ax.plot(x_ps, y_ps, z_val_rot * np.ones_like(x_ps), color=cmap3d(dn+i), alpha=0.5)

    if i == 0 or i == dn//2 or i == dn-1:
        section_rotor.append([x_ss, y_ss, x_ps, y_ps])

ax.set_aspect('equal')
plt.tight_layout()
ax.view_init(elev=30, azim=60)
plt.savefig('plots/RE22A_3D_rotor_geometry.png', dpi=300)
ax.view_init(elev=30, azim=-30)
plt.savefig('plots/RE22B_3D_rotor_geometry.png', dpi=300)
#plt.close()

print("Rotor blade geometry plotted in 3D.")

plt.figure()
for i in range(len(section_rotor)):
    plt.plot(section_rotor[i][0], section_rotor[i][1], label=lbls[i], color=cmapSec(i), alpha=0.7)
    plt.plot(section_rotor[i][2], section_rotor[i][3], color=cmapSec(i), alpha=0.7)

plt.axis('equal')
plt.title('Rotor Blade Geometry')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid(True)
plt.savefig('plots/RE25_rotor_sections.png', dpi=300)
plt.close()

print("Part 10 complete: Rotor Blade geometries Generated.")



# Radial Equilibrium - Part 11: Geometruc points Export

# === STATOR GEOMETRIC POINTS EXPORT ===
stator_points = []

for i in range(dn):
    alfa0_i = alpha0_re[i]
    alfa1_i = alpha1_re[i]

    res = minimize_scalar(
        lambda ax: stator_2D_meanline_plot(ax, max_thickness, t_TE_target, alfa0_i, alfa1_i, x_perc, t_perc, target_LE_TE_chord)[0],
        bounds=(0.01, 0.3), method='bounded', options={'xatol': 1e-6})

    _, _, _, x_ss, y_ss, x_ps, y_ps, _ = stator_2D_meanline_plot(
        res.x, max_thickness, t_TE_target, alfa0_i, alfa1_i, x_perc, t_perc, target_LE_TE_chord)

    z_val = i * b1_new / (dn - 1)
    ss_coords = np.column_stack((x_ss, y_ss, np.full_like(x_ss, z_val)))
    ps_coords = np.column_stack((x_ps, y_ps, np.full_like(x_ps, z_val)))

    stator_points.append(ss_coords)
    stator_points.append(ps_coords)
    stator_points.append(np.full((1, 3), np.nan))  # delimiter

stator_array = np.vstack(stator_points)
pd.DataFrame(stator_array).to_csv('data/RE_stator_blade_coordinates.csv', header=False, index=False)
print("Stator blade coordinates exported.")

# === ROTOR GEOMETRIC POINTS EXPORT ===
rotor_points = []

for i in range(dn):
    beta1_i = beta1_re[i]
    beta2_i = beta2_re[i]

    res = minimize_scalar(
        lambda ax: rotor_2D_meanline_plot(ax, max_thickness_rot, t_TE_target_rot, beta1_i, beta2_i, x_perc_rot, t_perc_rot, target_LE_TE_chord_rot)[0],
        bounds=(0.01, 0.3), method='bounded', options={'xatol': 1e-6})

    _, _, _, x_ss, y_ss, x_ps, y_ps, _ = rotor_2D_meanline_plot(
        res.x, max_thickness_rot, t_TE_target_rot, beta1_i, beta2_i, x_perc_rot, t_perc_rot, target_LE_TE_chord_rot)

    z_val = i * b2_final / (dn - 1)
    ss_coords = np.column_stack((x_ss, y_ss, np.full_like(x_ss, z_val)))
    ps_coords = np.column_stack((x_ps, y_ps, np.full_like(x_ps, z_val)))

    rotor_points.append(ss_coords)
    rotor_points.append(ps_coords)
    rotor_points.append(np.full((1, 3), np.nan))  # delimiter

rotor_array = np.vstack(rotor_points)
pd.DataFrame(rotor_array).to_csv('data/RE_rotor_blade_coordinates.csv', header=False, index=False)
print("Rotor blade coordinates exported.")
print("Part 11 complete: Geometries Exported.")


# Radial Equilibrium - Part 12: Velocity Triangles

import matplotlib.pyplot as plt
import numpy as np

def plot_velocity_triangle(ax, Vmag, alpha_deg, Wmag, beta_deg, U, title, i, section):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)

    # Absolute velocity V from origin
    Vx = Vmag * np.cos(alpha)
    Vy = Vmag * np.sin(alpha)

    # Relative velocity W from origin
    Wx = Wmag * np.cos(beta)
    Wy = Wmag * np.sin(beta)

    # Blade speed vector U starts at tip of W
    Ux = U
    Uy = 0

    # Rotate entire system 90 degrees clockwise
    def rotate(x, y):
        return y, -x

    Vx_r, Vy_r = rotate(Vx, Vy)
    Wx_r, Wy_r = rotate(Wx, Wy)
    Ux_r, Uy_r = rotate(Ux, Uy)

    ax.quiver(0, 0, -Wx_r, Wy_r, angles='xy', scale_units='xy', scale=1, color=cmap(i+3), label=section, alpha=0.5)
    ax.quiver(-Wx_r, Wy_r, Uy_r, Ux_r, angles='xy', scale_units='xy', scale=1, color=cmap(i+3), alpha=0.5)
    ax.quiver(0, 0, -Vx_r, Vy_r, angles='xy', scale_units='xy', scale=1, color=cmap(i+3), alpha=0.5)

    all_x = [0, Vx_r, Wx_r, Wx_r + Ux_r]
    all_y = [0, Vy_r, Wy_r, Wy_r + Uy_r]
    max_val = max(max(map(abs, all_x)), max(map(abs, all_y))) * 1.5

    ax.set_xlim(-max_val/2, max_val/2)
    #ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    ax.legend()

def velocity_triangle_3sections(Vmag_vals, alpha_vals, Wmag_vals, U_vals, beta_vals, station_label):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    sections = ['Hub', 'Mean', 'Tip']

    for i, section in enumerate(sections):
        Vmag = Vmag_vals[i]
        alpha = alpha_vals[i]
        Wmag = Wmag_vals[i]
        U = U_vals[i]
        beta = beta_vals[i]
        plot_velocity_triangle(axes, Vmag, alpha, Wmag, beta, U, f'Velocity Triangle at {station_label}', i, section)

    plt.tight_layout()
    plt.savefig(f'plots/RE34_{station_label}_triangles.png')
    plt.show()


# Span indices (hub, meanline, tip)
hub, mid, tip = 0, len(span)//2, -1
indices = [hub, mid, tip]

# Station 1 values
alpha1_vals = alpha1_re[indices]
V1mag_vals = V1_re[indices]
U1_vals = U_R_1[indices]
W1_vals = W1_re[indices]
beta1_vals = beta1_re[indices]

# Station 2 values
alpha2_vals = alpha2_re[indices]
V2mag_vals = V2_re[indices]
U2_vals = U_R_1[indices]
W2_vals = W2_re[indices]
beta2_vals = beta2_re[indices]

velocity_triangle_3sections(V1mag_vals, alpha1_vals, W1_vals, U1_vals, beta1_vals, 'Station1')
velocity_triangle_3sections(V2mag_vals, alpha2_vals, W2_vals, U2_vals, beta2_vals, 'Station2')

print("Part 12 complete: Velocity Triangles Drawn.")