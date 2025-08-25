import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# -------------------------
# SECTION 1: LOAD DATA
# -------------------------

data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Load flue gas properties using provided syntax
gas_props = pd.read_csv(os.path.join(data_dir, 'FG_flue_gas_properties_1700K.csv'))
props = {col: gas_props[col].values[5] for col in gas_props.columns}

cp_mixture = props['cp [kJ/kg.K]']
gamma_mixture = props['gamma']
R_mix = props['R [kJ/kg.K]']
mu_mix = gas_props.loc[gas_props['Gas'] == 'Viscosity [Pa.s]', 'cp [kJ/kg.K]'].values[0]
mass_flow_air = gas_props.loc[gas_props['Gas'] == 'm_dot_core [kg/s]', 'cp [kJ/kg.K]'].values[0]
mass_flow_fuel = gas_props.loc[gas_props['Gas'] == 'm_dot_fuel [kg/s]', 'cp [kJ/kg.K]'].values[0]

def cp_air(T):
    """
    Calculate specific heat capacity (cp) of air [kJ/kg.K] as a function of temperature T [K],
    using NASA polynomial coefficients for dry air (average composition).

    Source: Polynomial fit similar to what is used for flue gases.
    """
    # Coefficients for air (valid over ~200–2000K), assumed composition
    a1 = 3.725
    a2 = 1.307e-3
    a3 = -4.636e-7
    a4 = 9.222e-11
    a5 = -6.131e-15

    # Universal gas constant [J/mol.K] and molar mass of air [kg/mol]
    R = 8.314510
    M_air = 28.85e-3
    R_specific = R / M_air / 1000  # [kJ/kg.K]

    # cp/R as polynomial of T
    cp_R = a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4

    # Convert to kJ/kg.K
    cp = (cp_R * R / M_air) / 1000
    cv = cp - R_specific

    return cp, cv


# Input parameters
cooling_air = 0.10
P0t = 40  # bar
ExpRatio_tot = 2
P2t = P0t / ExpRatio_tot

mass_flow_total = mass_flow_air / (1 - cooling_air) + mass_flow_fuel

# Load station 0 and 2 mass-averaged data
turbineInputs = pd.read_csv(os.path.join(data_dir, 'TM_turbine_inputs.csv'), header=None, index_col=0).squeeze()
station0 = pd.read_csv(os.path.join(data_dir, 'TM_station0_results.csv'), header=None, index_col=0).squeeze()
station2 = pd.read_csv(os.path.join(data_dir, 'TM_station2_results.csv'), header=None, index_col=0).squeeze()
station1 = pd.read_csv(os.path.join(data_dir, 'TM_station1_results.csv'), header=None, index_col=0).squeeze()

# Extract values
T0t = station0['T0t']
P0t = station0['P0t']
T2t = station2['T2t']
omega = turbineInputs['omega'] # RPM
P_euler_calculated = mass_flow_total * cp_mixture * (T0t-T2t)

# -------------------------
# SECTION 2: COMPRESSOR THERMODYNAMICS
# -------------------------

LHV_fuel = 43e3
cp_a, cv_air = cp_air(T=298)
print(cp_a, cv_air)
gamma_air = cp_a / cv_air
bleed_percentage = 0.1
cp_fuel = 2.0
T_fuel = T_ref = 25 + 273.15
eta_tt_comp = 0.92

mass_flow_air_into_comb = (1 - bleed_percentage) * mass_flow_air
print(mass_flow_air_into_comb, mass_flow_fuel, cp_mixture)
print(T0t, T_ref)
print((mass_flow_air_into_comb + mass_flow_fuel) * cp_mixture * (T0t - T_ref))
print(mass_flow_fuel * (LHV_fuel + cp_fuel * (T_fuel - T_ref)))

Tout_t_comp = (((mass_flow_air_into_comb + mass_flow_fuel) * cp_mixture * (T0t - T_ref) - mass_flow_fuel * (LHV_fuel + cp_fuel * (T_fuel - T_ref))) / (cp_a * mass_flow_air_into_comb)) + T_ref

Pout_t_comp = P0t
Tin_t_comp = Tout_t_comp - (P_euler_calculated / (mass_flow_air * cp_a))
Tout_t_comp_is = eta_tt_comp * (Tout_t_comp - Tin_t_comp) + Tin_t_comp
CR = (Tout_t_comp_is / Tin_t_comp) ** (gamma_air / (gamma_air - 1))
Pin_t_comp = Pout_t_comp / CR

comp_ratio_per_stage = 1.29 # input
N_stages_HPC = math.ceil(math.log(CR) / math.log(comp_ratio_per_stage))
comp_ratio_per_stage_new = CR ** (1 / N_stages_HPC)
P_comp_required = mass_flow_air * cp_a * (Tout_t_comp - Tin_t_comp)

# Save summary
comp_summary = pd.DataFrame({
    'parameter': [
        'Tout_t_comp', 'Pout_t_comp', 'Tin_t_comp', 'CR', 'Pin_t_comp',
        'comp_ratio_per_stage', 'N_stages_HPC', 'comp_ratio_per_stage_new',
        'P_euler_calculated', 'P_comp_required'
    ],
    'value': [
        Tout_t_comp, Pout_t_comp, Tin_t_comp, CR, Pin_t_comp,
        comp_ratio_per_stage, N_stages_HPC, comp_ratio_per_stage_new,
        P_euler_calculated, P_comp_required
    ]
})
comp_summary.to_csv(os.path.join(data_dir, 'CO_comp_summary.csv'), index=False)
print(P_comp_required)
print(P_euler_calculated)

# -------------------------
# SECTION 3: GEOMETRY
# -------------------------

R_air = 287
RD_vavra = 0.5
FC_vavra = 0.5
WC_vavra = 0.25

comp_omega_radians = omega * 2 * math.pi / 60
Ttc_inlet = Tin_t_comp
Tout_t_target = Tout_t_comp
m_dot = mass_flow_air
cp = cp_a

delta_T_total = Tout_t_target - Ttc_inlet
delta_T_per_stage = delta_T_total / N_stages_HPC
L_euler_compressor_perStage = cp * delta_T_per_stage
U_comp = math.sqrt(L_euler_compressor_perStage * 1000 / WC_vavra)
Vax_c = FC_vavra * U_comp
r_mean_c = U_comp / comp_omega_radians

results = []
Ttc = Ttc_inlet
Ptc = Pin_t_comp
b_blades = []

for stage in range(1, N_stages_HPC + 2):
    Ptc_next = Ptc * comp_ratio_per_stage_new
    Ttc_next = Ttc + delta_T_per_stage
    cp, cv = cp_air(0.5*(Ttc_next+Ttc))
    Dht = cp * (Ttc_next - Ttc)
    Tstat = Ttc - (Vax_c ** 2 / (2 * cp * 1000))
    M_axial = Vax_c / math.sqrt(gamma_air * R_air * Tstat)
    Pstat = Ptc / (1 + ((gamma_air - 1)/2) * M_axial**2)**(gamma_air / (gamma_air - 1))
    rho = (Pstat * 1e5) / (R_air * Tstat)
    b_blade = m_dot / (2 * math.pi * r_mean_c * rho * Vax_c)

    b_blades.append(b_blade)
    results.append([stage, Ptc, Ttc, Dht, Tstat, M_axial, Pstat, rho, b_blade])

    Ptc = Ptc_next
    Ttc = Ttc_next

columns = ['Stage', 'Ptc [bar]', 'Ttc [K]', 'Δh [kJ/kg]', 'Tstat [K]',
           'Mach_axial', 'Pstat [bar]', 'rho [kg/m³]', 'Blade H. [m]']
geom_df = pd.DataFrame(results, columns=columns)
print(geom_df)
geom_df.to_csv(os.path.join(data_dir, 'CO_comp_geometry.csv'), index=False)

# -------------------------
# SECTION 4: BLADE TIP SHELL PLOT
# -------------------------

b_blades_mm = np.array(b_blades) * 1000
radii = b_blades_mm/2 + r_mean_c*1000
radii_in = -b_blades_mm/2 + r_mean_c*1000
nStages = len(radii)
nPoints = 100
spacing = 50

theta = np.linspace(0, 2 * np.pi, nPoints)
X, Y, Z = np.zeros((nStages, nPoints)), np.zeros((nStages, nPoints)), np.zeros((nStages, nPoints))
Xin, Yin, Zin = np.zeros((nStages, nPoints)), np.zeros((nStages, nPoints)), np.zeros((nStages, nPoints))

for i in range(nStages):
    X[i, :] = i * spacing
    Y[i, :] = radii[i] * np.cos(theta)
    Z[i, :] = radii[i] * np.sin(theta)
    Xin[i, :] = i * spacing
    Yin[i, :] = radii_in[i] * np.cos(theta)
    Zin[i, :] = radii_in[i] * np.sin(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.7, edgecolor='none', cmap='cool')
ax.plot_surface(Xin, Yin, Zin, alpha=0.7, edgecolor='none', cmap='cool_r')
ax.set_title('Compressor Blade Tip and Hub Shell Surface')
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_aspect('equal')
ax.view_init(elev=20, azim=-25)
plt.tight_layout()
plt.savefig('plots/CO_geometry.png', dpi=300)
plt.show()

plt.figure()
plt.plot(X[:, 0]/35+1, b_blades_mm)
plt.plot(X[:, 0]/35+1, np.zeros_like(b_blades_mm), color='black')
plt.title('Compressor Blade Cascade')
plt.xlabel('Stage')
plt.ylabel('Blade Height [mm]')
plt.grid()
plt.xticks(X[:, 0]/35+1)
plt.savefig('plots/CO_cascade.png', dpi=300)
plt.show()