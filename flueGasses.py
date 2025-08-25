import pandas as pd
import numpy as np

def combustion_model_forProject(m_dot_air, m_dot_fuel):
    M_fuel = 167.3e-3  # kg/mol, Jet-A1 approx. C12H23
    M_air = 28.85e-3   # kg/mol (average)

    n_fuel = m_dot_fuel / M_fuel
    n_air = m_dot_air / M_air

    n_O2_st = 17.75
    mass_air_st_per_mol_fuel = n_O2_st * (1 + 3.727 + 0.0444) * M_air
    AFR_st = mass_air_st_per_mol_fuel / M_fuel
    AFR_actual = m_dot_air / m_dot_fuel
    lambda_ = AFR_actual / AFR_st

    n_O2 = lambda_ * n_O2_st * n_fuel
    n_N2 = n_O2 * 3.727
    n_Ar = n_O2 * 0.0444

    n_CO2 = 12 * n_fuel
    n_H2O = 11.5 * n_fuel
    n_O2_prod = n_O2 - n_O2_st * n_fuel

    M = {"CO2": 0.04401, "H2O": 0.01802, "O2": 0.03200, "N2": 0.02802, "Ar": 0.03995}

    products = {"CO2": n_CO2, "H2O": n_H2O, "O2": n_O2_prod, "N2": n_N2, "Ar": n_Ar}
    total_moles = sum(products.values())
    mole_fractions = {gas: n / total_moles for gas, n in products.items()}
    mass_total = sum(products[gas] * M[gas] for gas in products)
    mass_fractions = {gas: (products[gas] * M[gas]) / mass_total for gas in products}

    return mole_fractions, mass_fractions

def compute_fuel_mass_flow(m_dot_air, excess_air=0.0):
    """
    Calculate fuel mass flow rate from air flow rate and excess air ratio (ε).

    Args:
        m_dot_air (float): mass flow rate of air in kg/s
        excess_air (float): excess air ratio (ε), e.g., 0.1 for 10%

    Returns:
        float: mass flow rate of fuel in kg/s
    """
    # Given ratios in dry air
    X_O2 = 0.21
    X_N2 = 0.78
    X_Ar = 0.01

    # Molar and mass properties
    M_fuel = 167.3e-3  # kg/mol (Jet-A1 ~ C12H23)
    M_air = X_O2 * 32.00e-3 + X_N2 * 28.013e-3 + X_Ar * 39.95e-3  # kg/mol

    # Stoichiometric O2 requirement per mole fuel
    beta_O2_stoich = 17.75  # mol O2 per mol fuel

    # α_stoich = (beta_O2 / X_O2) * (MW_air / MW_fuel)
    alpha_stoich = (beta_O2_stoich / X_O2) * (M_air / M_fuel)

    # Adjust for excess air: α = α_stoich * (1 + ε)
    alpha = alpha_stoich * (1 + excess_air)

    # Fuel mass flow rate: m_dot_fuel = m_dot_air / α
    m_dot_fuel = m_dot_air / alpha
    print(f'beta_O2_stoich = {beta_O2_stoich:.2f}')
    print(f'alpha_stoich = {alpha_stoich:.2f}')
    print(f'alpha = {alpha:.2f}')
    print(f'm_dot_fuel = {m_dot_fuel:.2f} kg/s')
    return m_dot_fuel



# ----------------------------
# INPUT DATA SECTION
# ----------------------------
T3 = 1700  # Temperature in Kelvin
R = 8.314510  # J/mol.K
m_dot = 1200.0  # kg/s
BPR = 10.0
cooling_air = 0.1
m_dot_core = m_dot / (1+BPR) * (1-cooling_air)
m_dot_bypassed = m_dot-m_dot_core

m_dot_fuel = compute_fuel_mass_flow(m_dot_core, excess_air=3.0)
print(m_dot_fuel)

# Gas data: molar mass [kg/mol] and NASA coefficients [a1-a5]
gas_data = {
    "CO2": {"MW": 0.04401, "coeffs": [4.63659493, 2.74131991e-3, -9.95828531e-7, 1.60373011e-10, -9.16103468e-15]},
    "H2O": {"MW": 0.018015, "coeffs": [2.67703787, 2.97318329e-3, -7.73769690e-7, 9.44336689e-11, -4.26900959e-15]},
    "O2":  {"MW": 0.03200, "coeffs": [3.66096083, 6.56365523e-4, -1.41149485e-7, 2.05797658e-11, -1.29913248e-15]},
    "N2":  {"MW": 0.028013, "coeffs": [3.11567530, 1.45886880e-3, -6.01731480e-7, 1.13484230e-10, -7.96585180e-15]},
    "Ar":  {"MW": 0.03995, "coeffs": [2.5, 0, 0, 0, 0]}
}


# ----------------------------
# MAIN PROPERTY CALCULATION
# ----------------------------
mole_fractions, mass_fractions = combustion_model_forProject(m_dot_core, m_dot_fuel)
print(f'mole fractions: {mole_fractions}')
print(f'mass fractions: {mass_fractions}')

cp_mix = cv_mix = 0.0
rows = []

for gas, props in gas_data.items():
    a = props["coeffs"]
    MW = props["MW"]
    cp_R = sum(a[i] * T3**i for i in range(5))
    cp = (cp_R * R / MW) / 1000  # kJ/kg.K
    R_i = R / MW / 1000  # kJ/kg.K
    cv = cp - R_i
    gamma = cp / cv

    cp_mix += mass_fractions[gas] * cp
    cv_mix += mass_fractions[gas] * cv

    rows.append({"Gas": gas, "cp [kJ/kg.K]": cp, "cv [kJ/kg.K]": cv, "gamma": gamma, "R [kJ/kg.K]": R_i})

M_mix = sum(mole_fractions[gas] * gas_data[gas]["MW"] for gas in gas_data)
R_mix = R / M_mix / 1000
gamma_mix = cp_mix / cv_mix

rows.append({"Gas": "Mixture", "cp [kJ/kg.K]": cp_mix, "cv [kJ/kg.K]": cv_mix, "gamma": gamma_mix, "R [kJ/kg.K]": R_mix})


# ----------------------------
# VISCOSITY CALCULATION
# ----------------------------
viscosity_file = "data/TURBOMACHINERY Design project-GasViscosityDatabase.xlsx"
df_mu = pd.read_excel(viscosity_file)

target_gases = ["CO2", "H2O", "O2", "N2", "AR"]
mu0, T0, S = {}, {}, {}

for gas in target_gases:
    row = df_mu[df_mu["Fluido"].str.upper() == gas]
    mu0[gas] = float(row["mu0"])
    T0[gas] = float(row["T0mu"])
    S[gas] = float(row["Smu"])

mu = {}
for gas in target_gases:
    mu[gas] = mu0[gas] * (T3 / T0[gas])**1.5 * (T0[gas] + S[gas]) / (T3 + S[gas])

molar_weights = {"CO2": 0.04401, "H2O": 0.018015, "O2": 0.032, "N2": 0.028013, "AR": 0.03995}
mol_fracs = {"CO2": mole_fractions["CO2"], "H2O": mole_fractions["H2O"], "O2": mole_fractions["O2"], "N2": mole_fractions["N2"], "AR": mole_fractions["Ar"]}

phi = {}
for i in target_gases:
    phi[i] = {}
    for j in target_gases:
        if i == j:
            phi[i][j] = 1
        else:
            mu_i = mu[i]
            mu_j = mu[j]
            Mi = molar_weights[i]
            Mj = molar_weights[j]
            phi[i][j] = (1 + (mu_i / mu_j)**0.5 * (Mj / Mi)**0.25)**2 / (8 * (1 + Mi / Mj))**0.5

mu_mix = 0
for i in target_gases:
    numerator = mol_fracs[i] * mu[i]
    denominator = sum(mol_fracs[j] * phi[i][j] for j in target_gases)
    mu_mix += numerator / denominator

# ----------------------------
# EXPORT TO CSV
# ----------------------------
df = pd.DataFrame(rows)
df.loc[df.shape[0]] = {"Gas": "Viscosity [Pa.s]", "cp [kJ/kg.K]": mu_mix}
df.loc[df.shape[0]] = {"Gas": "m_dot_core [kg/s]", "cp [kJ/kg.K]": m_dot_core}
df.loc[df.shape[0]] = {"Gas": "m_dot_fuel [kg/s]", "cp [kJ/kg.K]": m_dot_fuel}
df.to_csv("data/FG_flue_gas_properties_1700K.csv", index=False)
print(df)
print(mu_mix)