# Loss Profile Utilities for Radial Equilibrium
import numpy as np

def secondaryLossProfile(y_prime, delta_sh=0.25, delta_st=0.25, epsilon_mid=0.05):
    """
    Generates a normalized radial profile for secondary losses.
    Peaks at the hub and tip, with a floor value at midspan.
    """
    sigma_h = max(delta_sh / 2.5, 1e-6)
    sigma_t = max(delta_st / 2.5, 1e-6)
    # Gaussian Peaks
    profile_hub = np.exp(-y_prime**2 / (2 * sigma_h**2))
    profile_tip = np.exp(-(1 - y_prime)**2 / (2 * sigma_t**2))
    # Sum and normalizations
    profile_sum = profile_hub + profile_tip
    profile_sum /= np.max(profile_sum)
    return epsilon_mid + (1 - epsilon_mid) * profile_sum

def tipClearanceLossProfile(y_prime, delta_tc=0.10, shape_option='gaussian'):
    """
    Generates a normalized profile for tip clearance losses, peaking at the tip.
    """
    loss_profile_tc = np.zeros_like(y_prime)
    if shape_option.lower() == 'cosine':
        idx = y_prime >= (1 - delta_tc)
        arg_cos = (1 - y_prime[idx]) / delta_tc
        loss_profile_tc[idx] = np.cos(np.pi / 2 * arg_cos) ** 2
    elif shape_option.lower() == 'gaussian':
        sigma = max(delta_tc / 2.5, 1e-6)
        profile_gauss = np.exp(-((1 - y_prime) ** 2) / (2 * sigma ** 2))
        loss_profile_tc = profile_gauss / np.max(profile_gauss)
    else:
        raise ValueError("Invalid shape_option: choose 'cosine' or 'gaussian'")
    return loss_profile_tc



def scaleLossProfile(Y_target, normalized_profile, y_normalized, elemental_mass_flow):
    """
    Parameters:
        Y_target (float): Target average loss coefficient (mass-weighted).
        normalized_profile (ndarray): Profile with shape, peak ~ 1.
        y_normalized (ndarray): Radial coordinates (normalized 0 to 1).
        elemental_mass_flow (ndarray): Mass flow per y element [kg/s].

    Returns:
        distributed_loss (ndarray): Scaled profile across the span.
        scaling_factor (float): Applied scaling factor.
        total_mass_flow (float): Total mass flow used in calculation.
    """
    y_normalized = np.asarray(y_normalized).flatten()
    normalized_profile = np.asarray(normalized_profile).flatten()
    elemental_mass_flow = np.asarray(elemental_mass_flow).flatten()

    # 1. Total mass flow
    total_mass_flow = np.trapz(elemental_mass_flow, y_normalized)

    # 2. Weighted profile integral
    weighted_integrand = normalized_profile * elemental_mass_flow
    weighted_integral = np.trapz(weighted_integrand, y_normalized)

    # 3. Scaling logic
    if np.abs(total_mass_flow) < 1e-9:
        scaling_factor = 0.0
        if np.abs(Y_target) < 1e-9:
            print('WARNING: Total computed mass flow is near zero but Y_target is not. Setting scaling factor to 0.')
    elif np.abs(weighted_integral) < 1e-9:
        if np.abs(Y_target) < 1e-9:
            scaling_factor = 0.0
        else:
            scaling_factor = float('inf')
            print('WARNING: Weighted profile integral is near zero, but Y_target is not. Check inputs.')
    else:
        mean_weighted_profile = weighted_integral / total_mass_flow
        scaling_factor = Y_target / mean_weighted_profile

    # 4. Final scaled profile
    distributed_loss = scaling_factor * normalized_profile
    return distributed_loss, scaling_factor, total_mass_flow
