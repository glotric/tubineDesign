# === Stator 2D Meanline Plot with DCA Geometry ===

def stator_2D_meanline_plot(axial_chord, max_thickness, t_TE_target, alfa0, alfa1, x_perc, t_perc, target_LE_TE_chord):
    import numpy as np
    # Generate NACA-like thickness profile
    x_naca = (x_perc / 100) * axial_chord
    t_raw = (t_perc / 100) * axial_chord / 2

    t_TE_half = t_TE_target / 2
    t_max_half = max_thickness / 2
    scale = (t_raw - t_raw[-1]) / (max(t_raw) - t_raw[-1])
    t_m = scale * (t_max_half - t_TE_half) + t_TE_half

    # Camber line as circular arc
    R = axial_chord / (2 * np.sin(np.radians(abs(alfa1 - alfa0) / 2)))
    theta = np.radians(np.linspace(90 - alfa0, 90 - alfa1, 25))
    x_camber = R * np.cos(theta)
    y_camber = R * np.sin(theta) - R
    x_camber -= x_camber[0]
    scale_x = axial_chord / x_camber[-1]
    x_camber *= scale_x
    y_camber = (y_camber - y_camber[0]) * scale_x

    # Thickness to camber points
    t_interp = np.interp(x_camber, x_naca, t_m)

    # Surface normals
    dy_dx = np.gradient(y_camber, x_camber)
    angles = np.arctan(dy_dx)
    nx, ny = -np.sin(angles), np.cos(angles)

    x_ss = x_camber + t_interp * nx
    y_ss = y_camber + t_interp * ny
    x_ps = x_camber - t_interp * nx
    y_ps = y_camber - t_interp * ny

    dx_chord = x_camber[-1] - x_camber[0]
    dy_chord = y_camber[-1] - y_camber[0]
    chord_actual = np.sqrt(dx_chord**2 + dy_chord**2)
    error_val = abs(chord_actual - target_LE_TE_chord)

    return error_val, x_camber, y_camber, x_ss, y_ss, x_ps, y_ps, t_interp


# === Rotor 2D Meanline Plot with DCA Geometry ===

def rotor_2D_meanline_plot(axial_chord, max_thickness, t_TE_target, beta1, beta2, x_perc, t_perc, target_LE_TE_chord):
    import numpy as np
    # Generate NACA-like thickness profile
    x_naca = (x_perc / 100) * axial_chord
    t_raw = (t_perc / 100) * axial_chord / 2

    t_TE_half = t_TE_target / 2
    t_max_half = max_thickness / 2
    scale = (t_raw - t_raw[-1]) / (max(t_raw) - t_raw[-1])
    t_m = scale * (t_max_half - t_TE_half) + t_TE_half

    # Camber line as circular arc
    R = axial_chord / (2 * np.sin(np.radians(abs(beta2 - beta1) / 2)))
    theta = np.radians(np.linspace(90 - beta1, 90 - beta2, 25))
    x_camber = R * np.cos(theta)
    y_camber = R * np.sin(theta) - R
    x_camber -= x_camber[0]
    scale_x = axial_chord / x_camber[-1]
    x_camber *= scale_x
    y_camber = (y_camber - y_camber[0]) * scale_x

    # Thickness to camber points
    t_interp = np.interp(x_camber, x_naca, t_m)

    # Surface normals
    dy_dx = np.gradient(y_camber, x_camber)
    angles = np.arctan(dy_dx)
    nx, ny = -np.sin(angles), np.cos(angles)

    x_ss = x_camber + t_interp * nx
    y_ss = y_camber + t_interp * ny
    x_ps = x_camber - t_interp * nx
    y_ps = y_camber - t_interp * ny

    dx_chord = x_camber[-1] - x_camber[0]
    dy_chord = y_camber[-1] - y_camber[0]
    chord_actual = np.sqrt(dx_chord**2 + dy_chord**2)
    error_val = abs(chord_actual - target_LE_TE_chord)

    return error_val, x_camber, y_camber, x_ss, y_ss, x_ps, y_ps, t_interp
