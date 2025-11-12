# Cable heat transfer ODE solver
# Input parameters below

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Input
I = 8000                    # A
L = 2                      # m
A_mm2 = 25                 # mm²
A = A_mm2 * 1e-6           # m²
t2 = 0.1                   # seconds

# Physical constants
rho_elec = 1.68e-8         # ohm·m
rho_cu = 8960              # kg/m³
c = 385                   # J/kg°C
T_air = 30                # °C

# Sheath properties
k_sheath = 0.3             # W/m·°C (e.g., PVC or XLPE)
r1 = np.sqrt(A / np.pi)    # inner radius (bare copper)
r2 = 2.2 * r1 + 0.0005     # outer radius (with sheath) approximatrion...
h_conv = 10                # W/m²°C (natural convection)

# Resistance
R = rho_elec * L / A
P = I**2 * R

# Volume, mass
V = A * L
m = rho_cu * V

# Thermal resistance of sheath
R_sheath = np.log(r2 / r1) / (2 * np.pi * k_sheath * L)

# Effective heat transfer coefficient
h_eff = 1 / (1 / h_conv + R_sheath)

# Surface area (outer sheath)
As = 2 * np.pi * r2 * L

# ODE: dT/dt = (P - h_eff*A_s*(T - T_air)) / (m*c)
def dTdt(t, T):
    return (P - h_eff * As * (T - T_air)) / (m * c)

# Adiabatic (no loss) for comparison
def dTdt_adiabatic(t, T):
    return P / (m * c)

# Time grid
t_eval = np.linspace(0, t2, 500)
T0 = [T_air]

# Solve
sol_real = solve_ivp(dTdt, (0, t2), T0, t_eval=t_eval)
sol_adiabatic = solve_ivp(dTdt_adiabatic, (0, t2), T0, t_eval=t_eval)

# Plot
plt.plot(sol_real.t, sol_real.y[0], label='With Heat Loss')
plt.plot(sol_adiabatic.t, sol_adiabatic.y[0], '--', label='Adiabatic (No Loss)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Cable Temperature (%d A, %d mm², %d m)' % (I, A_mm2, L))
plt.grid()
plt.legend()
plt.show()
