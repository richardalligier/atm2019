import numpy as np

# unit conversion
FEET2METER = 0.3048
NM2METER = 1852.
KTS2MS = NM2METER/3600.
deg2rad =np.pi/180.

celsius2kelvin= 273.15

# atmospheric constants
beta_T_inf = -0.0065 # K/m
t_trop = 216.65 # K
alt_Hp_trop = 11000. # Hp in m
# Mean Sea Level Standard
t_0 = 288.15 # K
p_0 = 101325. # Pa
rho_0 = 1.225 # kg/m^3
a_0 = 340.294 # m/s

# Physical constants
k = 1.4
big_r = 287.05287 # m2/K.s2 -- real gas constant for air
g_0 = 9.80665 # m/s2
r_earth = 6325766. # m

# auxiliary constants
betaT_R_Over_g = -beta_T_inf*big_r/g_0
g_Over_BetaT_R = 1./betaT_R_Over_g
inv_mu=3.5
mu=1./inv_mu


