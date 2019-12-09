import numpy as np
from constants import *

def altHp2pressure(alt_Hp):
    def pressure_below(alt_Hp):
        betaT_Hp=beta_T_inf*np.minimum(alt_Hp,alt_Hp_trop)
        return p_0*(1.+betaT_Hp/t_0)**g_Over_BetaT_R
    def pressure_above(alt_Hp):
        p_trop = pressure_below(alt_Hp)
        c = -g_0/(big_r*t_trop)
        return p_trop*(np.exp(c*(alt_Hp-alt_Hp_trop)))
    is_inf=alt_Hp < alt_Hp_trop
    res=np.zeros(alt_Hp.shape)
    res[is_inf] = pressure_below(alt_Hp[is_inf])
    res[np.logical_not(is_inf)] = pressure_above(alt_Hp[np.logical_not(is_inf)])
    return res

def altHp2TempISA(alt_Hp):
    return t_0+beta_T_inf*np.minimum(alt_Hp,alt_Hp_trop)

def altHpTemp2deltaT(alt_Hp,temp):
    return temp-altHp2TempISA(alt_Hp)

def altHpDeltaT2temp(alt_Hp,deltaT):
    return altHp2TempISA(alt_Hp)+deltaT
    

def rho(pression,temp):
	return pression/(big_r*temp)

#print(altHp2pressure(np.array([80,410])*100*feet2meter))
