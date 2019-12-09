import numpy as np
from constants import *
import atmos
from atmos import *

two_over_mu= 2./mu

#temp K
#tas m/s
#return mach [-]
def tas2mach(temp,tas):
	return tas /np.sqrt(k*big_r*temp)

#temp K
#tas m/s
#return mach [-]
def mach2tas(temp,mach):
	return mach*np.sqrt(k*big_r*temp)

#altHp m
#temp K
#cas m/s
#return tas m/s
def cas2tas(altHp,temp,cas):
	coeff= rho_0/(two_over_mu*p_0)
	p=altHp2pressure(altHp)
	rho=atmos.rho(p,temp)
	a= (1.+coeff*cas*cas)**inv_mu-1.
	b= 1.+ p_0*a/p
	c= b**mu -1
	return np.sqrt(two_over_mu*c*p/rho)

#altHp m
#temp K
#tas m/s
#return cas m/s
def tas2cas(altHp,temp,tas):
	coeff= rho_0/(two_over_mu*p_0)
	p=altHp2pressure(altHp)
	rho=atmos.rho(p,temp)
	a= (1.+rho*tas*tas/(two_over_mu*p))**inv_mu-1
	b= 1.+p*a/p_0
	c= b**mu -1.
	return np.sqrt(c/coeff)

#cas m/s
#mach -
#return altconj m
def cas_mach_crossover_altitude(cas,mach):
	num = 1.+(k-1.)/2.*(cas/a_0)**2.
	den = 1.+(k-1.)/2.*mach**2.
	delta =(num**inv_mu-1.)/(den**inv_mu-1.)
	temp_0= t_0
	coeff = 1./(FEET2METER*(-beta_T_inf))
	alti_Hp = coeff *temp_0*(1.-(delta**(betaT_R_Over_g)))
	return(alti_Hp*FEET2METER)


def cas_altitude_crossover_mach(cas,alti_Hp):
	num = 1.+(k-1.)/2.*(cas/a_0)**2.
	temp_0= t_0
	coeff = 1./(FEET2METER*(-beta_T_inf))
	delta = (1.-alti_Hp/coeff/temp_0)**(1/betaT_R_Over_g)
	den=((num**inv_mu-1.)/delta+1)**mu
	mach=sqrt((den-1)*2/(k-1))
	return mach

#cas m/s
#mach -
#return tas m/s
def tasFromCasMach(cas,mach,altHp,temp):
#cas=traj.CasCible.values
#	cas=cas*KTS2MS
	casna= cas != cas
	machna= mach!=mach
	casnotna= np.logical_not(casna)
	machnotna= np.logical_not(machna)
	res=np.zeros(cas.shape)
	sel=np.logical_and(casna,machnotna)
	res[sel]=mach2tas(temp[sel],mach[sel])
	sel=np.logical_and(casnotna,machna)
	res[sel]=cas2tas(altHp[sel],temp[sel],cas[sel])
	selcasmach=np.logical_and(casnotna,machnotna)
	undercross = altHp < cas_mach_crossover_altitude(cas,mach)
	selcas = np.logical_and(undercross,selcasmach)
	selmach = np.logical_and(np.logical_not(undercross),selcasmach)
	res[selcas] = cas2tas(altHp[selcas],temp[selcas],cas[selcas])
	res[selmach] = mach2tas(temp[selmach],mach[selmach])
	return res

def tasFromCasCasMach(cas1,cas2,mach,altHp,temp):
	cas1na= cas1 != cas1
	cas1notna= np.logical_not(cas1na)
	res=np.zeros(cas1.shape)
	altinf=altHp<10000*FEET2METER
	res[cas1na]=tasFromCasMach(cas2[cas1na],mach[cas1na],altHp[cas1na],temp[cas1na])
	sel=np.logical_and(cas1notna,np.logical_not(altinf))
	res[sel]=tasFromCasMach(cas2[sel],mach[sel],altHp[sel],temp[sel])
	sel=np.logical_and(cas1notna,altinf)
	res[sel]=cas2tas(altHp[sel],temp[sel],cas1[sel])
	return res










