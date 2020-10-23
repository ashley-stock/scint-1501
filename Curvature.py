import numpy as np
from astropy.coordinates import SkyCoord
from pint.models import get_model
from pint import toa
from RickettEquations import SystemVel, OrbitMeanVel, RotateVector
from astropy import constants as const

def eta(t_start,t_end,t_nsteps,fitval,par,freq):
	"""
	This function calculates the expected curvature of scintillation arcs.
	Parameters:
		t_start:  The start of the epoch that the curvature is being calculated for in MJD
		t_end: The end of the epoch that the curvature is being calculated for in MJD
		i : the inclination angle of the orbit, float
		s : the fractional distance from the pulsar to the scintillation screen, float
		Oangle : the angle needed to rotate onto x-y plane in radians, float
		PsiAR : the fitted angle of direction of major axis of scintillation screen, float
		VIS : the best fit velocity of the interstellar scintillation screen,
			np array with two floats for x and y velocity
		par : parameter filename, string
		dpsr : distance to the pulsar, float
		freq : frequency of interest, float
	Returns:
		VA : The scintillation velocity parallel to the screen
		eta : the expected curvature of the scintillation arcs as a function of orbital phase 
	"""
	psr_m = get_model(par) # create model of pulsar

	psr = SkyCoord(ra=str(psr_m.RAJ.quantity), dec=str(psr_m.DECJ.quantity), pm_ra_cosdec=psr_m.PMRA.quantity, pm_dec=psr_m.PMDEC.quantity, distance=fitval['dpsr'])

	#Calculate values from binary model for given time frame
	t = toa.make_fake_toas(t_start,t_end,t_nsteps,psr_m,freq=freq,obs="GBT")
	psr_m.delay(t)
	bm = psr_m.binary_instance
	phase = bm.nu()+bm.omega()

	#Calculate scintillation velocity
	VC = SystemVel(t_start,t_end,1,fitval,psr)
	V0 = OrbitMeanVel(psr_m.PB.quantity,psr_m.A1.quantity/psr_m.SINI.quantity,psr_m.ECC.quantity)
	VAx = VC[:,0] - V0*psr_m.ECC.quantity*np.sin(bm.omega()) - V0*np.sin(phase)
	VAy = VC[:,1] + np.cos(fitval['i'])*(V0*psr_m.ECC.quantity*np.cos(bm.omega()) + V0*np.cos(phase))
	VA = RotateVector([VAx,VAy],-fitval['PsiAR'])

	eta = const.c*fitval['dpsr']*fitval['s']/(2*freq*freq*(1-fitval['s'])*VA[0]*VA[0]) 
	VA = [VA[1],-VA[0]]
	return eta, VA
