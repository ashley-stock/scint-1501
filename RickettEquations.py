import numpy as np
import math
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyOffsetFrame, SkyCoord
from pint.models import get_model
from astropy.time import Time
from astropy import constants as const


"""TODO:
-Make functions so they have default values
"""

#-------------------------------------------------------------------------------------
#Fitted values from Table 3 of Rickett et al 2014
#-------------------------------------------------------------------------------------
i = 88.7*u.deg
s = 0.71
Oangle = 61*u.deg
R = 0.71
PsiAR = 61*u.deg
VIS = np.array([-9,42])*u.km/u.s
s0 = 4.2e6*u.m
#--------------------------------------------------------------------------------------
# Other constant values
#--------------------------------------------------------------------------------------
par = "J0737-3039A.par" #parameter file from ATNF
t_start = 52997 #52997
t_end = 53561 #53561
t_nsteps = 564 #564
lt_s = u.Unit('lt_s', u.lightyear / u.yr * u.s)

#---------------------------------------------------------------------------------------

def Q_coeff(R,Psi_AR):
	"""This function calculates the quadratic coefficients which
	describe the ISS anisotropy as defined in equation 4 of Rickett et al. 2014
	Parameters:
		R: a bounded parameter related to the axial ratio, with range 0 to 1
		where 0 describes a circle and 1 describes a line
		Psi_AR: angle describing orientation of the major axis in radians
	Returns:
		[a,b,c]: an array of quadratic coefficients, floats
	"""

	a = (1-R*np.cos(2*Psi_AR))/np.sqrt(1-R*R)
	b = (1+R*np.cos(2*Psi_AR))/np.sqrt(1-R*R)
	c = -2*R*np.sin(2*Psi_AR)/np.sqrt(1-R*R)
	return np.array([a,b,c])

def OrbitMeanVel(PB,SMA,ECC):
	"""This function calculates the mean orbital velocity of the pulsar.
	Parameters:
		PB: the orbital period of the pulsar (units time), float
		SMA: the semi major axis (units length), float
		ECC: the orbital eccentricity (unitless)
	Returns:
		V0: the mean orbital velocity in km/s, float
	"""
	return (2*math.pi*SMA/(PB*np.sqrt(1-ECC*ECC))).to(u.km/u.s)

def SpatialScale(s0=s0,s=s):
	"""This function calculates the unitless spatial scale in the pulsar frame.
	Parameters:
		s0: the mean diffractive scale (units of length), float
		s: the fractional distance from the pulsar to the scintillation screen (unitless), float
	Returns:
		sp: the spatial scale in the pulsar frame, float
	"""
	return s0/(1-s)

def EarthVelocity(t,site,psr,rot):
	"""This function gets the proper earth velocity in RA-DEC coordinates for data taken
	from a site relative to the sun, but in the pulsar frame.
	Parameters:
		t: the time of the observation, astropy time format
		site: the name of the observatory where data was taken, a string
		psr: a SkyCoordinate object representation of the pulsar
	Returns:
		VE: a two element np array which gives earth velocity in RA and DEC directions 
	"""
	psr_frame = SkyOffsetFrame(origin=psr, rotation=rot)

	tel = EarthLocation.of_site(site)
	pos = tel.get_gcrs(t).transform_to(psr_frame).cartesian
	vel = pos.differentials['s']
	return vel.d_xyz.to(u.km/u.s)

def PulsarBCVelocity(psr):
	"""This function calculates and returns the proper motion of the barycentre
	of the pulsar binary system.	
	Parameters:
		psr: a SkyCoordinate object representation of the pulsar 
	Returns:
		pm_psr: 3D pulsar barycentre proper motion
	"""
	psr_frame = SkyOffsetFrame(origin=psr, rotation=0*u.deg) 
	#if I put this into the pulsar frame it seems that dy and dz are the Valpha and Vdelta
	pm_psr = psr.transform_to(psr_frame).cartesian.differentials['s']
	return pm_psr
	

def RotateVector(v,angle):
	"""This function rotates a vector from RA-DEC coordinates to pulsar orbital frame coordinates.
	If you are using it generally, note that this 
	Parameters:
		v: a np array with proper motions in km/s in order (PM_RA,PM_DEC), float
		angle: the angle needed to rotate onto x-y plane in radians, float
	Returns:
		[Vx,Vy]: a np array which is the rotated version of the vector
	"""
	new_v = [np.sin(angle)*v[0] + np.cos(angle)*v[1],-np.cos(angle)*v[0]+np.sin(angle)*v[1]]
	return [new_v[0].value, new_v[1].value]*u.km/u.s


def SystemVel(t_start,t_end,t_nsteps,fitval,psr):
	"""This function calculates the system velocity in the pulsar frame
	as defined in equation 6 of Rickett et al. 2014.
	Parameters:
		t_start : initial time of observation, integer (for now)
		t_end : last time of observation, integer (for now)
		s : the fractional distance from the pulsar to the scintillation screen, float
		Oangle: the angle needed to rotate onto x-y plane in radians, float
		psr : a SkyCoordinate object representation of the pulsar
		VIS : the best fit velocity of the interstellar scintillation screen,
			np array with two floats for x and y velocity
	Returns:
		VC: np array with two floats, representing x and y tranverse system velocity
	"""
	times = Time(np.array(range(t_start,t_end)), format = 'mjd')
	
	#Calculate Earth velocity
	VE = np.ones((t_nsteps,2))
	i=0
	for t in times:
		VE[i]=RotateVector(EarthVelocity(t,'gbt',psr,0*u.deg)[1:3],fitval['Oangle'])
		i +=1

	#Calculate pulsar velocity
	VP = np.array([-17.8,11.6])*u.km/u.s #Just use values from paper for now
	VP = RotateVector(VP,fitval['Oangle'])#rotate pulsar velocity by Oangle

	#Otherwise
	#VP = PulsarBCVelocity(psr)[1:3] 

	VC = np.ones((t_nsteps,2))
	i = 0
	for v in VE:
		VC[i] = np.add(np.add(VP,v*u.km/u.s*s/(1-s)),-fitval['VIS']/(1-fitval['s']))
		i +=1
	return np.array(VC)*u.km/u.s

def K_coeffs(t_start, fitval, psr, psr_m):
	"""This function calculates the orbital harmonic coefficients for the scintillation
	timescale as defined in equation 10 of Rickett et al. 2014.
	Parameters:
		V0: the mean orbital velocity of the binary pulsar system, float
		VC: related to scintillation velocity (eq.6), array of two floats [VCX,VCY]
		Qabc: quadratic coefficients (eq. 4), array of three floats [a,b,c]
		i : inclination angle in radians, float
		omega: longitude of periastron in radians, float
		ecc: eccentricity of binary pulsar orbit, float
		sp: scintillation spatial scale, float
	Returns:
		[K0,KS,KC,KS2,KC2]: an array of orbital harmonic coefficients, floats
	"""

	SMA = (psr_m.A1.quantity/psr_m.SINI.quantity) #convert projected semi major axis to actual value
	V0 = OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity)
	VC = SystemVel(t_start,t_start+1,1,fitval,psr)[0]
	Qabc = Q_coeff(fitval['R'],fitval['PsiAR'])
	omega = psr_m.OM.quantity
	ecc = psr_m.ECC.quantity
	sp = fitval['s0']/(1-fitval['s'])
	i = fitval['i']

	K0 = 0.5*V0*V0*(Qabc[0]+Qabc[1]*(np.cos(i))**2) + Qabc[0]*(VC[0]-V0*ecc*np.sin(omega))**2
	K0 = K0 + Qabc[1]*(VC[1]+V0*ecc*np.cos(omega)*np.cos(i))**2
	K0 = K0 + Qabc[2]*(VC[0]-V0*ecc*np.sin(omega))*(VC[1]+V0*ecc*np.cos(omega)*np.cos(i))
	K0 = K0.to(u.m*u.m/(u.s*u.s))

	KS = -V0*(2*Qabc[0]*(VC[0] - V0*ecc*np.sin(omega)) + Qabc[2]*(VC[1]+V0*ecc*np.cos(i)*np.cos(omega)))
	KS = KS.to(u.m*u.m/(u.s*u.s))

	KC = V0*np.cos(i)*(Qabc[2]*(VC[0]-V0*ecc*np.sin(omega)) + 2*Qabc[1]*(VC[1]+V0*ecc*np.cos(i)*np.cos(omega)))
	KC = KC.to(u.m*u.m/(u.s*u.s))

	KS2 = -0.5*Qabc[2]*V0*V0*np.cos(i)
	KS2 = KS2.to(u.m*u.m/(u.s*u.s))

	KC2 = 0.5*V0*V0*(-Qabc[0]+Qabc[1]*(np.cos(i))**2)
	KC2 = KC2.to(u.m*u.m/(u.s*u.s))

	return [K0/(sp*sp) ,KS/(sp*sp) ,KC/(sp*sp) ,KS2/(sp*sp) ,KC2/(sp*sp)]


def TISS(K,phi):
	"""This function returns the interstellar scintillation timescale
	as defined in equation 9 of Rickett et al. 2014. Note that measured timescale
	values for MJD 52997,53211,53311,53467,53560 are stored in .tiss5t files.
	This function will only return the timescale as a function of phase for one set
	of orbital harmonic coefficients (i.e. one observation day).
	Parameters:
		K: array of orbital harmonic coefficients (K0,KS,KC,KS2,KC2),float
		phi: orbital phase from the line of nodes in radians, float
	Returns:
		TISS: interstellar scintillation timescale at phi, float
	"""
	in_T = (K[0].value+K[1].value*np.sin(phi) + K[2].value*np.cos(phi) +K[3].value*np.sin(2*phi) + K[4].value*np.cos(2*phi))
	return np.sqrt(1/in_T)

def k_norm(VC,m,bm,V0,Qabc,i):
	"""This function calculates the values of ux, uy, and w from equation 13 of Rickett et al 2014.
	These variables combine to give normalized harmonic coefficients ks and k0, which are output by this
	function.
	Parameters:
		VC: related to scintillation velocity (eq.6), array of two floats [VCX,VCY]
		m: pint model of pulsar (determined from par file)
		bm: pint binary instance of pulsar
		V0: the mean orbital velocity of the binary pulsar system, float
		Qabc: quadratic coefficients (eq. 4), array of three floats [a,b,c]
		i: inclination angle in radians, float
	Returns:
		[ks,k0]: an array of unitless normalized harmonic coefficients, floats
		from eq. 14 of Rickett
	"""
	ux = VC[:,0]/V0 -np.array(m.ECC.quantity*np.sin(bm.omega()))
	uy = math.sqrt(Qabc[1]/Qabc[0])*(VC[:,1]/V0 + m.ECC.quantity*np.cos(i)*np.cos(bm.omega()))
	w = Qabc[2]/math.sqrt(Qabc[0]*Qabc[1])
	return [4*ux+2*w*uy,-1-2*ux*ux - 2*w*ux*uy - 2*uy*uy]

