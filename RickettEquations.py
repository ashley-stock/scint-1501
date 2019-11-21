import numpy as np
import matplotlib.pyplot as plt
import math
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyOffsetFrame, SkyCoord
from astropy.time import Time
from pint.models import get_model


"""TODO:
-get values from par file and paper
-create script that goes from parameters to TISS
"""

lt_s = u.Unit('lt_s', u.lightyear / u.yr * u.s)

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
		PB: the orbital period of the pulsar, float
		SMA: the semi major axis, float
		ECC: the orbital eccentricity
	Returns:
		V0: the mean orbital velocity in km/s, float
	"""
	return (2*math.pi*SMA/(PB*np.sqrt(1-ECC*ECC))).to(u.km/u.s)

def SpatialScale(s0,s):
	"""This function calculates the unitless spatial scale in the pulsar frame.
	Parameters:
		s0: the mean diffractive scale (unitless), float
		s: the fractional distance from the pulsar to the scintillation screen, float
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
		psr: a SkyCoordinate object represe
	Returns:
		VE: a two element np array which gives earth velocity in RA and DEC directions 
	"""
	psr_frame = SkyOffsetFrame(origin=psr, rotation=rot)

	tel = EarthLocation.of_site(site)
	pos = tel.get_gcrs(t).transform_to(psr_frame).cartesian
	vel = pos.differentials['s']
	return vel.d_xyz.to(u.km/u.s)

def PulsarBCVelocity(psr):
	return psr.cartesian.differentials['s'].d_xyz.to(u.km/u.s)
	

def RotateVector(v,angle):
	"""This function rotates a vector from RA-DEC coordinates to pulsar orbital frame coordinates.
	Parameters:
		v: a np array with proper motions in km/s in order (PM_RA,PM_DEC), float
		angle: the angle needed to rotate onto x-y plane in radians, float
	Returns:
		[Vx,Vy]: a np array which is the rotated version of the vector
	"""
	new_v = [np.sin(angle)*v[0] + np.cos(angle)*v[1],-np.cos(angle)*v[0]+np.sin(angle)*v[1]]
	return [new_v[0].value, new_v[1].value]*u.km/u.s

def SystemVel(VP,VE,VIS,s):
	"""This function calculates the system velocity in the pulsar frame
	as defined in equation 6 of Rickett et al. 2014.
	Parameters:
		VP: transverse velocity of the orbit barycentre, np array with two floats
		VE: transverse earth velocity, np array with two floats
		VIS: transerve interstellar medium velocity, np array with two floats
		s: fractional distance from the pulsar to the scintillation screen, float
	Returns:
		VC: np array with two floats, representing x and y tranverse system velocity
	"""
	return np.add(np.add(VP,VE*s/(1-s)),-VIS/(1-s))

def K_coeffs(V0,VC,Qabc,i,omega,ecc,sp):
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
	"""This function will return the interstellar scintillation timescale
	as defined in equation 9 of Rickett et al. 2014
	Parameters:
		K: array of orbital harmonic coefficients (K0,KS,KC,KS2,KC2),float
		phi: orbital phase from the line of nodes in radians, float
	Returns:
		TISS: interstellar scintillation timescale at phi, float
	"""
	in_T = (K[0].value+K[1].value*np.sin(phi) + K[2].value*np.cos(phi) +K[3].value*np.sin(2*phi) + K[4].value*np.cos(2*phi))
	return np.sqrt(1/in_T)
	


#Read in known values for par file (from ATNF)
par = "J0737-3039A.par"

psrm = get_model(par)


#In future probably want to read these values from the parameter file
PB = 0.10225156247*u.day
A1 = 1.415032*lt_s
SINI = 0.99974
ECC = 0.0877775
OM = 87.0331*u.deg

SMA = A1/SINI #convert projected semi major axis to actual value

#Fitted values from Table 3 of Rickett et al 2014
i = 88.7*u.deg
s = 0.7
Oangle = 61*u.deg
R = 0.19
PsiAR = 47*u.deg
VIS = np.array([-21,29])*u.km/u.s
s0 = 4.2e6*u.m


psr = SkyCoord(ra='7h37m51.248419s', dec='-30d39m40.71431s', pm_ra_cosdec=-3.82*u.mas/u.year, pm_dec=2.13*u.mas/u.year, distance=1150*u.pc)

#if I put this into the pulsar frame it seems that dy and dz are the Valpha and Vdelta (approximately)

psr_frame = SkyOffsetFrame(origin=psr, rotation=0*u.deg) 
pm_psr = psr.transform_to(psr_frame).cartesian.differentials['s']


#This is where I will use the functions to calculate the things
Qabc = Q_coeff(R,PsiAR)
#print(Qabc)

#Calculate System Velocity
VE = EarthVelocity(Time(53467,format='mjd'),'gbt',psr,0*u.deg)[1:3]
#print('Earth Velocity:',VE)
VP = np.array([-17.8,11.6])*u.km/u.s #Just use values from paper for now
#Otherwise
#VP = PulsarBCVelocity(psr)[1:3] 

#rotate earth and pulsar velocity by Oangle
VE = RotateVector(VE,Oangle)
#print('Earth Velocity:',VE)

VP = RotateVector(VP,Oangle)
#print('Pulsar Velocity:',VP)

VC = SystemVel(VP,VE,VIS,s)


#Calculate K coefficients
Ks0 = K_coeffs(OrbitMeanVel(PB,SMA,ECC),VC,Q_coeff(R,PsiAR),i,OM,ECC,s0/(1-s))
#print('K Coefficients with Paper Pulsar PM', Ks0)



"""
VP = PulsarBCVelocity(psr)[1:3]
VP = RotateVector(VP,Oangle)
print('Pulsar Velocity:',VP)

VC = SystemVel(VP,VE,VIS,s)

Ks1 = K_coeffs(OrbitMeanVel(PB,SMA,ECC),VC,Q_coeff(R,PsiAR),i,OM,ECC,s0/(1-s))
print('K Coefficients with ATNF Pulsar PM', Ks1)
"""

"""
phi = np.linspace(0,360,360)
phi = np.radians(phi)

TS_plot = []
R_plot = []

for angle in phi:
	TS_plot.append(TISS(Ks0,angle))
	R_plot.append(TISS(Ks1,angle))
	#R_plot.append(TISS([0.29e-4*u.s*u.s,0.16e-4*u.s*u.s,-0.00e-4*u.s*u.s,-0.01e-4*u.s*u.s,-0.26e-4*u.s*u.s],angle))

ax = plt.subplot(111)
ax.plot(np.degrees(phi),TS_plot, label='Rickett Pulsar PM')
ax.plot(np.degrees(phi),R_plot, label='ATNF Pulsar PM')
ax.legend()
plt.show()
"""

