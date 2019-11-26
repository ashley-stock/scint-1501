import numpy as np
import matplotlib.pyplot as plt
import math
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyOffsetFrame, SkyCoord
from astropy.time import Time
from pint.models import get_model
from pint import toa
from matplotlib.ticker import MultipleLocator
import math


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
	

#----------------------------------------
# Set up binary pulsar parameters	#
#----------------------------------------

#Read in known values for par file (from ATNF)
par = "J0737-3039A.par"

psr_m = get_model(par)


SMA = (psr_m.A1.quantity/psr_m.SINI.quantity) #convert projected semi major axis to actual value

psr = SkyCoord(ra=str(psr_m.RAJ.quantity), dec=str(psr_m.DECJ.quantity), pm_ra_cosdec=psr_m.PMRA.quantity, pm_dec=psr_m.PMDEC.quantity, distance=1150*u.pc)

#if I put this into the pulsar frame it seems that dy and dz are the Valpha and Vdelta

psr_frame = SkyOffsetFrame(origin=psr, rotation=0*u.deg) 
pm_psr = psr.transform_to(psr_frame).cartesian.differentials['s']


#--------------------------------------
# Set up PINT binary model	      #
#--------------------------------------


t = toa.make_fake_toas(53467,53468,100,psr_m,freq=820,obs="GBT")
"""
times = []

for i in range(10):
	times.append(toa.TOA(Time(53467,format='mjd')+i*psr_m.PB.quantity/10,error=0.0,obs='GBT',freq=820))

t = toa.get_TOAs_list(times,ephem='DE436') 
"""

#psr_m.phase(t) #should give the orbital phase as a function of time

psr_m.delay(t)
bm = psr_m.binary_instance

#NOTE: You can only get these binary model values IFF you call psr_m.delay(t) first!!!

#bm.nu() #this should give the true anomaly
#bm.orbits() #this gives the number of orbits
#bm.omega() # I think this gives the longitude of periastron

#----------------------------------------------------------

#Fitted values from Table 3 of Rickett et al 2014
i = 88.7*u.deg
s = 0.7
Oangle = 61*u.deg
R = 0.19
PsiAR = 47*u.deg
VIS = np.array([-21,29])*u.km/u.s
s0 = 4.2e6*u.m

#----------------------------------------------------------

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
Ks0 = K_coeffs(OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),VC,Q_coeff(R,PsiAR),i,psr_m.OM.quantity,psr_m.ECC.quantity,s0/(1-s))
#print('K Coefficients with Paper Pulsar PM', Ks0)



"""
VP = PulsarBCVelocity(psr)[1:3]
VP = RotateVector(VP,Oangle)
print('Pulsar Velocity:',VP)

VC = SystemVel(VP,VE,VIS,s)

Ks1 = K_coeffs(OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),VC,Q_coeff(R,PsiAR),i,psr_m.OM.quantity,psr_m.ECC.quantity,s0/(1-s))
print('K Coefficients with ATNF Pulsar PM', Ks1)
"""


phi = np.linspace(0,360,360)
phi = np.radians(phi)

phi0 = (bm.nu() + bm.omega())%(2*math.pi)

TS_plot = []
R_plot = []

for angle in phi:
	#TS_plot.append(TISS(Ks0,angle))
	#R_plot.append(TISS(Ks1,angle))
	R_plot.append(TISS([3.25e-4*u.s*u.s,-1.57e-4*u.s*u.s,-0.05e-4*u.s*u.s,0.01e-4*u.s*u.s,-2.57e-4*u.s*u.s],angle))

for angle in phi0:
	TS_plot.append(TISS([3.25e-4*u.s*u.s,-1.57e-4*u.s*u.s,-0.05e-4*u.s*u.s,0.01e-4*u.s*u.s,-2.57e-4*u.s*u.s],angle))

ax = plt.subplot(111)
plt.grid(linestyle='dotted')
plt.xlabel('Orbital Phase (deg)')
plt.ylabel('ISS Timescale (sec)')
ax.xaxis.set_major_locator(MultipleLocator(90))
ax.yaxis.set_major_locator(MultipleLocator(50))
plt.ylim(0,200)
#plt.xlim(0,360)
ax.plot(np.degrees(phi0),TS_plot, 'ro', label='Calculated Angle')
ax.plot(np.degrees(phi),R_plot, label='Rickett Angle')
ax.legend()
plt.show()


