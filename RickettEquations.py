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
from astropy.table import Table


"""TODO:
-make more self-consistent with units?
"""

#-------------------------------------------------------------------------------------
#Fitted values from Table 3 of Rickett et al 2014
#-------------------------------------------------------------------------------------
i = 88.7*u.deg
s = 0.71
Oangle = 61*u.deg
R = 0.19
PsiAR = 47*u.deg
VIS = np.array([-21,29])*u.km/u.s
s0 = 4.2e6*u.m
#--------------------------------------------------------------------------------------
# Other constant values
#--------------------------------------------------------------------------------------
par = "J0737-3039A.par" #parameter file from ATNF
t_start = 53467 #52997
t_end = 53468 #53560
t_nsteps = 2 #563
plot_TISS = True #toggle whether the interstellar scintillation timescale is plotted
plot_knorm_all = False #toggle whether the normalized harmonic coefficients are plotted
plot_knorm_VIS = False
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
	Parameters:
		v: a np array with proper motions in km/s in order (PM_RA,PM_DEC), float
		angle: the angle needed to rotate onto x-y plane in radians, float
	Returns:
		[Vx,Vy]: a np array which is the rotated version of the vector
	"""
	new_v = [np.sin(angle)*v[0] + np.cos(angle)*v[1],-np.cos(angle)*v[0]+np.sin(angle)*v[1]]
	return [new_v[0].value, new_v[1].value]*u.km/u.s


def SystemVel(t_start,t_end,s,Oangle,psr,VIS):
	"""This function calculates the system velocity in the pulsar frame
	as defined in equation 6 of Rickett et al. 2014.
	Parameters:
		t_start: initial time of observation, integer (for now)
		t_end: last time of observation, integer (for now)
		s : the fractional distance from the pulsar to the scintillation screen, float
		Oangle: the angle needed to rotate onto x-y plane in radians, float
		psr: a SkyCoordinate object representation of the pulsar
	Returns:
		VC: np array with two floats, representing x and y tranverse system velocity
	"""	
	times = Time(np.array(range(t_start,t_end)), format = 'mjd')
	
	#Calculate Earth velocity
	VE = []
	for t in times:
		VE.append(RotateVector(EarthVelocity(t,'gbt',psr,0*u.deg)[1:3],Oangle))
	VE = np.array(VE)

	#Calculate pulsar velocity
	VP = np.array([-17.8,11.6])*u.km/u.s #Just use values from paper for now
	VP = RotateVector(VP,Oangle)#rotate pulsar velocity by Oangle

	#Otherwise
	#VP = PulsarBCVelocity(psr)[1:3] 

	VC = []
	for v in VE:
		VC.append(np.add(np.add(VP,v*u.km/u.s*s/(1-s)),-VIS/(1-s)))
	return np.array(VC)*u.km/u.s

#TODO: Maybe make this so it works for VC at multiple times
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

#Might be more useful to just output the normalized coefficients
def uw_fit(VC,m,bm,V0,Qabc,i):
	"""This function calculates the values of ux, uy, and w from equation 13 of Rickett et al 2014.
	These variables combine to give normalized harmonic coefficients ks and k0.
	Parameters:
		VC: related to scintillation velocity (eq.6), array of two floats [VCX,VCY]
		m: pint model of pulsar (determined from par file)
		bm: pint binary instance of pulsar
		V0: the mean orbital velocity of the binary pulsar system, float
		Qabc: quadratic coefficients (eq. 4), array of three floats [a,b,c]
		i: inclination angle in radians, float
	Returns:
		[ux,uy,w]: an array of unitless variables (eq. 6) related to normalized 
		harmonic coefficients, floats
	"""
	ux = VC[:,0]/V0 -np.array(m.ECC.quantity*np.sin(bm.omega()))
	uy = math.sqrt(Qabc[1]/Qabc[0])*(VC[:,1]/V0 + m.ECC.quantity*np.cos(i)*np.cos(bm.omega()))
	w = Qabc[2]/math.sqrt(Qabc[0]*Qabc[1])
	return [ux,uy,w] 	


#----------------------------------------
# Set up binary pulsar parameters	#
#----------------------------------------

if (plot_TISS or plot_knorm_all or plot_knorm_VIS):
	psr_m = get_model(par)#Read in known values for par file (from ATNF)

	SMA = (psr_m.A1.quantity/psr_m.SINI.quantity) #convert projected semi major axis to actual value

	psr = SkyCoord(ra=str(psr_m.RAJ.quantity), dec=str(psr_m.DECJ.quantity), pm_ra_cosdec=psr_m.PMRA.quantity, pm_dec=psr_m.PMDEC.quantity, distance=1150*u.pc)

	t = toa.make_fake_toas(t_start,t_end,t_nsteps,psr_m,freq=820,obs="GBT")

	psr_m.delay(t) #NOTE: You can only get binary model values IFF you call psr_m.delay(t) first!!!
	bm = psr_m.binary_instance
	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)

if (plot_TISS):

	ax = plt.subplot(111)
	plt.grid(linestyle='dotted')
	plt.xlabel('Orbital Phase (deg)')
	plt.ylabel('ISS Timescale (sec)')
	ax.xaxis.set_major_locator(MultipleLocator(90))
	ax.yaxis.set_major_locator(MultipleLocator(50))
	#plt.ylim(0,250)
	plt.xlim(0,360)

	phi = np.linspace(0,360,360)
	phi = np.radians(phi)
	#phi = (bm.nu() + bm.omega())%(2*math.pi)

	s0 = 6.6e6*u.m

	#Calculate K coefficients
	Ks0 = K_coeffs(OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),VC[0],Q_coeff(R,PsiAR),i,psr_m.OM.quantity,psr_m.ECC.quantity,s0/(1-s))
	print('K Coefficients for i = 90', Ks0)

	TS_plot = []

	for angle in phi:
		TS_plot.append(TISS(Ks0,angle))

	ax.plot(np.degrees(phi),TS_plot, label='i=90.0$^\circ$')

	#Values from Column 2
	i = 91.3*u.deg
	s = 0.70
	Oangle = 111*u.deg
	R = 0.96
	PsiAR = 118*u.deg
	VIS = np.array([-79,100])*u.km/u.s
	s0 = 10.1e6*u.m

	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)

	Ks0 = K_coeffs(OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),VC[0],Q_coeff(R,PsiAR),i,psr_m.OM.quantity,psr_m.ECC.quantity,s0/(1-s))

	print('K Coefficients for i = 91.3', Ks0)

	TS_plot = []

	for angle in phi:
		TS_plot.append(TISS(Ks0,angle))

	ax.plot(np.degrees(phi),TS_plot, label='i=91.3$^\circ$')


	#Values from Column 3
	i = 88.7*u.deg
	s = 0.70
	Oangle = 61*u.deg
	R = 0.71
	PsiAR = 61*u.deg
	VIS = np.array([-9,42])*u.km/u.s
	s0 = 6.1e6*u.m

	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)

	Ks0 = K_coeffs(OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),VC[0],Q_coeff(R,PsiAR),i,psr_m.OM.quantity,psr_m.ECC.quantity,s0/(1-s))

	print('K Coefficients for i = 88.7', Ks0)

	TS_plot = []

	for angle in phi:
		TS_plot.append(TISS(Ks0,angle))

	ax.plot(np.degrees(phi),TS_plot, label='i=88.7$^\circ$')

	tiss_data = Table.read('Rickett/P0737_obs/52997/52997.tiss5t',format='ascii.no_header',data_start=11,names=('phi','T_iss','T_iss_error'))

	
	plt.errorbar(tiss_data['phi'],tiss_data['T_iss'],yerr=tiss_data['T_iss_error'],fmt = 'o',label='Data')
	ax.legend()
	plt.title('MJD 52997')
	plt.show()

if (plot_knorm_all):
	uw = uw_fit(VC,psr_m,bm,OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),Q_coeff(R,PsiAR),i)

	times = np.array(range(t_start-53000,t_end-53000))

	plt.plot(times,4*uw[0]+2*uw[2]*uw[1],"r-", label='i=90$^\circ$')
	plt.plot(times,-1-2*uw[0]*uw[0]-2*uw[2]*uw[0]*uw[1]-2*uw[1]*uw[1],"r--")


	#Values from Column 2
	i = 91.3*u.deg
	s = 0.70
	Oangle = 111*u.deg
	R = 0.96
	PsiAR = 118*u.deg
	VIS = np.array([-79,100])*u.km/u.s

	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)

	uw = uw_fit(VC,psr_m,bm,OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),Q_coeff(R,PsiAR),i)

	plt.plot(times,4*uw[0]+2*uw[2]*uw[1],"b-", label='i=91.3$^\circ$')
	plt.plot(times,-1-2*uw[0]*uw[0]-2*uw[2]*uw[0]*uw[1]-2*uw[1]*uw[1],"b--")

	#Values from Column 3
	i = 88.7*u.deg
	s = 0.70
	Oangle = 61*u.deg
	R = 0.71
	PsiAR = 61*u.deg
	VIS = np.array([-9,42])*u.km/u.s

	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)

	uw = uw_fit(VC,psr_m,bm,OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),Q_coeff(R,PsiAR),i)

	plt.plot(times,4*uw[0]+2*uw[2]*uw[1],"g-", label='i=88.7$^\circ$')
	plt.plot(times,-1-2*uw[0]*uw[0]-2*uw[2]*uw[0]*uw[1]-2*uw[1]*uw[1],"g--")
	

	#observation days from Table 1
	obs_day = [-3,211,311,379,467,202,203,274,312,319,374,378,415,451,462,505,560]

	#k_fit[0] is K0 (col. 1 of Table 1), k_fit[1] is Ks (col.2 of Table 1), k_fit[2] is Kc2 (col. 5 of Table 1)
	k_fit = np.array([[3.46,3.79,4.76,6.19,3.25,0.96,0.89,0.92,1.35,1.41,0.91,0.94,0.74,0.84,0.74,0.54,0.29],[-2.63,0.70,-3.63,-6.74,-1.57,0.41,0.29,-0.40,-1.23,-1.39,-1.02,-1.05,-0.80,-0.71,-0.60,0.06,0.16],[-2.77,-3.51,-4.02,-4.40,-2.57,-1.07,-0.86,-0.84,-1.15,-1.13,-0.59,-0.62,-0.53,-0.67,-0.58,-0.49,-0.26]])

	#k_err[0] is K0 (col. 1 of Table 1), k_err[1] is Ks (col.2 of Table 1), k_err[2] is Kc2 (col. 5 of Table 1)
	k_err = np.array([[0.06,0.10,0.13,0.14,0.08,0.09,0.05,0.06,0.06,0.05,0.05,0.03,0.04,0.03,0.02,0.03,0.01],[0.08,0.11,0.16,0.19,0.10,0.15,0.07,0.08,0.07,0.07,0.09,0.05,0.06,0.05,0.03,0.04,0.02],[0.06,0.10,0.12,0.12,0.08,0.12,0.05,0.06,0.06,0.05,0.06,0.03,0.04,0.04,0.02,0.04,0.01]])

	k0 = k_fit[0]/k_fit[2]
	k0_error = k0*(k_err[0]/k_fit[0] + k_err[2]/k_fit[2])

	ks = k_fit[1]/k_fit[2]
	ks_error = ks*(k_err[1]/k_fit[1] + k_err[2]/k_fit[2])

	plt.xlabel('MJD-53000')
	plt.ylabel('Coefficient Value')
	plt.errorbar(obs_day,k0,yerr=k0_error, fmt='kx', label='k0')
	plt.errorbar(obs_day,ks,yerr=ks_error, fmt='ks',label='ks')
	plt.legend()
	plt.show()

if (plot_knorm_VIS):
	#Values from Column 2
	i = 91.3*u.deg
	s = 0.70
	Oangle = 111*u.deg
	R = 0.96
	PsiAR = 118*u.deg
	VIS = np.array([-79,100])*u.km/u.s

	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)

	uw = uw_fit(VC,psr_m,bm,OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),Q_coeff(R,PsiAR),i)

	times = np.array(range(t_start-53000,t_end-53000))

	plt.plot(times,4*uw[0]+2*uw[2]*uw[1],"r-", label='VIS,y = 100 km/s')
	plt.plot(times,-1-2*uw[0]*uw[0]-2*uw[2]*uw[0]*uw[1]-2*uw[1]*uw[1],"r--")


	#Values from Column 2
	VIS = np.array([-79,200])*u.km/u.s
	
	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)
	uw = uw_fit(VC,psr_m,bm,OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),Q_coeff(R,PsiAR),i)

	plt.plot(times,4*uw[0]+2*uw[2]*uw[1],"b-", label='VIS,y = 200 km/s')
	plt.plot(times,-1-2*uw[0]*uw[0]-2*uw[2]*uw[0]*uw[1]-2*uw[1]*uw[1],"b--")

	#Values from Column 3
	VIS = np.array([-79,300])*u.km/u.s

	VC = SystemVel(t_start,t_end,s,Oangle,psr,VIS)

	uw = uw_fit(VC,psr_m,bm,OrbitMeanVel(psr_m.PB.quantity,SMA,psr_m.ECC.quantity),Q_coeff(R,PsiAR),i)

	plt.plot(times,4*uw[0]+2*uw[2]*uw[1],"g-", label='VIS,y = 300 km/s')
	plt.plot(times,-1-2*uw[0]*uw[0]-2*uw[2]*uw[0]*uw[1]-2*uw[1]*uw[1],"g--")
	

	#observation days from Table 1
	obs_day = [-3,211,311,379,467,202,203,274,312,319,374,378,415,451,462,505,560]

	#k_fit[0] is K0 (col. 1 of Table 1), k_fit[1] is Ks (col.2 of Table 1), k_fit[2] is Kc2 (col. 5 of Table 1)
	k_fit = np.array([[3.46,3.79,4.76,6.19,3.25,0.96,0.89,0.92,1.35,1.41,0.91,0.94,0.74,0.84,0.74,0.54,0.29],[-2.63,0.70,-3.63,-6.74,-1.57,0.41,0.29,-0.40,-1.23,-1.39,-1.02,-1.05,-0.80,-0.71,-0.60,0.06,0.16],[-2.77,-3.51,-4.02,-4.40,-2.57,-1.07,-0.86,-0.84,-1.15,-1.13,-0.59,-0.62,-0.53,-0.67,-0.58,-0.49,-0.26]])

	#k_err[0] is K0 (col. 1 of Table 1), k_err[1] is Ks (col.2 of Table 1), k_err[2] is Kc2 (col. 5 of Table 1)
	k_err = np.array([[0.06,0.10,0.13,0.14,0.08,0.09,0.05,0.06,0.06,0.05,0.05,0.03,0.04,0.03,0.02,0.03,0.01],[0.08,0.11,0.16,0.19,0.10,0.15,0.07,0.08,0.07,0.07,0.09,0.05,0.06,0.05,0.03,0.04,0.02],[0.06,0.10,0.12,0.12,0.08,0.12,0.05,0.06,0.06,0.05,0.06,0.03,0.04,0.04,0.02,0.04,0.01]])

	k0 = k_fit[0]/k_fit[2]
	k0_error = k0*(k_err[0]/k_fit[0] + k_err[2]/k_fit[2])

	ks = k_fit[1]/k_fit[2]
	ks_error = ks*(k_err[1]/k_fit[1] + k_err[2]/k_fit[2])

	plt.xlabel('MJD-53000')
	plt.ylabel('Coefficient Value')
	plt.errorbar(obs_day,k0,yerr=k0_error, fmt='kx', label='k0')
	plt.errorbar(obs_day,ks,yerr=ks_error, fmt='ks',label='ks')
	plt.legend()
	plt.show()


