
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pint.models import get_model
from pint import toa
from astropy.coordinates import SkyCoord
from RickettEquations import K_coeffs, fitval90, fitval88, fitval91

plt.ion()

# In[3]:


def calcKcoeff(fitval):
    """Calculate reduced k harmonic coefficients for the range observed."""
    K0, KS, KC, KS2, KC2 = K_coeffs((53000+obs_day[0],
                                     53000+obs_day[-1],
                                     obs_day[-1]-obs_day[0]+1), fitval,
                                    psr, psr_m)
    return [np.broadcast_to((K/KC2).to(u.one), K0.shape, subok=True)
            for K in (K0, KS, KC, KS2)]


# In[2]:


# These are just the values taken from Table 1 of Rickett et al 2014
obs_day = [-3, 211, 311, 379, 467, 202, 203, 274, 312, 319,
           374, 378, 415, 451, 462, 505, 560]

# k_fit[0] is K0 (col. 1 of Table 1), k_fit[1] is Ks (col.2 of Table 1), k_fit[2] is Kc (col.3 of Table 1), k_fit[3] is Kc2 (col.4 of Table 1), k_fit[4] is Kc2 (col. 5 of Table 1),
k_fit = np.array(
    [[3.46, 3.79, 4.76, 6.19, 3.25, 0.96, 0.89, 0.92, 1.35,
      1.41, 0.91, 0.94, 0.74, 0.84, 0.74, 0.54, 0.29],
     [-2.63, 0.70, -3.63, -6.74, -1.57, 0.41, 0.29, -0.40, -1.23,
      -1.39, -1.02, -1.05, -0.80, -0.71, -0.60, 0.06, 0.16],
     [-0.10, -0.02, -0.15, -0.04, -0.05, 0.12, 0.00, 0.03, -0.04,
      -0.06, -0.02, -0.02, 0.00, -0.02, -0.04, 0.00, 0.00],
     [0.10, -0.06, 0.17, 0.04, 0.01, -0.10, 0.04, -0.06, 0.04,
      0.05, 0.03, 0.02, 0.02, 0.05, 0.04, 0.01, -0.01],
     [-2.77, -3.51, -4.02, -4.40, -2.57, -1.07, -0.86, -0.84,
      -1.15, -1.13, -0.59, -0.62, -0.53, -0.67, -0.58, -0.49, -0.26]])

# k_err[0] is K0 (col. 1 of Table 1), k_err[1] is Ks (col.2 of Table 1), k_err[2] is Kc (col.3 of Table 1), k_err[3] is Ks2 (col.4 of Table 1), k_err[4] is Kc2 (col. 5 of Table 1)
k_err = np.maximum(0.01, np.array(
    [[0.06, 0.10, 0.13, 0.14, 0.08, 0.09, 0.05, 0.06, 0.06,
      0.05, 0.05, 0.03, 0.04, 0.03, 0.02, 0.03, 0.01],
     [0.08, 0.11, 0.16, 0.19, 0.10, 0.15, 0.07, 0.08, 0.07,
      0.07, 0.09, 0.05, 0.06, 0.05, 0.03, 0.04, 0.02],
     [0.03, 0.02, 0.05, 0.07, 0.03, 0.12, 0.01, 0.02, 0.02,
      0.03, 0.04, 0.02, 0.03, 0.02, 0.01, 0.01, 0.00],
     [0.05, 0.07, 0.08, 0.08, 0.06, 0.09, 0.04, 0.04, 0.03,
      0.04, 0.05, 0.03, 0.04, 0.03, 0.02, 0.03, 0.01],
     [0.06, 0.10, 0.12, 0.12, 0.08, 0.12, 0.05, 0.06, 0.06,
      0.05, 0.06, 0.03, 0.04, 0.04, 0.02, 0.04, 0.01]]))


norm_k = k_fit[:4] / k_fit[4]
norm_k_error = norm_k * np.sqrt((k_err[:4]/k_fit[:4])**2
                                + (k_err[4]/k_fit[4])**2)
k0, ks, kc, ks2 = norm_k
k0_error, ks_error, kc_error, ks2_error = norm_k_error

par = 'J0737-3039A.par'
psr_m = get_model(par)
psr = SkyCoord(ra=psr_m.RAJ.quantity, dec=psr_m.DECJ.quantity,
               pm_ra_cosdec=psr_m.PMRA.quantity, pm_dec=psr_m.PMDEC.quantity,
               distance=1150*u.pc)

# Uncomment to set the parallel velocity to zero but leave the rest unchanged
# fitval88['VIS'] = -RotateVector(np.array([28.2, 0]), fitval88['PsiAR'])
# fitval91['VIS'] = -RotateVector(np.array([22.8, 0]), fitval91['PsiAR'])

t = toa.make_fake_toas(53000+obs_day[0], 53000+obs_day[-1],
                       obs_day[-1]-obs_day[0]+1,
                       psr_m, freq=820, obs="GBT")
psr_m.delay(t)
bm = psr_m.binary_instance


model90 = calcKcoeff(fitval90)
model88 = calcKcoeff(fitval88)
model91 = calcKcoeff(fitval91)


time = np.linspace(obs_day[0], obs_day[-1], obs_day[-1]-obs_day[0]+1)

fig, axs = plt.subplots(4, sharex=True)
for ax, nk, ek, mk90, mk88, mk91, harmonic in zip(
        axs, norm_k, norm_k_error, model90, model88, model91,
        ['k_{0}', 'k_{s}', 'k_{c}', 'k_{s^{2}}']):
    ax.errorbar(obs_day, nk, yerr=ek, fmt='ko')
    ax.plot(time, mk90, 'r', label=r'i=90$^\ocirc$')
    ax.plot(time, mk91, 'b', label=r'i=91.3$^\ocirc$')
    ax.plot(time, mk88, 'g', label=r'i=88.7$^\ocirc$')
    ax.set_ylabel(f'Harmonic ${harmonic}$')

ax.set_xlabel('MJD-53000')

plt.show()
