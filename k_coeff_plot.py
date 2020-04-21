import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pint.models import get_model
from pint import toa
from astropy.coordinates import SkyCoord
from RickettEquations import K_coeffs
from RickettTables import fitvals, obs_day, harmonics


def calcKcoeff(fitval):
    """Calculate reduced k harmonic coefficients for the range observed."""
    K0, KS, KC, KS2, KC2 = K_coeffs((53000+obs_day[0],
                                     53000+obs_day[-1],
                                     obs_day[-1]-obs_day[0]+1), fitval,
                                    psr, psr_m)
    return [np.broadcast_to((K/KC2).to(u.one), K0.shape, subok=True)
            for K in (K0, KS, KC, KS2)]


# In[2]:

k_fit = harmonics['value']
k_err = harmonics['error']

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


t = toa.make_fake_toas(53000+obs_day[0], 53000+obs_day[-1],
                       obs_day[-1]-obs_day[0]+1,
                       psr_m, freq=820, obs="GBT")
psr_m.delay(t)
bm = psr_m.binary_instance

# Copies to ensure we do not screw up the original data.
fitval88 = fitvals[88].copy()
fitval90 = fitvals[90].copy()
fitval91 = fitvals[91].copy()
# Uncomment to set the parallel velocity to zero but leave the rest unchanged
# fitval88['VIS'] = -RotateVector(np.array([28.2, 0]), fitval88['PsiAR'])
# fitval91['VIS'] = -RotateVector(np.array([22.8, 0]), fitval91['PsiAR'])
model90 = calcKcoeff(fitval90)
model88 = calcKcoeff(fitval88)
model91 = calcKcoeff(fitval91)


time = np.linspace(obs_day[0], obs_day[-1], obs_day[-1]-obs_day[0]+1)


plt.ion()

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
