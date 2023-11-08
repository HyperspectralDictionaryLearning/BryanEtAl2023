import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import simpson
from astropy.io import ascii

# create main wavelength array
lamb_um = np.arange(0.1,5.0,1/10000)

# load EAZY templates
df1 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed1.dat',names=['lambda_ang','flux'],sep='\s+')
df2 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed2.dat',names=['lambda_ang','flux'],sep='\s+')
df3 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed3.dat',names=['lambda_ang','flux'],sep='\s+')
df4 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed4.dat',names=['lambda_ang','flux'],sep='\s+')
df5 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed5.dat',names=['lambda_ang','flux'],sep='\s+')
df6 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed6.dat',names=['lambda_ang','flux'],sep='\s+')
df7 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed7.dat',names=['lambda_ang','flux'],sep='\s+')

# concatenate these and unit-normalize each of them
# (special case for template 6, since it always shows up as a negative coefficient in the Brown templates)
templates_EAZY = np.vstack((np.interp(lamb_um*10000,df1['lambda_ang'],df1['flux']/np.std(df1['flux'])),\
                            np.interp(lamb_um*10000,df2['lambda_ang'],df2['flux']/np.std(df2['flux'])),\
                            np.interp(lamb_um*10000,df3['lambda_ang'],df3['flux']/np.std(df3['flux'])),\
                            np.interp(lamb_um*10000,df4['lambda_ang'],df4['flux']/np.std(df4['flux'])),\
                            np.interp(lamb_um*10000,df5['lambda_ang'],df5['flux']/np.std(df5['flux'])),\
                            np.interp(lamb_um*10000,df6['lambda_ang'],-1.0*df6['flux']/np.std(df6['flux'])),\
                            np.interp(lamb_um*10000,df7['lambda_ang'],df7['flux']/np.std(df7['flux'])),\
                            np.ones_like(lamb_um)))

# load each of the Brown galaxy spectra
from glob import glob
filenames = glob('galsedatlas_brown_etal_2014/*')
params_arr = np.zeros((8,len(filenames)))
for i in range(len(filenames)):
    print(i)
    # load this galaxy
    df = pd.read_csv(filenames[i],
                    names=['lambda_ang','flux','obs_lambda_ang','source'],
                    comment='#',sep='\s+')
    # interpolate the flux onto the main wavelength array
    this_flux = np.interp(lamb_um*10000,df['lambda_ang'],df['flux'])
    # fit the data to the template
    # params =  inv(D*D')*D*s'
    params = np.matmul(np.matmul(np.linalg.inv(np.matmul(templates_EAZY,templates_EAZY.transpose())),templates_EAZY),this_flux)
    ## create a model
    #model_flux = np.zeros_like(lamb_um)
    #for j in range(7):
    #    model_flux += params[j]*templates_EAZY[j,:]
    # save these parameters to the main array
    params_arr[:,i] = params

# calculate the covariance of these parameters
brown_cov = np.cov(params_arr)
# and the cholesky decomposition (from: https://www.quora.com/How-do-you-generate-random-variables-that-adhere-to-a-given-covariance-matrix)
L = np.linalg.cholesky(brown_cov)

# generate parameters for Ngal random new galaxy spectra
# where their covariance will be the same as the desired covariance
# make 10x more than needed since often the spectra have negative unphysical fluxes
Ngal = 20000
new_params = np.zeros((8,10*Ngal))
for i in range(2*Ngal):
    if not np.mod(i,1000):
        print('i = '+str(i)+' out of '+str(3*Ngal))
    # generage 8 random numbers and multiply by L
    r = np.matmul(L,np.random.randn(8))
    # shift by the mean
    r += np.mean(params_arr,axis=1)
    # save to the array
    new_params[:,i] = r

# qualitatively, the distribution seems like abs(big gaussian) + small gaussian
# for some of the columns
for i in [0,1,2,5]:
    new_params[i,:] = np.abs(new_params[i,:]) + 1e-15*np.random.randn(10*Ngal)

# generate the spectra
# make 10x more than needed since often the spectra have negative unphysical fluxes
new_spectra = np.zeros((2*Ngal,len(lamb_um)))
for i in range(2*Ngal):
    # create a model
    model_flux = np.zeros_like(lamb_um)
    for j in range(7):
        model_flux += new_params[j,i]*templates_EAZY[j,:]
    
    # add to array
    # note that because the distribution is not really gaussian, sometimes the flux is negative
    # artificially take the absolute value to resolve this
    new_spectra[i,:] = model_flux

# find the minimum value of each of these
min_flux = np.min(new_spectra,axis=1)
# cut out any where the flux is negative
igood = np.where(min_flux>0)[0]
# and only use the first Ngal of these
igood = igood[0:Ngal]
# select out the spectra and parameters based on this cut
new_spectra = new_spectra[igood,:]
new_params = new_params[:,igood]

# make evaluation plots
pl.ion()

pl.figure()
pl.plot(lamb_um,new_spectra[0:20,:].transpose())
pl.xlabel('[um]')
pl.ylabel('Flux [arb]')
pl.title('First 20 New Spectra')
pl.grid('on')

pl.figure(figsize=[16,6])
pl.subplot(1,2,1)
pl.imshow(np.cov(params_arr),vmin=-2e-27,vmax=5e-27)
pl.title('Actual Parameter Covariance Matrix')
pl.xlabel('Parameter #')
pl.ylabel('Parameter #')
pl.colorbar()
pl.subplot(1,2,2)
pl.imshow(np.cov(new_params),vmin=-2e-27,vmax=5e-27)
pl.title('Simulated Parameter Covariance Matrix')
pl.xlabel('Parameter #')
pl.ylabel('Parameter #')
pl.colorbar()

pl.figure(figsize=[16,18])
counter = 1
pl.subplot(8,2,1)
pl.title('Actual Parameter Histogram')
pl.subplot(8,2,2)
pl.title('Simulated Parameter Histogram')
for i in range(8):
    pl.subplot(8,2,counter)
    pl.hist(params_arr[i,:])
    pl.ylabel('Param '+str(i))
    counter += 1
    pl.subplot(8,2,counter)
    pl.hist(new_params[i,:])
    counter += 1
pl.tight_layout()


# SPHEREx passbands
# define a general function to calculate the channels over a wavelength range
def calc_channels(lamb_min,lamb_max,Nchannels):
    # equation 2.5 from Caleb Wheeler's thesis
    # https://scholar.colorado.edu/downloads/bz60cx29b
    x = np.exp(-(np.log(lamb_max)-np.log(lamb_min))/Nchannels)

    # create the geometric progression of the band centers
    band_centers = lamb_max*x**np.arange(1,Nchannels+1)

    # return bands, flipped to be in the correct order
    return np.flipud(band_centers)

# run the numbers from the spherex website spherex.caltech.edu
# calculate the band centers and edges
band_centers_123 = calc_channels(0.75,2.42,17*3)
band_centers_4   = calc_channels(2.42,3.82,17)
band_centers_5   = calc_channels(3.82,4.42,17)
band_centers_6   = calc_channels(4.42,5.00,17)

band_center = np.concatenate((band_centers_123,band_centers_4,band_centers_5,band_centers_6))

# figure out the format and length of each filter
filt1 = ascii.read('SPHEREx_filters/spherex_paoyu_001.txt')
n_f = len(filt1['lambda'])
nfilts = len(band_center)

filters = np.zeros((nfilts, 2, n_f))    # This array saves all info about filters
int_res = np.zeros(nfilts)              # Integrated filter responses for calculating average fluxes later

# loop over each filter to save filter info and integrated response
for i in range(0, nfilts):
    n_zeros = 3-len(str(i+1))
    i_idx = '0'*n_zeros + str(i+1)

    filt = ascii.read('SPHEREx_filters/spherex_paoyu_{}.txt'.format(i_idx))
    fl = filt['lambda']
    response = filt['transmission']
    filters[i, 0] = fl
    filters[i, 1] = response
    int_res[i] = simpson(response, fl)    # simpsons is apparently faster than trapezoid

z_array = np.zeros((Ngal,1))        # 1000x1 array to store z in
spectrum_array = np.zeros((Ngal,len(band_center)))  # ~1000x100 matrix to store redshift data
# create mock SPHEREx catalog
for igal in range(Ngal):
    print(igal)
    redshifted_spectrum = np.zeros_like(band_center)
    z = np.random.rand()*2.0
    z_array[igal] = z    # store redshift in the redshift array
    z_array = np.zeros((Ngal,1))        # 1000x1 array to store z in
spectrum_array = np.zeros((Ngal,len(band_center)))  # ~1000x100 matrix to store redshift data
# create mock SPHEREx catalog
for igal in range(Ngal):
    print(igal)
    redshifted_spectrum = np.zeros_like(band_center)
    # generate random redshift
    z = np.random.rand()*2.0
    z_array[igal] = z    # store redshift in the redshift array

    # for each filter, calculate the average flux by convolving it with redshifted template
    for j in range(0, nfilts):

        fl_j = filters[j,0]
        response_j = filters[j,1]
        flux_interp = np.interp(fl_j, lamb_um*(1+z)*10000, new_spectra[igal,:])
        ave_flux = simpson(flux_interp * response_j, fl_j) / int_res[j]  # average flux from integration of spectra and response curve; theoretical value

        spectrum_array[igal, j] = ave_flux


# make a plot of the first few spectra
pl.figure()
pl.plot(band_center,np.transpose(np.log10(spectrum_array[0:10,:])))
pl.legend()

# save the spectra
np.savez('simulated_spectra_eazy_brown_'+str(Ngal/1000)+'k_filtered.npz',z=z_array,wavelengths=band_center,spectra=spectrum_array)