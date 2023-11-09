import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

import time

tic = time.time()

# load simulated spectra data
data = np.load('simulated_spectra_eazy_brown_Bryan_etal_2023_paper.npz')
ztrue = data['z']     
lamb_obs = data['wavelengths']       
spec_obs = data['spectra']   

# add noise if desired (or set to inf to analyze noiseless data)
SNR = np.inf # median signal to noise ratio across entire catalog
spec_obs += np.median(spec_obs)*np.random.standard_normal(spec_obs.shape)/SNR

# only analyze the first few galaxies
Ndat = 2000
ztrue = ztrue[0:Ndat]
spec_obs = spec_obs[0:Ndat,:]

from glob import glob
# import one brown galaxy template to determine the scale of the problem
template = pd.read_csv('galsedatlas_brown_etal_2014/hlsp_galsedatlas_multi_multi_ngc-4385_multi_v1_spec.dat',
                 names=['lambda_ang','flux','obs_lambda_ang','source'],         
                 comment='#',sep='\s+')


# get one EAZY template for dictionary initialization later
df1 = pd.read_csv('EAZY_1p1_Spectra/eazy_v1.1_sed1.dat',names=['lambda_ang','flux'],sep='\s+')

# create the initial dictionary
lamb_rest = np.arange(0.01,6.0,0.01)
template_lowres = np.interp(lamb_rest,template['lambda_ang']/10000,template['flux']) # convert wavelength from angstroms to microns
# calculate the rough scale of this initial template
# so the other rows can be rescaled to be smaller relative to it
template_scale = np.std(template_lowres)
D_rest = np.vstack((np.interp(lamb_rest*10000,df1['lambda_ang'],df1['flux']/np.std(df1['flux']))*template_scale*0.1,\
                    np.random.randn(len(template_lowres))*template_scale*0.01,\
                    np.random.randn(len(template_lowres))*template_scale*0.001,\
                    np.random.randn(len(template_lowres))*template_scale*0.0001,\
                    np.random.randn(len(template_lowres))*template_scale*0.00001,\
                    np.random.randn(len(template_lowres))*template_scale*0.000001,\
                    np.random.randn(len(template_lowres))*template_scale*0.0000001,\
                    np.ones_like(template_lowres)*template_scale))
# note that the last row is a constant DC value to enable the code to fit out the average value
D_rest_initial = D_rest.copy()

# utility function to apply a redshift to spectral dictionary
def apply_redshift(D,z,lamb_in,lamb_out):
    # initialize output dictionary
    D_out = np.zeros((D.shape[0],len(lamb_out)))

    # interpolate and redshift each spectrum in the dictionary
    lamb_inx1pz = lamb_in*(1+z)
    for i in range(D.shape[0]):
        D_out[i,:] = np.interp(lamb_out,lamb_inx1pz,D[i,:])

    return D_out

# function to fit one spectrum to dictionary to obtain redshift
def fit_spectrum(lamb_data,spec_data,lamb_D,D,zinput=False):
    # choose redshifts to consider (steps 0-2 in 0.001 incr)
    ztrial = np.arange(0,2.0,0.001)

    # calculate residual at each redshift
    residual_vs_z = np.inf + np.zeros_like(ztrial) # initialize to infinity

    # reshape array in preparation for later calculation
    spec_data_reshaped = np.array((spec_data[:],)).transpose()

    if not zinput:
        # try each redshift
        for k in range(len(ztrial)):
            # make this redshifted template
            D_thisz = apply_redshift(D,ztrial[k],lamb_D,lamb_data)
            
            # fit the data to this template
            # params =  inv(D*D')*D*s'
            params = np.matmul(np.matmul(np.linalg.inv(np.matmul(D_thisz,D_thisz.transpose())),D_thisz),spec_data_reshaped)
            
            # calculate the model from these parameters and this template
            model = np.zeros_like(lamb_data)
            for i in range(D.shape[0]):
                model += params[i]*D_thisz[i,:]

            # calculate the RMS residual
            residual_vs_z[k] = np.sum((model - spec_data)**2)
        
        # find the trial redshift with the lowest residual
        kbest = int(np.where(residual_vs_z == np.min(residual_vs_z))[0])

        # return the redshift with the lowest residual
        z = ztrial[kbest]
    else:
        z = zinput

    # redo the fit at this redshift
    # make this redshifted template
    D_thisz = apply_redshift(D,z,lamb_D,lamb_data)
    
    # fit the data to this template
    # params = inv(D*D')*D*s'
    params = np.matmul(np.matmul(np.linalg.inv(np.matmul(D_thisz,D_thisz.transpose())),D_thisz),spec_data_reshaped)
    # calculate the model for these parameters and this template
    model = np.zeros_like(lamb_data)
    for i in range(D.shape[0]):
        model += params[i]*D_thisz[i,:]

    return z,params,model

Ngal = len(ztrue)

# assume as a calibration we are given the true redshifts of some well-studied reference galaxies
# select these at random
Ncalibrators = 50
from numpy.random import default_rng
rng = default_rng()
i_calibrator_galaxies = rng.choice(Ngal,size=Ncalibrators,replace=False)

# iterate over the data several times
Niterations = 10
resid_array = np.zeros(Niterations+1)
for i_iter in range(Niterations):
    print(str(i_iter)+' of '+str(Niterations)+' iterations')
    # go over the calibrator galaxies first N times to get some higher-quality dictionary updates first
    # then go over all the galaxies
    galaxies_to_evaluate = np.append(np.tile(i_calibrator_galaxies,Niterations),np.arange(Ngal).astype('int'))
    for i_gal in galaxies_to_evaluate:
        # update with number of galaxies processed
        if np.mod(i_gal,100)==0:
            print('    '+str(i_gal)+' of '+str(Ngal)+' spectra')
        
        # if this is a calibrator galaxy
        if i_gal in i_calibrator_galaxies:
            # use the known redshift
            zinput = ztrue[i_gal]
        else:
            # otherwise perform a best-fit for the redshift
            zinput = False

        # fit this spectrum and obtain the redshift
        z,params,model = fit_spectrum(lamb_obs,spec_obs[i_gal,:],lamb_rest,D_rest,zinput=zinput)
        
        # set the learning rate
        learning_rate = 0.01
        # if this is a calibrator galaxy
        if i_gal in i_calibrator_galaxies:
            # use a higher learning rate since we know the redshift is correct
            learning_rate *= 10
        # update the spectral dictionary using the residuals between the model and data
        residual = spec_obs[i_gal,:] - model

        # find the rest wavelength range to update
        j_update = np.where((lamb_rest > np.min(lamb_obs)/(1+z)) & (lamb_rest < np.max(lamb_obs)/(1+z)))[0]
        # interpolate the residual to these values
        interpolated_residual = np.interp(lamb_rest[j_update],lamb_obs/(1+z),residual)
        # inspired by the equation below equation 8 in https://dl.acm.org/doi/pdf/10.5555/1756006.1756008
        # update each item in the dictionary (do not modify the DC offset term at the end)
        for i in range(D_rest.shape[0]-1):
            update_factor = learning_rate*(params[i]/np.sqrt(np.sum(params**2)))
            D_rest[i,j_update] = D_rest[i,j_update] + update_factor*interpolated_residual
        
# plot results
pl.ion()
pl.figure(3)
pl.clf()
pl.plot(lamb_rest,D_rest.transpose(),'-')
pl.plot(np.nan,np.nan,'k-',label='Trained Template')
pl.xlabel('Wavelength [um]')
pl.ylabel('Flux [arb]')
pl.title('Estimating Redshift Templates from Data')
pl.legend()
pl.grid('on')
pl.tight_layout()
pl.savefig('trained_templateSNR'+str(SNR)+'.png',dpi=600)

np.savez_compressed('trained_templateSNR'+str(SNR)+'.npz',lamb_rest=lamb_rest,D_rest=D_rest)

pl.figure(6)
pl.clf()
for i in range(len(D_rest[:,0])-1):
    pl.subplot(len(D_rest[:,0])-1,1,i+1)
    pl.plot(lamb_rest,D_rest[i,:])
    pl.ylabel(str(i+1))
    pl.grid('on')
pl.xlabel('Wavelength [um]')
pl.subplot(len(D_rest[:,0])-1,1,1)
pl.title('Trained Spectral Type Dictionary')
pl.tight_layout()
pl.savefig('trained_template_multiplotSNR'+str(SNR)+'.png',dpi=600)

# fit all galaxies with final template
print('Final Redshift Estimation')
zbest_trained = np.zeros(Ngal)
for i in range(Ngal):
    # update with number of galaxies processed
    if np.mod(i,100)==0:
        print('    '+str(i)+' of '+str(Ngal)+' spectra')
    # fit this spectrum and obtain the redshift
    z,params,model = fit_spectrum(lamb_obs,spec_obs[i,:],lamb_rest,D_rest)
    # store the redshift
    zbest_trained[i] = z

# for comparison, fit again with original template
print('Untrained Redshift Estimation')
zbest_initial = np.zeros(Ngal)
for i in range(Ngal):
    # update with number of galaxies processed
    if np.mod(i,100)==0:
        print('    '+str(i)+' of '+str(Ngal)+' spectra')
    # fit this spectrum and obtain the redshift
    z,params,model = fit_spectrum(lamb_obs,spec_obs[i,:],lamb_rest,D_rest_initial)
    # store the redshift
    zbest_initial[i] = z

# correct dimsionality of ztrue
ztrue = ztrue.flatten()

# find % of catastrophic error and accuracy
igood_initial = np.where(np.abs(zbest_initial-ztrue) < 0.15)[0]
igood_trained = np.where(np.abs(zbest_trained-ztrue) < 0.15)[0]
## standard deviation method
#accuracy_initial = np.std((zbest_initial[igood_initial] - ztrue[igood_initial])/(1+ztrue[igood_initial]))
#accuracy_trained = np.std((zbest_trained[igood_trained] - ztrue[igood_trained])/(1+ztrue[igood_trained]))
# NMAD method
dz = zbest_initial - ztrue
accuracy_initial = 1.48*np.median(np.abs((dz-np.median(dz))/(1+ztrue)))
dz = zbest_trained - ztrue
accuracy_trained = 1.48*np.median(np.abs((dz-np.median(dz))/(1+ztrue)))
# note we could switch to nmad 1.48*median((dz-median(dz))/(1+z))

# plot redshift reconstruction
pl.figure(4)
pl.clf()
pl.subplot(2,1,1)
pl.plot(ztrue,zbest_initial,'*',label='Initial, '+str(100*(Ngal - len(igood_initial))/Ngal)[0:5]+'% Catastropic Error (>0.15)')
pl.plot(ztrue,zbest_trained,'x',label='Trained, '+str(100*(Ngal - len(igood_trained))/Ngal)[0:5]+'% Catastropic Error (>0.15)')
pl.ylabel('Estimated Redshift')
pl.legend()
pl.grid('on')
pl.subplot(2,1,2)
pl.plot(ztrue,zbest_initial-ztrue,'*',label='Initial, dz/(1+z) = '+str(100*accuracy_initial)[0:5]+'% (NMAD)')
pl.plot(ztrue,zbest_trained-ztrue,'x',label='Trained, dz/(1+z) = '+str(100*accuracy_trained)[0:5]+'% (NMAD)')
pl.ylim([-0.1,0.1])
pl.legend()
pl.ylabel('Estimated-True Redshift')
pl.xlabel('True Redshift')
pl.grid('on')
pl.tight_layout()
pl.savefig('redshift_estimation_performanceSNR'+str(SNR)+'.png',dpi=600)

# save estimated redshifts
np.savez('estimated_redshifts.npz',ztrue=ztrue,zest=zbest_trained,zest_initial=zbest_initial)

# for comparison, fit a single spectrum with the initial and trained template
# select galaxy
i = 67
# refit with the initial dictionary
zbest_initial_ex,params_initial,best_model_initial = fit_spectrum(lamb_obs,spec_obs[i,:],lamb_rest,D_rest_initial)
# refit with the trained dictionary
zbest_trained_ex,params_initial,best_model = fit_spectrum(lamb_obs,spec_obs[i,:],lamb_rest,D_rest)
    
# plot spectrum
pl.figure(5)
pl.clf()
pl.plot(lamb_obs,spec_obs[i,:],'k*',label='Data, ztrue='+str(ztrue[i])[0:5])
pl.plot(lamb_obs,best_model_initial,label='Initial Template, zest = '+str(zbest_initial_ex))
pl.plot(lamb_obs,best_model,label='Trained Template, zest = '+str(zbest_trained_ex))
pl.xlabel('Observed Wavelength [um]')
pl.ylabel('Flux [arb]')
pl.legend()
pl.grid('on')
pl.tight_layout()
pl.savefig('spectrum_fittingSNR'+str(SNR)+'.png',dpi=600)

print('Elapsed Time = '+str(time.time()-tic)+' seconds')
