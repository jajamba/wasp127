# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from astropy.io import fits
import pickle as pickle
import matplotlib.pyplot as plt
import pdb
from mpfit import mpfit
import batman
import ackBar2
import ackBar2Multi
import os, sys
from limbdark import ld, atlas
import emcee
import math
import corner

orbit_option = 'transit' # 'transit', 'eclipse'
do_mcmc = True
syst_option = 'trap' # 'trap', 'poly4'
nwalkers = 16
nsteps = 500
nburn = 50
nthreads = 12
wlrange = [ 1.06, 1.75 ] # wavelength range in microns
period = 4.17806
pl_pars = [ 0.1 ]
tpars = [ 30, 30, 1, 1, -0.001, 0.9999 ]
p4_pars = [ 0, 0, 0, 0, 1, 0, 1 ]
jd0 = 2458218.051694
apradius = 75
exptime = 95.782104
specfile = 'G141.apradius75.00pix.unsmoothed.maskrad75pix.bg2.spectra.rdiff.zapped.wasp127.pkl'
imfile1 = 'ida504dwq_ima.fits'
path = '/home/jspake/projects/in_progress/wasp127/data/fits'


# main function
def main( orbit_option=orbit_option, do_mcmc=do_mcmc, syst_option=syst_option, \
			nwalkers=nwalkers, nsteps=nsteps, nburn=nburn, nthreads=nthreads, \
			wlrange=wlrange, period=period, pl_pars=pl_pars, tpars=tpars, \
			p4_pars=p4_pars, jd0=jd0, apradius=apradius, \
			exptime=exptime, specfile=specfile, \
			imfile1=imfile1, path=path ):
	plt.ioff()
	os.chdir(path)
	# extract and prep light curves
	lc_dict = prep_lc_dict( specfile, wlrange, syst_option, exptime=exptime, \
		period=period, jd0=jd0 )

	smalldata = prep_trap_model( specfile, imfile1, wlrange ) / exptime


	# first get the right parameter guesses
	if syst_option == 'trap':
		syst_pars = tpars
	elif syst_option == 'poly4':
		syst_pars = p4_pars
	transit_pars = np.concatenate(( pl_pars, syst_pars ))
	print('Running first transit-only mpfit...')
	fa = { 'lc_dict':lc_dict, 'wlrange':wlrange, 'smalldata':smalldata, 'orbit_option':orbit_option, 'syst_option':syst_option }
	results_t1 = mpfit( resids_mpfit_transit, transit_pars, functkw=fa, quiet=1 )
	print('Finding outliers...')
	lc_dict = cut_outliers_transit( lc_dict, results_t1.params, wlrange, smalldata )
	if do_mcmc == False:
		print('Running second transit-only mpfit...')
		fa = { 'lc_dict':lc_dict, 'wlrange':wlrange, 'smalldata':smalldata, 'orbit_option':orbit_option, 'syst_option':syst_option  }
		results_t2 = mpfit( resids_mpfit_transit, results_t1.params, functkw=fa, quiet=1 )
		# print results_t2.params
		# print results_t2.perror
		# add final results to dictionary
		lc_dict_transit = add_results2dict( lc_dict, results_t2, wlrange, smalldata, do_mcmc, syst_option, orbit_option, jd0 )
		# save pickle file
		pickname = 'wasp127_mpfit_{0}_{1}_{2:.2f}-{3:.2f}micron_aprad{4}.pkl'.format( orbit_option, syst_option, wlrange[0], wlrange[1], apradius )
		f = open( pickname, 'wb' )
		pickle.dump( lc_dict_transit, f )
		f.close()
	elif do_mcmc == True:
		print('Running emcee...')
		pars0 = initiate_walkers_trap_trans( nwalkers, results_t1.params )
		ndim = len( results_t1.params )
		sampler = emcee.EnsembleSampler( nwalkers, ndim, log_likelihood_trap_trans, args=[ lc_dict, wlrange, smalldata ], threads=nthreads )
		sampler.run_mcmc( pars0, nsteps )
		samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
		fig = corner.corner(samples, labels=[ "RpRs", "tpops", "tpopf", "dts", "dtf","slope", "c"], plot_contours=False)
		figname = 'wasp19_triangle_{0}_{1}_{2:.2f}-{3:.2f}micron_aprad{4}.eps'.format( orbit_option, syst_option, wlrange[0], wlrange[1], apradius )
		fig.savefig(figname)
		plt.close()
		#find the 50th, 16th and 84th percentiles)
		rprs_mcmc, tpops_mcmc, tpopf_mcmc, dts_mcmc, dtf_mcmc, linm_mcmc, linc_mcmc = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(samples,[16,50,84], axis=0))]
		mcmc_results = np.array([ rprs_mcmc, tpops_mcmc, tpopf_mcmc, dts_mcmc, dtf_mcmc, linm_mcmc, linc_mcmc ])
		lc_dict_transit = add_results2dict( lc_dict, mcmc_results, wlrange, smalldata, do_mcmc, syst_option, orbit_option, jd0 )
		pickname = 'wasp127_mcmcfit_{0}_{1}_{2:.2f}-{3:.2f}micron_aprad{4}.pkl'.format( orbit_option, syst_option, wlrange[0], wlrange[1], apradius )
		f = open( pickname, 'wb' )
		pickle.dump( lc_dict_transit, f )
		f.close()
		

	




def get_ld_coeffs_w127(wlrange):
	print('Finding limbdarkening coefficients for wavelength range {0}'.format( wlrange ))
	ATLAS_TEFF = 5500. # closest temperature in the model grid to WASP-127's Teff=5500K
	ATLAS_LOGG = 4.0 # closest logg in the model grid to WASP-127's logg=4.0cgs
	mu, wav, intens = atlas.read_grid( model_filepath='ip00k2new.pck', teff=ATLAS_TEFF, logg=ATLAS_LOGG, new_grid=True )
	# The WFC3 G141 bandpass:
	g141_bandpass = np.loadtxt( 'WFC3.G141.throughput.txt' )
	tr_wavs = g141_bandpass[:,0]
	tr_vals = g141_bandpass[:,1]
	cuton_micron = wlrange[0]
	cutoff_micron = wlrange[1]
	# This line fits limb darkening laws to the intensities read from the ATLAS
	# model modulated by the WFC3 G141 throughput:
	ld_coeffs = ld.fit_law( mu, wav, intens, tr_wavs, \
	                        cuton_wav_nm=cuton_micron*1000, cutoff_wav_nm=cutoff_micron*1000, \
	                        passband_sensitivity=tr_vals, plot_fits=False )
	# The output is a dictionary containing limb darkening coefficients for
	# various limb darkening laws:
	return ld_coeffs['fourparam_nonlin']


# function to prepare lightcurve dictionary
# calculate phase and times in seconds
# order alljd, allphase and alllc
def prep_lc_dict( fname, wlrange, syst_option, exptime=exptime, period=period, jd0=jd0 ):
	indict = { }
	# first extract spectra:
	indict['jd'], indict['lc'], indict['err'], indict['y'], indict['wavshift'], _, indict['hstphase'] = extractspec(fname,wlrange)

	# if using the polynomial systematics model, remove first orbit
	if syst_option == 'poly4':
		ixs = np.arange( 27 )
		indict['jd'] = np.delete(jd, ixs ) 
		indict['lc'] = np.delete(lc, ixs )
		indict['err'] = np.delete(err, ixs )
		indict['y'] = np.delete(y, ixs )
		indict['wavshift'] = np.delete( wavshift_pixels1, ixs )
		indict['hstphase'] = np.delete( hstphase1, ixs )
	
	ixmed = [ 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73 ]

	# normalise errors, lc
	med = np.median(indict['lc'][ixmed])
	indict['err'] = indict['err']/med
	indict['lc'] = indict['lc']/med

	indict['jd'] = indict['jd'] - jd0
	texp_sec = indict['jd'] * 24 * 60 * 60
	indict['texp_sec'] = texp_sec - ( exptime / 2 )
	
	# calculate phase
	indict['ph'] = indict['jd']/period

	# add outlier array
	indict['out'] = np.ones( len(indict['jd']), dtype=bool )
	return indict



def prep_trap_model(specfile, imfile, wlrange):
	# from pickle file containing reduced spectra get:
	# wavelength solutions and position of spectra on raw images
	ifile = open( specfile, 'rb' )
	z = pickle.load( ifile, encoding='latin1' )
	ifile.close()
	wavesol = z['wavsol_micron']
	edgetrim = 10
	dispersion_lower = z['trim_disp_bound_ixs'][0] + edgetrim
	dispersion_upper = z['trim_disp_bound_ixs'][1] + edgetrim
	crossdispersion_lower = np.median( z['auxvars']['y'] ) - apradius + edgetrim
	crossdispersion_upper = np.median( z['auxvars']['y'] ) + apradius + edgetrim
	crossdispersion_upper = int(round(crossdispersion_upper))
	crossdispersion_lower = int(round(crossdispersion_lower))
	# select corresponding section of first exposure to use as image template
	hdulist = fits.open( imfile )
	ima = hdulist[1].data
	hdulist.close()
	bandis = np.where((wlrange[0] <  wavesol[0,:]) & (wavesol[0,:] < wlrange[1] )  )
	band_lower = dispersion_lower + bandis[0][0]
	band_upper = dispersion_lower + bandis[0][-1]
	smalldata = ima[ crossdispersion_lower:crossdispersion_upper+1, band_lower:band_upper+1 ] 
	#print np.shape(smalldata)
	smalldata = np.median(smalldata, axis=1)
	#print np.shape(smalldata)
	smalldata = np.reshape(smalldata,[len(smalldata),1])
	return smalldata




def batman_params_w127(rp, wlrange):
	blockPrint()
	params = batman.TransitParams()
	params.per= 4.17806
	params.a= 8.044
	params.inc= 88.7
	params.ecc= 0.0
	params.w= 90
	params.rp = rp
	params.limb_dark = "nonlinear"
	params.u = get_ld_coeffs_w127(wlrange)
	enablePrint()
	return params

def transit_model( params, jd ):
	blockPrint()
	params.t0 = 0
	p = batman.TransitModel(params,jd)
	transit_flux = p.light_curve(params)
	enablePrint()
	return transit_flux



# function to compute residuals - ackbar and linear time trend for single visit
def trap_lin_model(pars, smalldata, texp_sec, jd, exptime=95.782104 ):
	tpops, tpopf, dts, dtf, m_lin, c_lin = pars
	intrinsic = np.ones( len(texp_sec) )
	# get ramp model functions:
	trap_model = ackBar2Multi.ackBar2Multi(smalldata, intrinsic, texp_sec, exptime, trap_pop_s=tpops, trap_pop_f=tpopf, dTrap_s=dts, dTrap_f=dtf)
	trap_model = trap_model / np.median( trap_model )
	lin_model = jd * m_lin + c_lin
	return trap_model * lin_model 

def poly4_lin_model( pars, lc_dict, orbit_option ):
	p4, p3, p2, p1, p0, m_lin, c_lin = pars
	ph = lc_dict['hstphase']
	jd = lc_dict['jd']
	poly = p4*ph**4 + p3*ph**3 + p2*ph**2 + p1*ph + p0
	lin = jd * m_lin + c_lin
	return poly * lin



def resids_mpfit_transit( pars, fjac=None, lc_dict=None, wlrange=None, smalldata=None, orbit_option=None, syst_option=None ):
	model = model_eval_transit( lc_dict, pars, wlrange, smalldata, orbit_option, syst_option )
	resids = ( lc_dict['lc'] - model ) / lc_dict['err']
	resids = resids[lc_dict['out']]
	status = 0
	return status, resids

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def extractspec(fname,wlrange):
	ifile = open( fname, 'rb' )
	z = pickle.load( ifile, encoding='latin1' )
	ifile.close()
	ecounts = z['ecounts'] # time series spectra
	auxvars = z['auxvars']
	wavesol = z['wavsol_micron'] # wavelength solution
	jd = auxvars['jd'] # JD from FITS headers
	hstphase = auxvars['hstphase'] # HST orbital phase
	y = auxvars['y'] # cross-dispersion centroid time series
	bg_ppix = auxvars['bg_ppix'] # background counts per pixel time series
	wavshift_micron = auxvars['wavshift_micron']
	wavshift_pixels = auxvars['wavshift_pixels']
	x=auxvars['x'] # dummy variable, ignore this
	# make lighcturve for selected wavelength range
	ndat = len( y )	
	speclc = np.ones( ndat )
	for i in range( ndat ):
		iwavesol = wavesol[i,:]
		iecounts = ecounts[i,:]
		iwavel = np.where((wlrange[0] < iwavesol) & (iwavesol < wlrange[1]) )
		speclc[i] = np.sum(iecounts[iwavel])
	speclc_uncs = np.sqrt( speclc )
	return jd, speclc, speclc_uncs, y, wavshift_micron, wavshift_pixels, hstphase



def model_eval_transit( lc_dict, pars, wlrange, smalldata, orbit_option, syst_option ):
	rp = pars[0]
	bat_params = batman_params_w127( rp, wlrange )
	transit_flux = transit_model( bat_params, lc_dict['jd'] )
	if syst_option == 'trap':
		trap_mod = trap_lin_model( pars[-6:], smalldata, lc_dict['texp_sec'], lc_dict['jd'] )
		model = trap_mod * transit_flux
	elif syst_option == 'poly4':
		poly4_mod = poly4_lin_model( pars[-7:], lc_dict, orbit_option )
		model = poly4_mod * transit_flux
	return model


def cut_outliers_transit(lc_dict, pars, wlrange, smalldata ):
	model = model_eval_transit( lc_dict, pars, wlrange, smalldata, orbit_option, syst_option )
	noes = np.where( abs( lc_dict['lc'] - model ) > ( 4 * lc_dict['err'] ) )[0]
	lc_dict['out'][noes] = False
	return lc_dict

# function to add final results to final dictionary
def add_results2dict( lc_dict, results, wlrange, smalldata, do_mcmc, syst_option, orbit_option, jd0 ):
	# make copy of dictionary and add results
	lc_dict_final = lc_dict.copy()
	if orbit_option == 'transit':
		jd = lc_dict['jd']
		texp_sec = lc_dict['texp_sec']
	if do_mcmc == False:
		final_params, final_errors = results.params, results.perror
		lc_dict_final['results'] = np.column_stack(( final_params, final_errors ))
	elif do_mcmc == True:
		lc_dict_final['results'] = results
		final_params = results[:,0]
	# also add best-fit transit and systematics model
	bat_params = batman_params_w127( final_params[0], wlrange )
	final_transit = transit_model( bat_params, jd )
	# add transit model with high cadence for plotting
	fig_times = np.arange( jd[0], jd[-1], 0.0001 )
	fig_transit = transit_model( bat_params, fig_times )
	if syst_option == 'trap':
		final_syst = trap_lin_model( final_params[-6:], smalldata, texp_sec, jd )
	elif syst_option == 'poly4':
		final_syst = poly4_lin_model( final_params[-7:], lc_dict, orbit_option )
	else:
		pdb.set_trace()
	lc_dict_final['best_transit'] = final_transit
	lc_dict_final['best_systematics'] = final_syst
	lc_dict_final['fig_transit'] = fig_transit
	lc_dict_final['fig_times'] = fig_times
	lc_dict_final['jd0'] = jd0
	return lc_dict_final

# emcee functions down here
def log_likelihood_trap_trans( pars, lc_dict, wlrange, smalldata ):
	ndat = len(lc_dict['jd'])
	rprs, tpops, tpopf, dts, dtf, lin_m, lin_c = pars 
	logp_prior = log_prior_trap_trans( pars )
	if np.isfinite( logp_prior )==True:
		model = model_eval_transit( lc_dict, pars, wlrange, smalldata, orbit_option, syst_option )
		resids = ( lc_dict['lc'] - model ) / lc_dict['err']
		resids = resids[lc_dict['out']]
		yerr = lc_dict['err'][lc_dict['out']]
		logp_data = -0.5*ndat*math.log(2*math.pi) - np.sum([math.log(v) for v in yerr]) - 0.5*np.sum((resids/yerr)**2)
	else:
		logp_data = -np.inf
	return logp_data + logp_prior

def initiate_walkers_trap_trans( nwalkers, parsguess ):
	ndim = len(parsguess) # we have 8 free parameters
	pars0 = np.zeros( [ nwalkers, ndim ] )
	pars0[:,0] = parsguess[0] + (1e-07)*np.random.randn( nwalkers ) # rprs
	pars0[:,1] = parsguess[1] + (1e-01)*np.random.randn( nwalkers ) # tpops
	pars0[:,2] = parsguess[2] + (1e-01)*np.random.randn( nwalkers ) # tpopf
	pars0[:,3] = parsguess[3] + (1e-01)*np.random.randn( nwalkers ) # dts
	pars0[:,4] = parsguess[4] + (1e-01)*np.random.randn( nwalkers ) # dtf
	pars0[:,5] = parsguess[5] + (1e-07)*np.random.randn( nwalkers ) # lin_m
	pars0[:,6] = parsguess[6] + (1e-07)*np.random.randn( nwalkers ) # lin_c
	return pars0

def log_prior_trap_trans( pars ):
	rprs, tpops, tpopf, dts, dtf, lin_m, lin_c = pars  
	# Adopt simple uniform priors for all parameters
	if ( rprs >= 0 )*( rprs <= 0.5):
		logp_rprs = 0.
	else:
		logp_rprs = -np.inf
	if ( tpops >= -200 )*( tpops <= 5000 ):
		logp_tpops = 0.
	else:
		logp_tpops = -np.inf
	if ( tpopf>= -200 )*( tpopf<= 5000 ):
		logp_tpopf = 0.
	else:
		logp_tpopf = -np.inf
	if ( dts>=-400 )*( dts<=1000 ):
		logp_dts = 0.
	else:
		logp_dts = -np.inf
	if ( dtf>=-400 )*( dtf<=1000 ):
		logp_dtf = 0.
	else:
		logp_dtf = -np.inf
	if ( lin_m>=-0.1 )*( lin_m<=0.1 ):
		logp_lin_m = 0.
	else:
		logp_lin_m = -np.inf
	if ( lin_c>=0.9 )*( lin_c<=1.1 ):
		logp_lin_c= 0.
	else:
		logp_lin_c = -np.inf
	return logp_rprs+logp_tpops+logp_tpopf+logp_dts+logp_dtf+logp_lin_m+logp_lin_c
