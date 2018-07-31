# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import batman
from limbdark import ld, atlas
import os, sys
import pdb
from mpfit import mpfit
from scipy.optimize import curve_fit
import bin_data
import emcee
import math
import corner


def main(channel=1, ap='4.00', do_mcmc=True, nwalkers = 24, nsteps=200, nburn=20 ):
	fname = 'fluxweight_skycorners_fixedap{0}.phot'.format( ap )
	bjd, x, y, noisepix, flux, goodbad_xy, goodbad_flux = np.genfromtxt( fname, unpack=True )
	err = np.sqrt( flux )
	# find out of transit indices
	pars_guess = [ 0.1, 0, 0, 0, 0, 0, 0, 1, 0, 1 ]

	# plt.plot( bjd, flux, 'o' )
	# plt.show()
	# pdb.set_trace()

	if channel == 1:
		mid_guess = 2.4578462e6 # ch1
		oots = [ 2.45784607e6, 2.457846112e6, 2.45784629e6, 2.45784632e6 ] # ch1
		u = [ 0.062568001, 0.17339600 ]
	elif channel == 2:
		mid_guess = 2.45785038e6
		oots = [ 2.457850188e6, 2.4578502892e6, 2.4578504797e6, 2.457850521e6 ]
		u = [ 0.063892000,  0.13742400 ]

	oot_mean = np.mean( np.concatenate(( flux[ ( bjd > oots[0] )*( bjd < oots[1] ) ], flux[ ( bjd > oots[2] )*( bjd < oots[3] ) ] )) )
	flux_norm = flux / oot_mean
	small_bjd = bjd - mid_guess
	err = err / oot_mean
	# cut first bit off lightcurve
	flux_norm = flux_norm[ bjd > oots[0] ]
	small_bjd = small_bjd[ bjd > oots[0] ]
	x = x[ bjd > oots[0] ]
	y = y[ bjd > oots[0] ]
	noisepix = noisepix[ bjd > oots[0] ]
	goodbad_flux = goodbad_flux[ bjd > oots[0] ]
	goodbad_xy = goodbad_xy[ bjd > oots[0] ]
	err = err[ bjd > oots[0] ]
	# only use good ones
	keep = np.where( goodbad_xy*goodbad_flux == 1 )
	flux_norm2 = flux_norm[keep]
	small_bjd2 = small_bjd[keep]
	x = x[keep]
	y = y[keep]
	noisepix = noisepix[keep]
	err = err[keep]


	# testo = model( small_bjd2, x, y, pars_guess )
	# plt.plot( small_bjd2, flux_norm2, 'o' )
	# plt.plot( small_bjd2, testo, linewidth=3 )
	# plt.show()
	# pdb.set_trace()

	# do mpfit
	fa = { 'flux':flux_norm2, 'err':err, 't':small_bjd2, 'x':x, 'y':y, 'u':u } # mpfit needs you to pass it this dictionary
	results = mpfit( resids_mpfit, pars_guess, functkw=fa, quiet=1 )
	resmod = model( small_bjd2, x, y, results.params, u )
	print results.params
	print results.perror

	#transit_flux  = model( small_bjd2, x, y, pars_guess)
	#plt.plot( small_bjd2, transit_flux, zorder=2 )
	plt.plot( small_bjd2, resmod, zorder=3 )
	plt.plot( small_bjd2, flux_norm2, 'o', zorder=1 )
	plt.xlim( [ small_bjd2[0], small_bjd2[-1] ] )
	plt.xlabel( 'Time (JD - t0)' )
	plt.ylabel( 'Relative Flux' )
	plt.ylim( [0.97, 1.02] )
	plt.savefig( 'w127_spitzer_ch{0}_ap{1}_raw.pdf'.format(channel, ap) )
	plt.clf()
	plt.figure()
	sys1 = systematics( x, y, results.params[2:], small_bjd2 )
	plt.plot( small_bjd2, flux_norm2/sys1, 'o' )
	plt.xlim( [ small_bjd2[0], small_bjd2[-1] ] )
	plt.plot( small_bjd2, trans_model(small_bjd2, results.params[:2], u), linewidth=3, label='Rp/Rs = {0:.5f} +/- {1:.5f}'.format(results.params[0], results.perror[0]) )
	plt.ylim( [0.975, 1.015] )
	plt.legend()
	plt.ylabel( 'Relative Flux' )
	plt.xlabel( 'Time (JD - t0)' )
	plt.savefig( 'w127_spitzer_ch{0}_ap{1}_norm.pdf'.format(channel, ap) )
	plt.clf()
	plt.figure()
	newtimes,binneddata,binnederr = bin_data.bin_data( small_bjd2, flux_norm2/sys1 , 5 )
	plt.errorbar( newtimes, binneddata, yerr=binnederr, fmt='o' )
	plt.xlim( [ small_bjd2[0], small_bjd2[-1] ] )
	plt.plot( small_bjd2, trans_model(small_bjd2, results.params[:2], u), linewidth=2, label='Rp/Rs = {0:.5f} +/- {1:.5f}'.format(results.params[0], results.perror[0]) )
	plt.ylim( [0.975, 1.015] )
	plt.legend()
	plt.ylabel( 'Relative Flux' )
	plt.xlabel( 'Time (JD - t0)' )
	plt.savefig( 'w127_bin_spitzer_ch{0}_ap{1}_norm.pdf'.format(channel, ap) )
	plt.clf()
	plt.figure()
	plt.plot( small_bjd2, flux_norm2-resmod, 'o', zorder=1 )
	plt.plot( plt.xlim(), [0,0], zorder=2 , linewidth=3 )
	plt.ylim( [-0.015, 0.015] )
	plt.ylabel( 'Residuals' )
	plt.xlabel( 'Time (JD - t0)' )
	plt.xlim( [ small_bjd2[0], small_bjd2[-1] ] )
	plt.savefig( 'w127_spitzer_ch{0}_ap{1}_resids.pdf'.format(channel, ap) )
	plt.clf()


	# now try tom's method down here
	offset = np.ones( small_bjd2.size )
	phi = np.column_stack( [ offset, x, y, x**2, y**2, x*y ] )
	pars2 = [ 0.1, 0, 0, 1 ]
	fa = { 'flux':flux_norm2, 'err':err, 't':small_bjd2, 'x':x, 'y':y, 'phi':phi, 'u':u } # mpfit needs you to pass it this dictionary
	results2 = mpfit( matrix_model, pars2, functkw=fa, quiet=1 )

	t_fullmod, t_psignal, t_syst = matrix_model_out( results2.params, flux=flux_norm2, err=err, t=small_bjd2, x=x, y=y, phi=phi, u=u )

	
	plt.plot( small_bjd2, t_fullmod, zorder=3 )
	plt.plot( small_bjd2, flux_norm2, 'o', zorder=1 )
	plt.xlim( [ small_bjd2[0], small_bjd2[-1] ] )
	plt.xlabel( 'Time (JD - t0)' )
	plt.ylabel( 'Relative Flux' )
	plt.ylim( [0.97, 1.02] )
	plt.savefig( 'w127_spitzer_matrix_ch{0}_ap{1}_raw.pdf'.format(channel, ap) )
	plt.clf()
	plt.figure()
	plt.plot( small_bjd2, flux_norm2/t_syst, 'o' )
	plt.xlim( [ small_bjd2[0], small_bjd2[-1] ] )
	plt.plot( small_bjd2, t_psignal, linewidth=3, label='Rp/Rs = {0:.5f} +/- {1:.5f}'.format(results2.params[0], results2.perror[0]) )
	plt.ylim( [0.975, 1.015] )
	plt.legend()
	plt.ylabel( 'Relative Flux' )
	plt.xlabel( 'Time (JD - t0)' )
	plt.savefig( 'w127_spitzer_matrix_ch{0}_ap{1}_norm.pdf'.format(channel, ap) )
	plt.clf()
	plt.figure()
	plt.plot( small_bjd2, flux_norm2-t_fullmod, 'o', zorder=1 )
	plt.plot( plt.xlim(), [0,0], zorder=2 , linewidth=3 )
	plt.ylim( [-0.015, 0.015] )
	plt.ylabel( 'Residuals' )
	plt.xlabel( 'Time (JD - t0)' )
	plt.xlim( [ small_bjd2[0], small_bjd2[-1] ] )
	plt.savefig( 'w127_spitzer_matrix_ch{0}_ap{1}_resids.pdf'.format(channel, ap) )
	plt.clf()

	if do_mcmc == True:
		print 'Running emcee...'
		pars0 = initiate_walkers( nwalkers, results.params )
		ndim = len( results.params )
		sampler = emcee.EnsembleSampler( nwalkers, ndim, log_likelihood, args=[ small_bjd2, x, y, flux_norm2, err, u ] )
		sampler.run_mcmc( pars0, nsteps )
		samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
		fig = corner.corner(samples, labels=[ "RpRs", "T0", "x2", "y2", "xy1", "x1","y1", "c", "tm", "tc"], plot_contours=False)
		figname = 'wasp127_triangle_aprad{0}.eps'.format( ap )
		fig.savefig(figname)
		plt.close()
		#find the 50th, 16th and 84th percentiles)
		rprs_mcmc, t0_mcmc, x2_mcmc, y2_mcmc, xy1_mcmc, x1_mcmc, y1_mcmc, c_mcmc, tm_mcmc, tc_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples,[16,50,84], axis=0)))
		mcmc_results = [ rprs_mcmc, t0_mcmc, x2_mcmc, y2_mcmc, xy1_mcmc, x1_mcmc, y1_mcmc, c_mcmc, tm_mcmc, tc_mcmc ]
		print mcmc_results



def matrix_model( pars, fjac=None, flux=None, err=None, t=None, x=None, y=None, phi=None, u=None ):
	RpRs = pars[0]
	Tmid = pars[1]
	psignal = trans_model( t, pars[:2], u )
	ttrend = pars[3]+pars[2]*t
	resids = flux/( psignal*ttrend )-1
	coeffs = np.linalg.lstsq( phi, resids )[0]
	xysignal = np.dot( phi, coeffs )
	fullmodel = ttrend*psignal*( xysignal+1 )
	resids2 = ( flux - fullmodel ) / err
	status = 0
	return status, resids2


def matrix_model_out( pars, fjac=None, flux=None, err=None, t=None, x=None, y=None, phi=None, u=None ):
	RpRs = pars[0]
	Tmid = pars[1]
	psignal = trans_model( t, pars[:2], u )
	ttrend = pars[3]+pars[2]*t
	resids = flux/( psignal*ttrend )-1
	coeffs = np.linalg.lstsq( phi, resids )[0]
	xysignal = np.dot( phi, coeffs )
	fullmodel = ttrend*psignal*( xysignal+1 )
	return fullmodel, psignal, ttrend*( xysignal+1 )

def model( bjd, x, y, pars, u ):
	blockPrint()
	rp, t0, x2, y2, xy1, x1, y1, c, tm, tc = pars
	params = batman.TransitParams()
	params.per= 4.178062
	params.a= 8.04
	params.inc= 88.7
	params.ecc= 0.0
	params.w= 90
	params.limb_dark = "quadratic"
	params.u = u
	params.rp = rp
	params.t0 = t0
	p = batman.TransitModel(params, bjd)
	transit_flux = p.light_curve(params)
	sys_flux = x2*x**2 + y2*y**2 + xy1*x*y + x1*x + y1*y + c
	time_trend = tm*bjd + tc
	model = sys_flux * transit_flux * time_trend
	enablePrint()
	return model

def trans_model( bjd, pars, u ):
	rp, t0 = pars
	params = batman.TransitParams()
	params.per= 4.178062
	params.a= 8.04
	params.inc= 88.7
	params.ecc= 0.0
	params.w= 90
	params.limb_dark = "quadratic"
	params.u = u
	params.rp = rp
	params.t0 = t0
	p = batman.TransitModel(params, bjd)
	return p.light_curve(params)


def systematics( x, y, pars, bjd ):
	x2, y2, xy1, x1, y1, c, tm, tc = pars
	xy_trend = x2*x**2 + y2*y**2 + xy1*x*y + x1*x + y1*y + c
	time_trend = tm*bjd + tc
	return xy_trend * time_trend


# fmin function
def resids_mpfit(pars, fjac=None, flux=None, err=None, t=None, x=None, y=None, u=None ):
	#print pars
	modele = model(t, x, y, pars, u)
	resids = ( flux - modele ) / err
	status = 0 # return status 0 so mpfit knows it worked OK
	return status, resids

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# emcee functions down here 
def log_likelihood( pars, t, x, y, flux, err, u ):
	ndat = len( t ) 
	rp, t0, x2, y2, xy1, x1, y1, c, tm, tc = pars 
	logp_prior = log_prior( pars )
	if np.isfinite( logp_prior )==True:
		modele = model(t, x, y, pars, u)
		resids = flux - modele 
		logp_data = -0.5*ndat*math.log(2*math.pi) - np.sum([math.log(v) for v in err]) - 0.5*np.sum((resids/err)**2)
	else:
		logp_data = -np.inf
	return logp_data + logp_prior

def initiate_walkers( nwalkers, parsguess ):
	ndim = len( parsguess ) # we have 8 free parameters
	pars0 = np.zeros( [ nwalkers, ndim ] )
	pars0[:,0] = parsguess[0] + (1e-07)*np.random.randn( nwalkers ) # rprs
	pars0[:,1] = parsguess[1] + (1e-07)*np.random.randn( nwalkers ) # t0
	pars0[:,2] = parsguess[2] + (1e-06)*np.random.randn( nwalkers ) # tpops
	pars0[:,3] = parsguess[3] + (1e-06)*np.random.randn( nwalkers ) # tpopf
	pars0[:,4] = parsguess[4] + (1e-06)*np.random.randn( nwalkers ) # dts
	pars0[:,5] = parsguess[5] + (1e-06)*np.random.randn( nwalkers ) # dtf
	pars0[:,6] = parsguess[6] + (1e-06)*np.random.randn( nwalkers ) # lin_m
	pars0[:,7] = parsguess[7] + (1e-06)*np.random.randn( nwalkers ) # lin_c
	pars0[:,8] = parsguess[8] + (1e-07)*np.random.randn( nwalkers ) # lin_m
	pars0[:,9] = parsguess[9] + (1e-06)*np.random.randn( nwalkers ) # lin_c
	return pars0

def log_prior( pars ):
	rp, t0, x2, y2, xy1, x1, y1, c, tm, tc = pars   
	# Adopt simple uniform priors for all parameters
	if ( rp >= 0 )*( rp <= 0.5):
		logp_rp = 0.
	else:
		logp_rp = -np.inf
	if ( t0 >= -0.1 )*( t0 <= 0.1):
		logp_t0 = 0.
	else:
		logp_t0 = -np.inf
	if ( x2 >= -1 )*( x2 <= 1 ):
		logp_x2 = 0.
	else:
		logp_x2 = -np.inf
	if ( y2>= -1 )*( y2<= 1 ):
		logp_y2 = 0.
	else:
		logp_y2 = -np.inf
	if ( xy1>=-1 )*( xy1<=1 ):
		logp_xy1 = 0.
	else:
		logp_xy1 = -np.inf
	if ( x1>=-1 )*( x1<=1 ):
		logp_x1 = 0.
	else:
		logp_x1 = -np.inf
	if ( y1>=-1 )*( y1<=1 ):
		logp_y1 = 0.
	else:
		logp_y1 = -np.inf
	if ( c>=-1 )*( c<=1 ):
		logp_c= 0.
	else:
		logp_c = -np.inf
	if ( tm>=-1 )*( tm<=1 ):
		logp_tm= 0.
	else:
		logp_tm = -np.inf
	if ( tc>=-10 )*( tc<=10 ):
		logp_tc= 0.
	else:
		logp_tc = -np.inf
	return logp_rp+logp_t0+logp_x2+logp_y2+logp_xy1+logp_x1+logp_y1+logp_c+logp_tm+logp_tc