from emcee.utils import MPIPool
from mpfit import mpfit
from statistics import mean
import copy
import corner
import datetime
import emcee
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import statistics
import sys
import subprocess
from itertools import islice


##################################################
#                                                #
#   Set expected ranges for target environment   #
#                                                #
##################################################

logTgas_range   = [0, 3] # molex struggles below 10K, reduce with caution
lognH2_range = [1, 7]
logNOH_range = [11, 16]
fortho_range = [0, 1]
FWHM_range   = [0.1, 10]
Av_range     = [0, 10]
logxOH_range    = [-8, -6]
logxHe_range    = [-2, 0]
logxe_range     = [-5, -3]
logTdint_range  = [0, 3]
logTd_range     = [0, 3]

# Variables used throughout:
#
# parameter_list = list of parameters for molex found in dictionary object. Those =True are to be fit
# p = full set of parameters for molex for all gaussians (including vel!)
# x = subset of parameters for molex for all gaussians (the parameters we're fitting)
# params = full set of v, FWHM, tau(x4), Texp(x4) for all gaussians


#######
# Log #
#######

log = open('log.txt', 'a')




###############################
#                             #
#     tau and Texp spectra    #
#                             #
###############################
#              |
#              |
#            \ | /
#             \|/
#              V
def findranges(data, num_chan = 5): # adds 'interesting_vel' and 'sig_vel_ranges' to dictionary
	data['interesting_vel'] = interestingvel(data)

	if data['interesting_vel'] != None:
		sig_vel_list = data['interesting_vel']
	else:
		sig_vel_list = []

	if data['Texp_spectrum']['1665'] != []:
		vel_alogxes = [data['vel_axis']['1612'], data['vel_axis']['1665'], 
					data['vel_axis']['1667'], data['vel_axis']['1720']] * 2
		spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720'], 
					data['Texp_spectrum']['1612'], data['Texp_spectrum']['1665'], 
					data['Texp_spectrum']['1667'], data['Texp_spectrum']['1720']]
		spectra_rms = [data['tau_rms']['1612'], data['tau_rms']['1665'], 
					data['tau_rms']['1667'], data['tau_rms']['1720'], 
					data['Texp_rms']['1612'], data['Texp_rms']['1665'], 
					data['Texp_rms']['1667'], data['Texp_rms']['1720']]
	else:
		vel_alogxes = [data['vel_axis']['1612'], data['vel_axis']['1665'], 
					data['vel_axis']['1667'], data['vel_axis']['1720']]
		spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720']]
		spectra_rms = [data['tau_rms']['1612'], data['tau_rms']['1665'], 
					data['tau_rms']['1667'], data['tau_rms']['1720']]

	for s in range(len(vel_alogxes)):
		vel_axis = vel_alogxes[s]
		spectrum = spectra[s]
		spectrum_rms = spectra_rms[s]

		for veli in range(len(vel_axis) - num_chan):
			test_vel = vel_axis[veli:veli + num_chan]
			test_spec = spectrum[veli:veli + num_chan]

			test = [np.abs(test_spec[a]) <= spectrum_rms for a in range(num_chan)]
			if not np.any(test):
				if sig_vel_list == []:
					sig_vel_list = [np.median(test_vel)]
				else:
					sig_vel_list = np.concatenate((sig_vel_list, [np.median(test_vel)]))

	sig_vel_list = reducelist(sig_vel_list, 1., 5)
	sig_vel_ranges = [[x[0], x[-1]] for x in sig_vel_list if x[0] != x[-1]]
	data['sig_vel_ranges'] = sig_vel_ranges
	return data
def interestingvel(data = None): # returns interesting_vel
	id_vel_list = []
	dv = np.abs(data['vel_axis']['1612'][1] - data['vel_axis']['1612'][0])
	# Flag features
	if data['Texp_spectrum']['1665'] != []:
		vel_alogxes = [data['vel_axis']['1612'], data['vel_axis']['1665'], 
					data['vel_axis']['1667'], data['vel_axis']['1720']] * 2
		spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720'], 
					data['Texp_spectrum']['1612'], data['Texp_spectrum']['1665'], 
					data['Texp_spectrum']['1667'], data['Texp_spectrum']['1720']]
		spectra_rms = [data['tau_rms']['1612'], data['tau_rms']['1665'], 
					data['tau_rms']['1667'], data['tau_rms']['1720'], 
					data['Texp_rms']['1612'], data['Texp_rms']['1665'], 
					data['Texp_rms']['1667'], data['Texp_rms']['1720']]
	else:
		vel_alogxes = [data['vel_axis']['1612'], data['vel_axis']['1665'], 
					data['vel_axis']['1667'], data['vel_axis']['1720']]
		spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720']]
		spectra_rms = [data['tau_rms']['1612'], data['tau_rms']['1665'], 
					data['tau_rms']['1667'], data['tau_rms']['1720']]

	for s in range(len(vel_alogxes)):
		vel_axis = vel_alogxes[s]
		spectrum = spectra[s]
		spectrum_rms = spectra_rms[s]

		dx = derivative(vel_axis, spectrum)
		dx2 = derivative(vel_axis, dx, 2)

		rms_dx = findrms(dx)
		rms_dx2 = findrms(dx2)

		spectrum_zero = [abs(x) < 2. * spectrum_rms for x in spectrum]
		spectrum_pos = [x > 2. * spectrum_rms for x in spectrum]
		
		dx_pos = [x > 2. * rms_dx for x in dx]
		dx2_pos = [x > 2. * rms_dx2 for x in dx2]
		
		dx_zero = zeros(vel_axis, dx, rms_dx)
		dx2_zero = zeros(vel_axis, dx2, rms_dx2)		

		vel_list1 = [vel_axis[z] for z in range(int(len(dx_pos) - 1)) if dx_zero[z] == True and spectrum_pos[z] != dx2_pos[z]]
		vel_list2 = [vel_axis[z] for z in range(1,int(len(dx2_zero) - 1)) if dx2_zero[z] == True and spectrum_zero[z] == False and np.any([x <= dx2_zero[z+1] and x >= dx2_zero[z-1] for x in dx_zero]) == False]
		
		vel_list = np.concatenate((vel_list1, vel_list2))
		
		id_vel_list.append(vel_list)
	
	id_vel_list = sorted([val for sublist in id_vel_list for val in sublist])

	if len(id_vel_list) != 0:
		id_vel_list = reducelist(id_vel_list, 3 * dv, 1) # removes duplicates
		id_vel_list = sorted([val for sublist in id_vel_list for val in sublist])
		return id_vel_list
	else:
		# print('No interesting velocities identified')
		return None
def derivative(vel_axis = None, spectrum = None, _range = 20): # returns dx
	extra = [0] * int(_range / 2) # _range will be even
	dx = []

	for start in range(int(len(vel_axis) - _range)):
		x = vel_axis[start:int(start + _range + 1)]
		y = spectrum[start:int(start + _range + 1)]

		guess = [(y[0] - y[-1]) / (x[0] - x[-1]), 0]
		parinfo = [	{'parname':'gradient','step':0.0001, 'limited': [1, 1], 'limits': [-20., 20.]}, 
					{'parname':'y intercept','step':0.001, 'limited': [1, 1], 'limits': [-1000., 1000.]}]
		fa = {'x': x, 'y': y}
		
		mp = mpfit(mpfitlinear, guess, parinfo = parinfo, functkw = fa, maxiter = 10000, quiet = True)
		gradient = mp.params[0]
		dx.append(gradient)
	dx = np.concatenate((extra, dx, extra))
	return dx
def mpfitlinear(a = None, fjac = None, x = None, y = None): # returns [0, residuals]
	'''
	x and y should be small arrays of length '_range' (from parent function). 
	'''
	[m, c] = a # gradient and y intercept of line
	model_y = m * np.array(x) + c
	residuals = (np.array(y) - model_y)

	return [0, residuals]
def findrms(spectrum = None): # returns rms
	x = len(spectrum)
	a = int(x / 10)
	rms_list = []
	for _set in range(9):
		rms = np.std(spectrum[(_set * a):(_set * a) + (2 * a)])
		rms_list.append(rms)
	median_rms = np.median(rms_list)
	return median_rms
def zeros(x_axis = None, y_axis = None, y_rms = None): # returns boolean array len = len(x)
	'''
	produces a boolean array of whether or not an x value is a zero
	'''	
	gradient_min = abs(2.* y_rms / (x_axis[10] - x_axis[0]))

	zeros = np.zeros(len(x_axis))
	for x in range(5, int(len(x_axis) - 5)):
		x_axis_subset = x_axis[x-5:x+6]
		y_axis_subset = y_axis[x-5:x+6]

		guess = [1., 1.]
		parinfo = [	{'parname':'gradient','step':0.001}, 
					{'parname':'y intercept','step':0.001}]
		fa = {'x': x_axis_subset, 'y': y_axis_subset}

		mp = mpfit(mpfitlinear, guess, parinfo = parinfo, functkw = fa, maxiter = 10000, quiet = True)
		[grad_fit, y_int_fit] = mp.params

		if abs(grad_fit) >= gradient_min:

			# find y values on either side of x to test sign. True = pos
			if grad_fit * x_axis[x-1] + y_int_fit > 0 and grad_fit * x_axis[x+1] + y_int_fit < 0:
				zeros[x] = 1
			elif grad_fit * x_axis[x-1] + y_int_fit < 0 and grad_fit * x_axis[x+1] + y_int_fit > 0:
				zeros[x] = 1
	return zeros
def reducelist(master_list = None, merge_size = 0.5, group_spacing = 5):	# returns merged and grouped list
	'''
	Merges values in master_list separated by less than merge_size, and groups features separated by 
	less than group_spacing into blended features.
	Returns a list of lists: i.e. [[a], [b, c, d], [e]] where 'a' and 'e' are isolated
	features and 'b', 'c' and 'd' are close enough to overlap in velocity.

	Parameters:
	master_list - list of velocities
	merge_size - any velocities separated by less than this distance will be merged into one velocity. This 
			is performed in 4 stages, first using merge_size / 4 so that the closest velocities are merged 
			first. Merged velocities are replaced by their mean. 
	group_spacing - any velocities separated by less than this value will be grouped together so they can 
			be fit as blended features. Smaller values are likely to prevent the accurate identification 
			of blended features, while larger values will increase running time.

	Returns nested list of velocities:
		reduced_vel_list = [[v1], [v2, v3], [v4]]
		
		where v1 and v4 are isolated features that can be fit independently, but v2 and v3 are close 
		enough in velocity that they must be fit together as a blended feature.
	'''
	try:
		master_list = sorted([val for sublist in master_list for val in sublist])
	except TypeError:
		pass

	master_list = np.array(master_list)
	
	# Step 1: merge based on merge_size
	new_merge_list = np.sort(master_list.flatten())

	for merge in [merge_size / 4, 2 * merge_size / 4, 3 * merge_size / 4, merge_size]:
		new_merge_list = mergefeatures(new_merge_list, merge, 'merge')
	
	# Step 2: identify components likely to overlap to be fit together
	final_merge_list = mergefeatures(new_merge_list, group_spacing, 'group')

	return final_merge_list
def mergefeatures(master_list = None, size = None, action = None): # returns merged or grouped list (action)
	'''
	Does the work for ReduceList

	Parameters:
	master_list - list of velocities generated by AGD()
	size - Distance in km/sec for the given action
	action - Action to perform: 'merge' or 'group'

	Returns nested list of velocities:
		reduced_vel_list = [[v1], [v2, v3], [v4]]
		
		where v1 and v4 are isolated features that can be fit independently, but v2 and v3 are close 
		enough in velocity that they must be fit together as a blended feature.
	'''	
	new_merge_list = []
	check = 0
	while check < len(master_list):
		skip = 1
		single = True

		if action == 'merge':
			while check + skip < len(master_list) and master_list[check + skip] - master_list[check] < size:
				skip += 1
				single = False
			if single == True:
				new_merge_list = np.append(new_merge_list, master_list[check])
			else:
				new_merge_list = np.append(new_merge_list, mean(master_list[check:check + skip]))
			check += skip

		elif action == 'group':
			while check + skip < len(master_list) and master_list[check + skip] - master_list[check + skip - 1] < size:
				skip += 1
			new_merge_list.append(master_list[check:check + skip].tolist())
			check += skip
		else:
			print('Error defining action in MergeFeatures')

	return new_merge_list
#              |
#              |
#            \ | /
#             \|/
#              V
###############################
#                             #
# significant velocity ranges #
#                             #
###############################
#              |
#              |
#            \ | /
#             \|/
#              V
def placegaussians(data = None, Bayes_threshold = 10): # returns final_p
	'''
	mpfit, emcee, decide whether or not to add another gaussian
	'''
	accepted_x_full = []
	total_num_gauss = 0
	plot_num = 0
	for vel_range in data['sig_vel_ranges']:
		print('vel_range: ' + str(vel_range))
		last_accepted_x_full = []
		[min_vel, max_vel] = vel_range
		modified_data = trimdata(data, min_vel, max_vel)

		num_gauss = 1
		keep_going = True
		(null_evidence, _, _) = nullevidence(modified_data)
		print('null_evidence = ' + str(null_evidence))
		prev_evidence = null_evidence

		while keep_going == True:
			nwalkers = 30 * num_gauss
			p0 = p0gen(	vel_range = vel_range, 
						num_gauss = num_gauss, 
						modified_data = modified_data, 
						accepted_x = accepted_x_full,
						last_accepted_x = last_accepted_x_full, 
						nwalkers = nwalkers)
			# print('modified_data input into sampleposterior: ' + str(modified_data))
			# print('num_gauss input into sampleposterior: ' + str(num_gauss))
			# print('p0 input into sampleposterior: ' + str(p0))
			# print('[min_vel, max_vel] input into sampleposterior: ' + str([min_vel, max_vel]))
			# print('nwalkers input into sampleposterior: ' + str(nwalkers))
			(x_flat_chain, x_lnprob) = sampleposterior(	modified_data = modified_data, 
												num_gauss = num_gauss, 
												p0 = p0, 
												vel_range = [min_vel, max_vel],  
												accepted_x = accepted_x_full, 
												nwalkers = nwalkers)
			if len(x_flat_chain) != 0:
				(current_x_full, current_evidence) = bestparams(x_flat_chain, x_lnprob)
				plotmodel(data = modified_data, x = [x[1] for x in current_x_full], num_gauss = num_gauss, plot_num = plot_num)
				plot_num += 1
				print('evidence for ' + str(num_gauss) + ' = ' + str(current_evidence))
				if current_evidence - prev_evidence > Bayes_threshold: # working on some tests to refine this
					last_accepted_x_full = current_x_full
					prev_evidence = current_evidence
					num_gauss += 1
					total_num_gauss += 1
				else:
					keep_going = False
			else:
				keep_going = False
		accepted_x_full = list(itertools.chain(accepted_x_full, last_accepted_x_full))
	
	plotmodel(data = modified_data, x = [x[1] for x in accepted_x_full], num_gauss = total_num_gauss, plot_num = 'Final')
	return accepted_x_full
def trimdata(data = None, min_vel = None, max_vel = None): # returns modified_data
	
	data_temp = copy.deepcopy(data)

	data_temp['interesting_vel'] = [x for x in data_temp['interesting_vel'] if x >= min_vel and x <= max_vel]

	vel_1612 = np.array(data_temp['vel_axis']['1612'])
	vel_1665 = np.array(data_temp['vel_axis']['1665'])
	vel_1667 = np.array(data_temp['vel_axis']['1667'])
	vel_1720 = np.array(data_temp['vel_axis']['1720'])

	tau_1612 = np.array(data_temp['tau_spectrum']['1612'])
	tau_1665 = np.array(data_temp['tau_spectrum']['1665'])
	tau_1667 = np.array(data_temp['tau_spectrum']['1667'])
	tau_1720 = np.array(data_temp['tau_spectrum']['1720'])
	
	if data['Texp_spectrum']['1665'] != []:
		Texp_1612 = np.array(data_temp['Texp_spectrum']['1612'])
		Texp_1665 = np.array(data_temp['Texp_spectrum']['1665'])
		Texp_1667 = np.array(data_temp['Texp_spectrum']['1667'])
		Texp_1720 = np.array(data_temp['Texp_spectrum']['1720'])

	min_vel -= 10.
	max_vel += 10.

	mini_1612 = np.amin([np.argmin(np.abs(vel_1612 - min_vel)), np.argmin(np.abs(vel_1612 - max_vel))])
	maxi_1612 = np.amax([np.argmin(np.abs(vel_1612 - min_vel)), np.argmin(np.abs(vel_1612 - max_vel))])
	mini_1665 = np.amin([np.argmin(np.abs(vel_1665 - min_vel)), np.argmin(np.abs(vel_1665 - max_vel))])
	maxi_1665 = np.amax([np.argmin(np.abs(vel_1665 - min_vel)), np.argmin(np.abs(vel_1665 - max_vel))])	
	mini_1667 = np.amin([np.argmin(np.abs(vel_1667 - min_vel)), np.argmin(np.abs(vel_1667 - max_vel))])
	maxi_1667 = np.amax([np.argmin(np.abs(vel_1667 - min_vel)), np.argmin(np.abs(vel_1667 - max_vel))])
	mini_1720 = np.amin([np.argmin(np.abs(vel_1720 - min_vel)), np.argmin(np.abs(vel_1720 - max_vel))])
	maxi_1720 = np.amax([np.argmin(np.abs(vel_1720 - min_vel)), np.argmin(np.abs(vel_1720 - max_vel))])

	data_temp['vel_axis']['1612'] = vel_1612[mini_1612:maxi_1612 + 1]
	data_temp['vel_axis']['1665'] = vel_1665[mini_1665:maxi_1665 + 1]
	data_temp['vel_axis']['1667'] = vel_1667[mini_1667:maxi_1667 + 1]
	data_temp['vel_axis']['1720'] = vel_1720[mini_1720:maxi_1720 + 1]

	data_temp['tau_spectrum']['1612'] = tau_1612[mini_1612:maxi_1612 + 1]
	data_temp['tau_spectrum']['1665'] = tau_1665[mini_1665:maxi_1665 + 1]
	data_temp['tau_spectrum']['1667'] = tau_1667[mini_1667:maxi_1667 + 1]
	data_temp['tau_spectrum']['1720'] = tau_1720[mini_1720:maxi_1720 + 1]

	if data['Texp_spectrum']['1665'] != []:
		data_temp['Texp_spectrum']['1612'] = Texp_1612[mini_1612:maxi_1612 + 1]
		data_temp['Texp_spectrum']['1665'] = Texp_1665[mini_1665:maxi_1665 + 1]
		data_temp['Texp_spectrum']['1667'] = Texp_1667[mini_1667:maxi_1667 + 1]
		data_temp['Texp_spectrum']['1720'] = Texp_1720[mini_1720:maxi_1720 + 1]

	return data_temp
# Find null evidence
def nullevidence(modified_data = None): # returns null_evidence
	nwalkers = 20
	burn_iterations = 100
	final_iterations = 50
	if modified_data['Texp_spectrum']['1665'] != []:
		ndim = 8
	else:
		ndim = 4
	p0 = np.random.uniform(-1e-4,1e-4, size = (nwalkers, ndim))

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobnull, args = [modified_data])
	pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
	sampler.reset()
	sampler.run_mcmc(pos, final_iterations)
	# for a in range(len(sampler.chain)):
	# 	for b in range(len(sampler.chain[a])):
	# 		print('lnprob: ' + str(sampler.lnprobability[a][b]))
	# 		print(str(sampler.chain[a][b]))

	plotchain(modified_data = modified_data, chain = sampler.chain, phase = 'Null')
	(params, evidence) = bestparams(chain = sampler.flatchain, lnprob = sampler.flatlnprobability)
	# print('params for null: ' + str(params))

	plt.figure()
	plt.plot(modified_data['vel_axis']['1612'], modified_data['tau_spectrum']['1612'], color = 'blue')
	plt.plot(modified_data['vel_axis']['1665'], modified_data['tau_spectrum']['1665'], color = 'green')
	plt.plot(modified_data['vel_axis']['1667'], modified_data['tau_spectrum']['1667'], color = 'red')
	plt.plot(modified_data['vel_axis']['1720'], modified_data['tau_spectrum']['1720'], color = 'cyan')
	plt.hlines(params[0][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'blue')
	plt.hlines(params[1][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'green')
	plt.hlines(params[2][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'red')
	plt.hlines(params[3][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'cyan')
	plt.title('Null Model')
	plt.savefig('Plots/null_model.pdf')
	plt.close()
	return (evidence, sampler.flatchain, sampler.flatlnprobability)
def lnprobnull(x = None, modified_data = None): # returns lnprobnull
	lnprprior = lnprpriornull(yint = x, modified_data = modified_data)
	np.zeros(len(modified_data['tau_spectrum']['1612']))

	lnllh_tau_1612 = lnlikelihood(model = np.zeros(len(modified_data['tau_spectrum']['1612'])) + x[0], spectrum = modified_data['tau_spectrum']['1612'], spectrum_rms = modified_data['tau_rms']['1612'])
	lnllh_tau_1665 = lnlikelihood(model = np.zeros(len(modified_data['tau_spectrum']['1665'])) + x[0], spectrum = modified_data['tau_spectrum']['1665'], spectrum_rms = modified_data['tau_rms']['1665'])
	lnllh_tau_1667 = lnlikelihood(model = np.zeros(len(modified_data['tau_spectrum']['1667'])) + x[0], spectrum = modified_data['tau_spectrum']['1667'], spectrum_rms = modified_data['tau_rms']['1667'])
	lnllh_tau_1720 = lnlikelihood(model = np.zeros(len(modified_data['tau_spectrum']['1720'])) + x[0], spectrum = modified_data['tau_spectrum']['1720'], spectrum_rms = modified_data['tau_rms']['1720'])

	lnllh = np.sum([lnllh_tau_1612, lnllh_tau_1665, lnllh_tau_1667, lnllh_tau_1720])

	if modified_data['Texp_spectrum']['1665'] != []:
		lnllh_Texp_1612 = lnlikelihood(model = np.zeros(len(modified_data['Texp_spectrum']['1612'])) + x[0], spectrum = modified_data['Texp_spectrum']['1612'], spectrum_rms = modified_data['Texp_rms']['1612'])
		lnllh_Texp_1665 = lnlikelihood(model = np.zeros(len(modified_data['Texp_spectrum']['1665'])) + x[0], spectrum = modified_data['Texp_spectrum']['1665'], spectrum_rms = modified_data['Texp_rms']['1665'])
		lnllh_Texp_1667 = lnlikelihood(model = np.zeros(len(modified_data['Texp_spectrum']['1667'])) + x[0], spectrum = modified_data['Texp_spectrum']['1667'], spectrum_rms = modified_data['Texp_rms']['1667'])
		lnllh_Texp_1720 = lnlikelihood(model = np.zeros(len(modified_data['Texp_spectrum']['1720'])) + x[0], spectrum = modified_data['Texp_spectrum']['1720'], spectrum_rms = modified_data['Texp_rms']['1720'])

		lnllh += np.sum([lnllh_Texp_1612, lnllh_Texp_1665, lnllh_Texp_1667, lnllh_Texp_1720])
	return lnprprior + lnllh
def lnprpriornull(yint = None, modified_data = None): # returns lnprpriornull

	[yint_tau_1612, yint_tau_1665, yint_tau_1667, yint_tau_1720] = yint[:4]

	lnprprior_tau_1612 = -np.log(np.sqrt(2*np.pi) * modified_data['tau_rms']['1612']) - ((yint_tau_1612**2.) / (2. * modified_data['tau_rms']['1612']**2))
	lnprprior_tau_1665 = -np.log(np.sqrt(2*np.pi) * modified_data['tau_rms']['1665']) - ((yint_tau_1665**2.) / (2. * modified_data['tau_rms']['1665']**2))
	lnprprior_tau_1667 = -np.log(np.sqrt(2*np.pi) * modified_data['tau_rms']['1667']) - ((yint_tau_1667**2.) / (2. * modified_data['tau_rms']['1667']**2))
	lnprprior_tau_1720 = -np.log(np.sqrt(2*np.pi) * modified_data['tau_rms']['1720']) - ((yint_tau_1720**2.) / (2. * modified_data['tau_rms']['1720']**2))
	# print('yint:\t' + str(yint) + '\tlnprpriors:\t' + str([lnprprior_tau_1612, lnprprior_tau_1665, lnprprior_tau_1667, lnprprior_tau_1720]))

	lnprprior = np.sum([lnprprior_tau_1612, lnprprior_tau_1665, lnprprior_tau_1667, lnprprior_tau_1720])

	if modified_data['Texp_spectrum']['1665'] != []:
		[yint_Texp_1612, yint_Texp_1665, yint_Texp_1667, yint_Texp_1720] = yint[4:]

		lnprprior_Texp_1612 = -np.log(np.sqrt(2*np.pi) * modified_data['Texp_rms']['1612']) - ((yint_Texp_1612**2.) / (2. * modified_data['Texp_rms']['1612']**2))
		lnprprior_Texp_1665 = -np.log(np.sqrt(2*np.pi) * modified_data['Texp_rms']['1665']) - ((yint_Texp_1665**2.) / (2. * modified_data['Texp_rms']['1665']**2))
		lnprprior_Texp_1667 = -np.log(np.sqrt(2*np.pi) * modified_data['Texp_rms']['1667']) - ((yint_Texp_1667**2.) / (2. * modified_data['Texp_rms']['1667']**2))
		lnprprior_Texp_1720 = -np.log(np.sqrt(2*np.pi) * modified_data['Texp_rms']['1720']) - ((yint_Texp_1720**2.) / (2. * modified_data['Texp_rms']['1720']**2))

		lnprprior += np.sum([lnprprior_Texp_1612, lnprprior_Texp_1665, lnprprior_Texp_1667, lnprprior_Texp_1720])

	return lnprprior
def lnlikelihood(model = None, spectrum = None, spectrum_rms = None): # returns lnlikelihood
	return -len(spectrum) * np.log(spectrum_rms * np.sqrt(2. * np.pi)) - (np.sum((np.array(model) - np.array(spectrum))**2.) / (2. * spectrum_rms**2.))
def bestparams(chain = None, lnprob = None): # returns ([-sig, med, +sig] for all variables in chain, evidence)
	'''
	Tested and verified 22/3/19
	'''
	# print('chain for bestparams:')
	# for x in chain:
	# 	print(x)
	# print('\nlnprob for bestparams:')
	# print(lnprob)
	num_steps = len(chain)
	num_param = len(chain[0])

	final_array = [list(reversed(sorted(lnprob)))]
	final_darray = [list(reversed(sorted(lnprob)))]

	for param in range(num_param):
		param_chain = [chain[x][param] for x in range(num_steps)]
		final_array = np.concatenate((final_array, [[x for _,x in list(reversed(sorted(zip(lnprob, param_chain))))]])) # this makes an array with [[lnprob], [param1 iterations], etc] all sorted in descending order of lnprob (seems ok)
		zipped = sorted(zip(param_chain, lnprob))
		sorted_param_chain, sorted_lnprob = zip(*zipped) # then sort all by this one parameter

		dparam_chain = [0] + [sorted_param_chain[x] - sorted_param_chain[x-1] for x in range(1, len(sorted_param_chain))] # calculate dparam
		sorted_dparam_chain = [[x for _,x in list(reversed(sorted(zip(sorted_lnprob, dparam_chain))))]] # put back into correct order
		final_darray = np.concatenate((final_darray, sorted_dparam_chain), axis = 0) # this makes an array with [[lnprob], [dparam1 iterations], etc] all sorted in descending order of lnprob (seems ok)








		######
		# This needs to be fixed


	accumulated_evidence = np.zeros(num_steps)
	for step in range(num_steps):
		# multiply all dparam values
		param_volume = 1
		for param in range(1, len(final_darray)):
			param_volume *= final_darray[param][step]
		if param_volume != 0:
			contribution_to_lnevidence = np.log(param_volume) + final_darray[0][step]
		else:
			contribution_to_lnevidence = -np.inf
		if step == 0:
			accumulated_evidence[step] = contribution_to_lnevidence
		else:
			accumulated_evidence[step] = np.logaddexp(accumulated_evidence[step - 1], contribution_to_lnevidence)

	#############









	total_evidence = accumulated_evidence[-1]
	evidence_68 = total_evidence + np.log(0.6825)

	sigma_index = np.argmin(abs(accumulated_evidence - evidence_68))

	results = np.zeros([num_param, 3])

	for param in range(num_param):
		results[param][0] = np.amin(final_array[param + 1][:sigma_index + 1])
		results[param][1] = np.median(final_array[param + 1])
		results[param][2] = np.amax(final_array[param + 1][:sigma_index + 1])
	# print('results: ' + str(results))
	# print('total_evidence: ' + str(total_evidence))
	return (results, total_evidence)
# initial fit using mpfit
def p0gen(vel_range = None, num_gauss = None, modified_data = None, accepted_x = [], last_accepted_x = [], nwalkers = None): # returns p0
	if accepted_x != []:
		ndim = np.sum([type(a) == bool and a == True for a in modified_data['parameter_list']]) + 1
		accepted_x = [x[1] for x in accepted_x]
		accepted_params = molex(x = accepted_x, modified_data = modified_data, num_gauss = int(len(accepted_x) / ndim))
	else:
		accepted_params = []
	if last_accepted_x != []:
		ndim = np.sum([type(a) == bool and a == True for a in modified_data['parameter_list']]) + 1
		last_accepted_x = [x[1] for x in last_accepted_x]
		last_accepted_params = molex(x = last_accepted_x, modified_data = modified_data, num_gauss = int(len(last_accepted_x) / ndim))
	else:
		last_accepted_params = []
	
	if num_gauss > 2: # i.e. too many to fit every combo
		vel_list = []
		for ind in range(0, len(last_accepted_params), 10):
			np.append(vel_list, last_accepted_params[ind])
		vel_list = tuple(vel_list)
		if len(modified_data['interesting_vel']) != 0:
			interesting_vel = [tuple([x]) for x in modified_data['interesting_vel']]
		else:
			interesting_vel = np.arange(vel_range[0], vel_range[1], (vel_range[1] - vel_range[0]) / 10.)
		vel_combos = list(itertools.product([vel_list], interesting_vel))
		for x in range(len(vel_combos)):
			vel_combos[x] = [d for e in vel_combos[x] for d in e]
	else:
		vel_combos = list(itertools.product(modified_data['interesting_vel'], repeat = num_gauss))
	vel_combos = [sorted(x) for x in vel_combos]

	standard_parinfo = [{'parname': 'vel', 'fixed': False, 'step': 1.e-1, 'limited': [1,1], 'limits': vel_range}, 
						{'parname': 'logTgas', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': logTgas_range}, 
						{'parname': 'lognH2', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': lognH2_range},
						{'parname': 'logNOH', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': logNOH_range},
						{'parname': 'fortho', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': fortho_range},
						{'parname': 'FWHM', 'fixed': False, 'step': 1.e-1, 'limited': [1,1], 'limits': FWHM_range},
						{'parname': 'Av', 'fixed': False, 'step': 1.e-1, 'limited': [1,1], 'limits': Av_range},
						{'parname': 'logxOH', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': logxOH_range},
						{'parname': 'logxHe', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': logxHe_range},
						{'parname': 'logxe', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': logxe_range},
						{'parname': 'logTdint', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': logTdint_range},
						{'parname': 'logTd', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': logTd_range}]
	
	range_list = [vel_range, logTgas_range, lognH2_range, logNOH_range, fortho_range, FWHM_range, Av_range, logxOH_range, logxHe_range, logxe_range, logTdint_range, logTd_range]
	guess_base = [True] + modified_data['parameter_list']
	guess_base = [np.mean(range_list[a]) if type(guess_base[a]) == bool and guess_base[a] == True else guess_base[a] for a in range(len(guess_base))]
	guess = guess_base * num_gauss

	np.array([[np.mean(vel_range), np.mean(logTgas_range), np.mean(lognH2_range), np.mean(logNOH_range), np.mean(fortho_range), np.mean(FWHM_range), np.mean(Av_range), np.mean(logxOH_range), np.mean(logxHe_range), np.mean(logxe_range), np.mean(logTdint_range), np.mean(logTd_range)] for b in range(num_gauss)]).flatten()
	
	parinfo = standard_parinfo * num_gauss
	full_param_list = ([True] + modified_data['parameter_list']) * num_gauss
	[best_p, best_llh] = [[], -np.inf]
	
	for a in range(len(parinfo)):
		if type(full_param_list[a]) != bool:
			parinfo[a]['fixed'] = True

	for vel_combo in vel_combos:
		for vel_ind in range(len(vel_combo)):
			guess[int(vel_ind * 12)] = vel_combo[vel_ind]
		for c in range(len(full_param_list)):
			if type(full_param_list[c]) != bool:
				guess[c] = full_param_list[c]
		fa = {'modified_data': modified_data, 'vel_range': vel_range, 'num_gauss': num_gauss, 'accepted_params': accepted_params}
		mp = mpfit(mpfitp0, guess, parinfo = parinfo, functkw = fa, maxiter = 1000, quiet = True)
		fitted_p = mp.params
		llh = lnprob(	modified_data = modified_data, 
						p = fitted_p, 
						vel_range = vel_range, 
						num_gauss = num_gauss, 
						accepted_params = accepted_params)
		if llh > best_llh:
			[best_p, best_llh] = [fitted_p, llh]
	
	if best_p != []:
		best_x = xlist(p = best_p, data = modified_data, num_gauss = num_gauss)
		# print('base of p0: ' + str(best_x))
		p0 = [[x * np.random.uniform(0.999, 1.001) for x in best_x] for y in range(nwalkers)]
	else:
		p0_0 = [	np.random.uniform(vel_range[0], vel_range[1]), 
				np.random.uniform(logTgas_range[0], logTgas_range[1]), 
				np.random.uniform(lognH2_range[0], lognH2_range[1]), 
				np.random.uniform(logNOH_range[0], logNOH_range[1]), 
				np.random.uniform(fortho_range[0], fortho_range[1]), 
				np.random.uniform(FWHM_range[0], FWHM_range[1]), 
				np.random.uniform(Av_range[0], Av_range[1]), 
				np.random.uniform(logxOH_range[0], logxOH_range[1]), 
				np.random.uniform(logxHe_range[0], logxHe_range[1]), 
				np.random.uniform(logxe_range[0], logxe_range[1]), 
				np.random.uniform(logTdint_range[0], logTdint_range[1]), 
				np.random.uniform(logTd_range[0], logTd_range[1])] * num_gauss
		p0_0 = [p0_0[a] for a in range(len(p0_0)) if type(full_param_list[a]) == bool and full_param_list[a] == True]
		# print('base of p0: ' + str(p0_0))
		p0 = [[a * np.random.uniform(0.999, 1.001) for a in p0_0] for b in range(nwalkers)]
	return p0
def molex(p = [], x = None, modified_data = None, num_gauss = None): # returns params
	if p == []:
		p = plist(x, modified_data, num_gauss)

	output = list(np.zeros(int(10 * num_gauss)))

	for gaussian in range(num_gauss):

		[vel, logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd] = p[gaussian * 12:(gaussian + 1) * 12]

		# check if temp.txt exists - erase
		try: 
			subprocess.run('rm temp.txt', shell = True, stderr=subprocess.DEVNULL)
		except:
			pass

		# Write ohslab.in file
		with open('oh_slab.in', 'w') as oh_slab:
			# Based on template! Don't touch!
			oh_slab.write('\'temp.txt\'\nF\nF\n' + str(10**logTgas) + '\n' + str(FWHM) + '\n' + str(fortho) + '\n')
			oh_slab.write(str(10**logxOH) + '\n' + str(10**logxHe) + '\n' + str(10**logxe) + '\nF\n' + str(10**logTdint) + '\n' + str(10**logTd) + '\n')
			oh_slab.write(str(Av) + '\nF\n2\n' + str(lognH2) + '\n' + str(lognH2) + '\n0.05\n' + str(logNOH) + '\n')
			oh_slab.write(str(logNOH) + '\n0.1\n3,2\n3,1\n4,2\n4,1\n -1, -1\nF\nT\n1.0\n4\n0.1\n32\n1\n1.d-6\n1.d-16')
			oh_slab.write('\nT\n20\n1.d-6\nF\n1\n17\nF\nF\nF\nF\nF\nF\nF\nF\nF\nF\nF\nF\nF\nT\nF\nF\nF\n')

		subprocess.call('make ohslab', shell = True, stdout=subprocess.DEVNULL)
		subprocess.run(['/Users/anitahafner/Documents/Marks_OH_ex_code/ohslab'], shell = True, stdout=subprocess.DEVNULL)
		try:
			with open('temp.txt', 'r') as f:
				for line in islice(f, 33, 34):
					Tex_1612, tau0_1612, Tex_1665, tau0_1665, Tex_1667, tau0_1667, Tex_1720, tau0_1720 = line.split()[3:11]
			[Tex_1612, tau0_1612, Tex_1665, tau0_1665, Tex_1667, tau0_1667, Tex_1720, tau0_1720] = [float(Tex_1612), float(tau0_1612), float(Tex_1665), float(tau0_1665), float(Tex_1667), float(tau0_1667), float(Tex_1720), float(tau0_1720)]
		except:
			[Tex_1612, tau0_1612, Tex_1665, tau0_1665, Tex_1667, tau0_1667, Tex_1720, tau0_1720] = np.zeros(8)
		Texp_1612 = Texp(tau = tau0_1612, Tbg = modified_data['Tbg']['1612'], Tex = Tex_1612)
		Texp_1665 = Texp(tau = tau0_1665, Tbg = modified_data['Tbg']['1665'], Tex = Tex_1665)
		Texp_1667 = Texp(tau = tau0_1667, Tbg = modified_data['Tbg']['1667'], Tex = Tex_1667)
		Texp_1720 = Texp(tau = tau0_1720, Tbg = modified_data['Tbg']['1720'], Tex = Tex_1720)
		output[int(10 * gaussian):int(10 * (gaussian + 1))] = [vel, FWHM, tau0_1612, tau0_1665, tau0_1667, tau0_1720, Texp_1612, Texp_1665, Texp_1667, Texp_1720]
		# erase temp.txt
		subprocess.run('rm temp.txt', shell = True, stderr = subprocess.DEVNULL)

	return output
def plist(x = None, data = None, num_gauss = None): # returns p
	p = ([True] + list(data['parameter_list'])) * num_gauss
	x_counter = 0
	for a in range(len(p)):
		if type(p[a]) == bool and p[a] == True:
			p[a] = x[x_counter]
			x_counter += 1
	return p
def xlist(p = None, data = None, num_gauss = None): # returns x
	parameter_list = ([True] + list(data['parameter_list'])) * num_gauss
	x = [p[a] for a in range(len(p)) if type(parameter_list[a]) == bool and parameter_list[a] == True]
	return x
def Texp(tau = None, Tbg = None, Tex = None): # returns Texp
	Texp = (Tex - Tbg) * (1 - np.exp(-tau))
	return Texp
def mpfitp0(p = None, fjac = None, modified_data = None, vel_range = None, num_gauss = None, accepted_params = []): # returns [0, residuals]
	log.write(str(p) + '\n')
	params = molex(p = p, modified_data = modified_data, num_gauss = num_gauss)
	(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = makemodel(params, modified_data, accepted_params)
	if modified_data['Texp_spectrum']['1665'] != []:
		residuals = np.concatenate((
					(tau_m_1612 - modified_data['tau_spectrum']['1612']) / modified_data['tau_rms']['1612'], 
					(tau_m_1665 - modified_data['tau_spectrum']['1665']) / modified_data['tau_rms']['1665'], 
					(tau_m_1667 - modified_data['tau_spectrum']['1667']) / modified_data['tau_rms']['1667'], 
					(tau_m_1720 - modified_data['tau_spectrum']['1720']) / modified_data['tau_rms']['1720'], 
					(Texp_m_1612 - modified_data['Texp_spectrum']['1612']) / modified_data['Texp_rms']['1612'], 
					(Texp_m_1665 - modified_data['Texp_spectrum']['1665']) / modified_data['Texp_rms']['1665'], 
					(Texp_m_1667 - modified_data['Texp_spectrum']['1667']) / modified_data['Texp_rms']['1667'], 
					(Texp_m_1720 - modified_data['Texp_spectrum']['1720']) / modified_data['Texp_rms']['1720']))
	else:
		residuals = np.concatenate((
					(tau_m_1612 - modified_data['tau_spectrum']['1612']) / modified_data['tau_rms']['1612'], 
					(tau_m_1665 - modified_data['tau_spectrum']['1665']) / modified_data['tau_rms']['1665'], 
					(tau_m_1667 - modified_data['tau_spectrum']['1667']) / modified_data['tau_rms']['1667'], 
					(tau_m_1720 - modified_data['tau_spectrum']['1720']) / modified_data['tau_rms']['1720']))
	return [0, residuals]
def makemodel(params = None, modified_data = None, accepted_params = []): # returns tau and Texp models
	vel_1612 = modified_data['vel_axis']['1612']
	vel_1665 = modified_data['vel_axis']['1665']
	vel_1667 = modified_data['vel_axis']['1667']
	vel_1720 = modified_data['vel_axis']['1720']
	
	if accepted_params != []:
			(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = makemodel(params = accepted_params, modified_data = modified_data)
	else:
		tau_m_1612 = np.zeros(len(vel_1612))
		tau_m_1665 = np.zeros(len(vel_1665))
		tau_m_1667 = np.zeros(len(vel_1667))
		tau_m_1720 = np.zeros(len(vel_1720))

		Texp_m_1612 = np.zeros(len(vel_1612))
		Texp_m_1665 = np.zeros(len(vel_1665))
		Texp_m_1667 = np.zeros(len(vel_1667))
		Texp_m_1720 = np.zeros(len(vel_1720))
	
	for component in range(int(len(params) / 10)): 
		[vel, FWHM, tau_1612, tau_1665, tau_1667, tau_1720, Texp_1612, Texp_1665, Texp_1667, Texp_1720] = params[component * 10:component * 10 + 10]

		tau_m_1612 += gaussian(mean = vel, FWHM = FWHM, height = tau_1612)(vel_1612)
		tau_m_1665 += gaussian(mean = vel, FWHM = FWHM, height = tau_1665)(vel_1665)
		tau_m_1667 += gaussian(mean = vel, FWHM = FWHM, height = tau_1667)(vel_1667)
		tau_m_1720 += gaussian(mean = vel, FWHM = FWHM, height = tau_1720)(vel_1720)

		Texp_m_1612 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1612)(vel_1612)
		Texp_m_1665 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1665)(vel_1665)
		Texp_m_1667 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1667)(vel_1667)
		Texp_m_1720 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1720)(vel_1720)

	return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720)	
def gaussian(mean = None, FWHM = None, height = None, sigma = None, amp = None): # returns lambda gaussian
	'''
	Generates a gaussian profile with the given parameters.
	'''
	if sigma == None:
		sigma = FWHM / (2. * math.sqrt(2. * np.log(2.)))

	if height == None:
		height = amp / (sigma * math.sqrt(2.* math.pi))

	return lambda x: height * np.exp(-((x - mean)**2.) / (2.*sigma**2.))
# sample posterior using emcee
def sampleposterior(modified_data = None, num_gauss = None, p0 = None, vel_range = None, accepted_x = [], nwalkers = None): # returns (flat_chain, 
	ndim = num_gauss
	for a in modified_data['parameter_list']:
		if type(a) == bool and a == True:
			ndim += num_gauss

	burn_iterations = 30
	final_iterations = 50
	if accepted_x != []:
		accepted_x = [x[1] for x in accepted_x]
		accepted_params = molex(x = accepted_x, modified_data = modified_data, num_gauss = num_gauss)
	else:
		accepted_params = []
	args = [modified_data, [], vel_range, num_gauss, accepted_params]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)
	# burn
	[burn_run, test_result] = [0, 'Fail']
	while burn_run <= 2 and test_result == 'Fail':
		try:
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		except ValueError: # sometimes there is an error within emcee (random.randint)
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		# test convergence
		(test_result, p0) = convergencetest(sampler_chain = sampler.chain, num_gauss = num_gauss, pos = pos)
		burn_run += 1
	plotchain(modified_data = modified_data, chain = sampler.chain, phase = 'Burn')
	# final run
	sampler.reset()
	sampler.run_mcmc(pos, final_iterations)
	# remove steps where lnprob = -np.inf
	flat_chain = [sampler.flatchain[a] for a in range(len(sampler.flatchain)) if sampler.flatlnprobability[a] != -np.inf]
	flat_lnprob = [a for a in sampler.flatlnprobability if a != -np.inf]
	return (np.array(flat_chain), np.array(flat_lnprob))
def lnprob(x = None, modified_data = None, p = [], vel_range = None, num_gauss = None, accepted_params = []): # returns lnprob
	'''
	need to add a test of whether or not the velocities are in order
	'''
	if p == []:
		p = plist(x, modified_data, num_gauss)
	prior = lnprprior(modified_data = modified_data, p = p, vel_range = vel_range, num_gauss = num_gauss)
	# if prior != -np.inf:
	params = molex(p = p, modified_data = modified_data, num_gauss = num_gauss)
	models = makemodel(params = params, modified_data = modified_data, accepted_params = accepted_params)
	(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = models
	if modified_data['Texp_spectrum']['1665'] != []:
		spectra = [	modified_data['tau_spectrum']['1612'], modified_data['tau_spectrum']['1665'], 
					modified_data['tau_spectrum']['1667'], modified_data['tau_spectrum']['1720'], 
					modified_data['Texp_spectrum']['1612'], modified_data['Texp_spectrum']['1665'], 
					modified_data['Texp_spectrum']['1667'], modified_data['Texp_spectrum']['1720']]
		rms = [		modified_data['tau_rms']['1612'], modified_data['tau_rms']['1665'], 
					modified_data['tau_rms']['1667'], modified_data['tau_rms']['1720'], 
					modified_data['Texp_rms']['1612'], modified_data['Texp_rms']['1665'], 
					modified_data['Texp_rms']['1667'], modified_data['Texp_rms']['1720']]
	else:
		spectra = [	modified_data['tau_spectrum']['1612'], modified_data['tau_spectrum']['1665'], 
					modified_data['tau_spectrum']['1667'], modified_data['tau_spectrum']['1720']]
		rms = [		modified_data['tau_rms']['1612'], modified_data['tau_rms']['1665'], 
					modified_data['tau_rms']['1667'], modified_data['tau_rms']['1720']]
	llh = prior 
	for a in range(len(spectra)):
		llh += lnlikelihood(model = models[a], spectrum = spectra[a], spectrum_rms = rms[a])
	# if num_gauss > 1:
	# 	log.write(str(p) + '\t' + str(prior) + '\t' + str(llh - prior) + '\n')
	# 	log.flush()
	return llh
	# else:
	# 	return -np.inf
def lnprprior(modified_data = None, p = [], vel_range = None, num_gauss = None): # returns lnprprior
	parameter_list = [True] + modified_data['parameter_list'] # add velocity
	lnprprior = 0
	for gauss in range(num_gauss):
		[vel, logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd] = p[int(gauss * 12):int((gauss + 1) * 12)]
		priors = [priorvel(vel, vel_range = vel_range), priorlogTgas(logTgas), priorlognH2(lognH2), priorlogNOH(logNOH), priorfortho(fortho), 
				priorFWHM(FWHM), priorAv(Av), priorlogxOH(logxOH), priorlogxHe(logxHe), priorlogxe(logxe), priorlogTdint(logTdint), priorlogTd(logTd)]
		priors = [priors[a] for a in range(len(priors)) if type(parameter_list[a]) == bool and parameter_list[a] == True]
		lnprprior = lnprprior + np.sum(priors)
	return lnprprior
def priorvel(vel = None, vel_range = None): # returns lnpriorvel
	if vel >= vel_range[0] and vel <= vel_range[1]:
		return -np.log(vel_range[1] - vel_range[0])
	else:
		return -np.inf
def priorlogTgas(logTgas = None, logTgas_range = logTgas_range): # returns lnpriorlogTgas
	if logTgas >= logTgas_range[0] and logTgas <= logTgas_range[1]:
		return -np.log(logTgas_range[1] - logTgas_range[0])
	else:
		return -np.inf
def priorlognH2(lognH2 = None, lognH2_range = lognH2_range): # returns lnpriorlognH2
	if lognH2 >= lognH2_range[0] and lognH2 <= lognH2_range[1]:
		return -np.log(lognH2_range[1] - lognH2_range[0])
	else:
		return -np.inf
def priorlogNOH(logNOH = None, logNOH_range = logNOH_range): # returns lnpriorlogNOH
	if logNOH >= logNOH_range[0] and logNOH <= logNOH_range[1]:
		return -np.log(logNOH_range[1] - logNOH_range[0])
	else:
		return -np.inf
def priorfortho(fortho = None, fortho_range = fortho_range): # returns lnpriorfortho
	if fortho >= fortho_range[0] and fortho <= fortho_range[1]:
		return -np.log(fortho_range[1] - fortho_range[0])
	else:
		return -np.inf
def priorFWHM(FWHM = None, FWHM_range = FWHM_range): # returns lnpriorFWHM
	if FWHM >= FWHM_range[0] and FWHM <= FWHM_range[1]:
		return -np.log(FWHM_range[1] - FWHM_range[0])
	else:
		return -np.inf
def priorAv(Av = None, Av_range = Av_range): # returns lnpriorAv
	if Av >= Av_range[0] and Av <= Av_range[1]:
		return -np.log(Av_range[1] - Av_range[0])
	else:
		return -np.inf
def priorlogxOH(logxOH = None, logxOH_range = logxOH_range): # returns lnpriorlogxOH
	if logxOH >= logxOH_range[0] and logxOH <= logxOH_range[1]:
		return -np.log(logxOH_range[1] - logxOH_range[0])
	else:
		return -np.inf
def priorlogxHe(logxHe = None, logxHe_range = logxHe_range): # returns lnpriorlogxHe
	if logxHe >= logxHe_range[0] and logxHe <= logxHe_range[1]:
		return -np.log(logxHe_range[1] - logxHe_range[0])
	else:
		return -np.inf
def priorlogxe(logxe = None, logxe_range = logxe_range): # returns lnpriorlogxe
	if logxe >= logxe_range[0] and logxe <= logxe_range[1]:
		return -np.log(logxe_range[1] - logxe_range[0])
	else:
		return -np.inf
def priorlogTdint(logTdint = None, logTdint_range = logTdint_range): # returns lnpriorlogTdint
	if logTdint >= logTdint_range[0] and logTdint <= logTdint_range[1]:
		return -np.log(logTdint_range[1] - logTdint_range[0])
	else:
		return -np.inf
def priorlogTd(logTd = None, logTd_range = logTd_range): # returns lnpriorlogTd
	if logTd >= logTd_range[0] and logTd <= logTd_range[1]:
		return -np.log(logTd_range[1] - logTd_range[0])
	else:
		return -np.inf
def plotchain(modified_data = None, chain = None, phase = None):
	ndim = chain.shape[2]
	for parameter in range(ndim):
		plt.figure()
		for walker in range(chain.shape[0]):
			plt.plot(range(chain.shape[1]), chain[walker,:,parameter])
		plt.title(modified_data['source_name'] + ' for param ' + str(parameter) + ': burn in')
		# plt.show()
		plt.savefig('Plots/Chain_plot_' + str(phase) + '_' + modified_data['source_name'] + '_' + str(parameter) + '.pdf')
		plt.close()
def convergencetest(sampler_chain = None, num_gauss = None, pos = None): # returns (test_result, final_pos of walkers)
	'''
	sampler_chain has dimensions [nwalkers, iterations, ndim]

	Tests if the variance across chains is comparable to the variance within the chains.

	Returns 'Pass' or 'Fail'
	'''
	model_dim = int(sampler_chain.shape[2] / num_gauss)
	orig_num_walkers = sampler_chain.shape[0]
	counter = 0

	# remove dead walkers
	for walker in reversed(range(sampler_chain.shape[0])):
		if sampler_chain[walker,0,0] == sampler_chain[walker,-1,0]: # first vel doesn't change
			sampler_chain = np.delete(sampler_chain, walker, 0)
			counter += 1
	# replace removed walkers
	if counter > 0 and counter < orig_num_walkers / 2:
		for x in range(counter):
			sampler_chain = np.concatenate((sampler_chain, [sampler_chain[0]]), axis = 0)
	elif counter > orig_num_walkers / 2:
		return ('Fail', pos)

	# test convergence in velocity
	for component in range(num_gauss):

		var_within_chains = np.median([np.var(sampler_chain[x,-25:-1,component * model_dim]) for x in range(sampler_chain.shape[0])])
		var_across_chains = np.median([np.var(sampler_chain[:,-x-1,component * model_dim]) for x in range(24)])
		ratio = max([var_within_chains, var_across_chains]) / min([var_within_chains, var_across_chains])
		max_var = max([var_within_chains, var_across_chains])

		if ratio > 15. and max_var < 1.:
			return ('Fail', sampler_chain[:,-1,:])

	return ('Pass', sampler_chain[:,-1,:])
def plotmodel(data = None, x = None, p = [], params = None, accepted_params = [], num_gauss = None, plot_num = None):
	'''
	Still needs axis labels!
	'''
	if params == None:
		params = molex(p = p, x = x, modified_data = data, num_gauss = num_gauss)
	(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = makemodel(params = params, modified_data = data, accepted_params = accepted_params)
	fig, axes = plt.subplots(nrows = 5, ncols = 2, sharex = True)

	axes[0,0].plot(data['vel_axis']['1612'], data['tau_spectrum']['1612'], color = 'blue', label = '1612 MHz', linewidth = 1)
	axes[0,0].plot(data['vel_axis']['1612'], tau_m_1612, color = 'black', linewidth = 1)
	axes[1,0].plot(data['vel_axis']['1665'], data['tau_spectrum']['1665'], color = 'green', label = '1665 MHz', linewidth = 1)
	axes[1,0].plot(data['vel_axis']['1665'], tau_m_1665, color = 'black', linewidth = 1)
	axes[2,0].plot(data['vel_axis']['1667'], data['tau_spectrum']['1667'], color = 'red', label = '1667 MHz', linewidth = 1)
	axes[2,0].plot(data['vel_axis']['1667'], tau_m_1667, color = 'black', linewidth = 1)
	axes[3,0].plot(data['vel_axis']['1720'], data['tau_spectrum']['1720'], color = 'cyan', label = '1720 MHz', linewidth = 1)
	axes[3,0].plot(data['vel_axis']['1720'], tau_m_1720, color = 'black', linewidth = 1)
	axes[4,0].plot(data['vel_axis']['1612'], data['tau_spectrum']['1612'] - tau_m_1612, color = 'blue', linewidth = 1)
	axes[4,0].plot(data['vel_axis']['1665'], data['tau_spectrum']['1665'] - tau_m_1665, color = 'green', linewidth = 1)
	axes[4,0].plot(data['vel_axis']['1667'], data['tau_spectrum']['1667'] - tau_m_1667, color = 'red', linewidth = 1)
	axes[4,0].plot(data['vel_axis']['1720'], data['tau_spectrum']['1720'] - tau_m_1720, color = 'cyan', linewidth = 1)
	axes[0,1].plot(data['vel_axis']['1612'], Texp_m_1612, color = 'black', linewidth = 1)
	axes[1,1].plot(data['vel_axis']['1665'], Texp_m_1665, color = 'black', linewidth = 1)
	axes[2,1].plot(data['vel_axis']['1667'], Texp_m_1667, color = 'black', linewidth = 1)
	axes[3,1].plot(data['vel_axis']['1720'], Texp_m_1720, color = 'black', linewidth = 1)
	if data['Texp_spectrum']['1665'] != []:
		axes[0,1].plot(data['vel_axis']['1612'], data['Texp_spectrum']['1612'], color = 'blue', label = '1612 MHz', linewidth = 1)
		axes[1,1].plot(data['vel_axis']['1665'], data['Texp_spectrum']['1665'], color = 'green', label = '1665 MHz', linewidth = 1)
		axes[2,1].plot(data['vel_axis']['1667'], data['Texp_spectrum']['1667'], color = 'red', label = '1667 MHz', linewidth = 1)
		axes[3,1].plot(data['vel_axis']['1720'], data['Texp_spectrum']['1720'], color = 'cyan', label = '1720 MHz', linewidth = 1)
		axes[4,1].plot(data['vel_axis']['1612'], data['Texp_spectrum']['1612'] - Texp_m_1612, color = 'blue', linewidth = 1)
		axes[4,1].plot(data['vel_axis']['1665'], data['Texp_spectrum']['1665'] - Texp_m_1665, color = 'green', linewidth = 1)
		axes[4,1].plot(data['vel_axis']['1667'], data['Texp_spectrum']['1667'] - Texp_m_1667, color = 'red', linewidth = 1)
		axes[4,1].plot(data['vel_axis']['1720'], data['Texp_spectrum']['1720'] - Texp_m_1720, color = 'cyan', linewidth = 1)

	# plt.show()
	plt.savefig('Plots/' + data['source_name'] + '_plot_' + str(plot_num) + '.pdf')
	plt.close()


#############################
#                           #
#   M   M        i          #
#   MM MM   aa      n nn    #
#   M M M   aaa  i  nn  n   #
#   M   M  a  a  i  n   n   #
#   M   M   aaa  i  n   n   #
#                           #
#############################

def Main(source_name = None, # prints final_p
	vel_alogxes = None, 
	tau_spectra = None, 
	tau_rms = None, 
	Texp_spectra = None, 
	Texp_rms = None, 
	Tbg = None, 
	quiet = True, 
	Bayes_threshold = 10., 
	con_test_limit = 15, 
	tau_tol = 5, 
	max_cores = 10, 
	test = False, 
	report_type = 'terminal', 
	logTgas = True, 
	lognH2 = True, 
	logNOH = True, 
	fortho = True, 
	FWHM = True, 
	Av = True, 
	logxOH = True, 
	logxHe = True, 
	logxe = True, 
	logTdint = True, 
	logTd = True):
	'''
	Performs Bayesian gaussian Decomposition on velocity spectra of the 2 Pi 3/2 J = 3/2 ground state 
	transitions of OH.

	Parameters:
	source_name - unique identifier for sightline, used in plots, dictionaries etc.
	vel_alogxes - list of velocity alogxes: [vel_axis_1612, vel_axis_1665, vel_axis_1667, vel_axis_1720]
	spectra - list of spectra (brightness temperature or tau): [spectrum_1612, spectrum_1665, spectrum_1667, 
			spectrum_1720]
	rms - list of estimates of rms error in spectra: [rms_1612, rms_1665, rms_1667, rms_1720]. Used by AGD
	expected_min_FWHM - estimate of the minimum full width at half-maximum of features expected in the data 
			in km/sec. Used when categorising features as isolated or blended.

	Returns parameters of gaussian component(s): [vel_1, FWHM_1, height_1612_1, height_1665_1, height_1667_1, 
			height_1720_1, ..., _N] for N components
	'''

	# print('starting Main(source_')
	# initialise data dictionary
	data = {'source_name': source_name, 'vel_axis': {'1612': [], '1665': [], '1667': [], '1720': []}, 'tau_spectrum': {'1612': [], '1665': [], '1667': [], '1720': []}, 'tau_rms': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Texp_spectrum': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Texp_rms': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Tbg': {'1612': [], '1665': [], '1667': [], '1720': []}, 'parameter_list': [logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd], 'sig_vel_ranges': [[]], 'interesting_vel': []}

		###############################################
		#                                             #
		#   Load Data into 'data' dictionary object   #
		#                                             #	
		###############################################

	data['vel_axis']['1612']		= vel_alogxes[0]
	data['tau_spectrum']['1612']	= tau_spectra[0]
	data['tau_rms']['1612']			= tau_rms[0]

	data['vel_axis']['1665']		= vel_alogxes[1]
	data['tau_spectrum']['1665']	= tau_spectra[1]
	data['tau_rms']['1665']			= tau_rms[1]

	data['vel_axis']['1667']		= vel_alogxes[2]
	data['tau_spectrum']['1667']	= tau_spectra[2]
	data['tau_rms']['1667']			= tau_rms[2]

	data['vel_axis']['1720']		= vel_alogxes[3]
	data['tau_spectrum']['1720']	= tau_spectra[3]
	data['tau_rms']['1720']			= tau_rms[3]

	if Texp_spectra != None: #absorption and emission spectra are available (i.e. Arecibo observations)
		
		data['Tbg']['1612']				= Tbg[0]
		data['Tbg']['1665']				= Tbg[1]
		data['Tbg']['1667']				= Tbg[2]
		data['Tbg']['1720']				= Tbg[3]

		data['Texp_rms']['1612']		= Texp_rms[0]
		data['Texp_rms']['1665']		= Texp_rms[1]
		data['Texp_rms']['1667']		= Texp_rms[2]
		data['Texp_rms']['1720']		= Texp_rms[3]

		data['Texp_spectrum']['1612']	= Texp_spectra[0]
		data['Texp_spectrum']['1665']	= Texp_spectra[1]
		data['Texp_spectrum']['1667']	= Texp_spectra[2]
		data['Texp_spectrum']['1720']	= Texp_spectra[3]


	###############################################
	#                                             #
	#         Identify significant ranges         #
	#                                             #
	###############################################

	findranges(data)

	###############################################
	#                                             #
	#                Fit gaussians                #
	#                                             #
	###############################################
	
	final_p = placegaussians(data, Bayes_threshold)

	print(final_p)
