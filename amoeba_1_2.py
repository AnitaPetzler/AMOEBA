# from emcee.utils import MPIPool
from mpfit import mpfit
# from mpmath import mp
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

logTgas_range   = [0., 3.] # molex struggles below 10K, reduce with caution
lognH2_range    = [-2., 7.]
logNOH_range    = [11., 17.]
fortho_range    = [0., 1.]
FWHM_range      = [0.1, 15.]
Av_range        = [0., 10.]
logxOH_range    = [-8., -6.]
logxHe_range    = [-2., 0.]
logxe_range     = [-5., -3.]
logTdint_range  = [0., 3.]
logTd_range     = [0., 3.]


# Variables used throughout:
#
# parameter_list = list of parameters for molex found in dictionary object. Those =True are to be fit
# p = full set of parameters for molex for all gaussians (including vel!)
# x = subset of parameters for molex for all gaussians (the molex parameters we're fitting)
# params = full set of v, FWHM, [tau(x3) or N(x4)] for all gaussians (parameters that relate to the spectra)






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
def findranges(data, num_chan = 1, sigma_tolerance = 1.5): # adds 'interesting_vel' and 'sig_vel_ranges' to dictionary
	# Makes sure that any 'interesting velocities' are included in the identified ranges
	# data['interesting_vel'] = interestingvel(data)
	if data['interesting_vel'] != None:
		sig_vel_list = data['interesting_vel']
	else:
		sig_vel_list = []

	summed_spectra = sumspectra(data)
	bool_summed = [np.mean(summed_spectra[x:x+num_chan]) >= sigma_tolerance for x in range(len(data['vel_axis']['1612']) - num_chan)]

	sig_vel_list += [data['vel_axis']['1612'][x] for x in range(len(bool_summed)) if bool_summed[x] == True]

	# merges closely spaced velocities, groups moderately spaced velocities
	sig_vel_list = reducelist(sig_vel_list)
	sig_vel_ranges = [[x[0], x[-1]] for x in sig_vel_list if x[0] != x[-1]]
	data['sig_vel_ranges'] = sig_vel_ranges
	return data
def interestingvel(data = None): # returns interesting_vel
	id_vel_list = []
	dv = np.abs(data['vel_axis']['1612'][1] - data['vel_axis']['1612'][0])
	# Flag features
	if data['Texp_spectrum']['1665'] != []:
		vel_axes = [data['vel_axis']['1612'], data['vel_axis']['1665'], 
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
		vel_axes = [data['vel_axis']['1612'], data['vel_axis']['1665'], 
					data['vel_axis']['1667'], data['vel_axis']['1720']]
		spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720']]
		spectra_rms = [data['tau_rms']['1612'], data['tau_rms']['1665'], 
					data['tau_rms']['1667'], data['tau_rms']['1720']]

	for s in range(len(vel_axes)):
		vel_axis = vel_axes[s]
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
		return []
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
def sumspectra(data):
	vel_axes = [data['vel_axis']['1612'], data['vel_axis']['1665'], # 1612 should be shortest
				data['vel_axis']['1667'], data['vel_axis']['1720']]
	tau_spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'],
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720']]
	tau_spectra_rms = [ data['tau_rms']['1612'], data['tau_rms']['1665'], 
						data['tau_rms']['1667'], data['tau_rms']['1720']]
	num_spec = 4
	
	for a in range(4):
		if np.all(np.diff(vel_axes[a])):
			vel_axes[a], tau_spectra[a] = zip(*sorted(zip(vel_axes[a], tau_spectra[a])))

	regridded_tau_spectra = [np.interp(vel_axes[0], vel_axes[x], tau_spectra[x]) for x in range(4)]
	norm_tau_spectra = [regridded_tau_spectra[x] / tau_spectra_rms[x] for x in range(4)]
	sqrd_tau_spectra = [norm_tau_spectra[x]**2 for x in range(4)]
	summed = np.zeros(len(sqrd_tau_spectra[0]))

	for b in range(4):
		summed += sqrd_tau_spectra[b]

	if data['Texp_spectrum']['1665'] != []:
		num_spec = 8
		Texp_spectra = [	data['Texp_spectrum']['1612'], data['Texp_spectrum']['1665'],
							data['Texp_spectrum']['1667'], data['Texp_spectrum']['1720']]
		Texp_spectra_rms = [data['Texp_rms']['1612'], data['Texp_rms']['1665'], 
							data['Texp_rms']['1667'], data['Texp_rms']['1720']]
		for a in range(4):
			if np.all(np.diff(vel_axes[a])):
				vel_axes[a], Texp_spectra[a] = zip(*sorted(zip(vel_axes[a], Texp_spectra[a])))
		
		regridded_Texp_spectra = [np.interp(vel_axes[0], vel_axes[x], Texp_spectra[x]) for x in range(4)]
		norm_Texp_spectra = [regridded_Texp_spectra[x] / Texp_spectra_rms[x] for x in range(4)]	
		sqrd_Texp_spectra = [norm_Texp_spectra[x]**2 for x in range(4)]
		summed += sqrd_Texp_spectra[b]

	root_mean_sum = np.sqrt(summed/num_spec)

	return root_mean_sum
def reducelist(master_list = None, merge_size = 0.5, group_spacing = 0.5*FWHM_range[1]):# returns merged and grouped list
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
	
	# Step 2: identify comps likely to overlap to be fit together
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
def placegaussians(data = None, Bayes_threshold = 10, use_molex = True, molex_path = None, a = None, test = False, file_suffix = None): # returns final_p
	'''
	mpfit, emcee, decide whether or not to add another gaussian
	'''
	accepted_full = []
	total_num_gauss = 0
	plot_num = 0
	for vel_range in data['sig_vel_ranges']:
		last_accepted_full = []
		[min_vel, max_vel] = vel_range
		modified_data = trimdata(data, min_vel, max_vel)
		if modified_data['Texp_spectrum']['1665'] != []:
			N_ranges = NrangetauTexp(modified_data)
			modified_data['N_ranges'] = N_ranges
		num_gauss = 1
		keep_going = True
		extra = 0
		null_evidence = nullevidence(modified_data)
		prev_evidence = null_evidence
		evidences = [prev_evidence]
		print(data['source_name'] + '\t' + str(vel_range) + '\t' + str(null_evidence))

		while keep_going == True:
			# print('Currently attempting to fit '+ str(num_gauss) + ' Gaussian(s) in the range ' + str(vel_range) + ' km/s')
			nwalkers = 30 * num_gauss
			p0 = p0gen(	vel_range = vel_range, 
						num_gauss = num_gauss, 
						modified_data = modified_data, 
						accepted_params = accepted_full,
						last_accepted_params = last_accepted_full, 
						nwalkers = nwalkers, 
						use_molex = use_molex)
			(chain, lnprob_) = sampleposterior(	modified_data = modified_data, 
												num_gauss = num_gauss, 
												p0 = p0, 
												vel_range = [min_vel, max_vel],
												accepted = accepted_full, 
												nwalkers = nwalkers, 
												use_molex = use_molex,
												molex_path = molex_path,  
												a = a, file_suffix = file_suffix)

			if len(chain) != 0:
				(current_full, current_evidence) = bestparams(chain, lnprob_)
				plot_num += 1
				evidences += [current_evidence]
				if current_evidence - prev_evidence > Bayes_threshold:
					extra = 0
					last_accepted_full = current_full
					prev_evidence = current_evidence
					num_gauss += 1
					total_num_gauss += 1
				elif extra > 3 or current_evidence - prev_evidence < -1e4:
					keep_going = False
				else:
					extra += 1 # This needs to be tested
					num_gauss += 1
			else:
				# print('Process failed because no finite likelihood found. Will try one more time.')
				nwalkers = 30 * num_gauss
				p0 = p0gen(	vel_range = vel_range, 
							num_gauss = num_gauss, 
							modified_data = modified_data, 
							accepted_params = accepted_full,
							last_accepted_params = last_accepted_full, 
							nwalkers = nwalkers, 
							use_molex = use_molex)
				(chain, lnprob_) = sampleposterior(	modified_data = modified_data, 
													num_gauss = num_gauss, 
													p0 = p0, 
													vel_range = [min_vel, max_vel],  
													accepted = accepted_full, 
													nwalkers = nwalkers, 
													use_molex = use_molex, 
													molex_path = molex_path, 
													a = a, file_suffix = file_suffix)
				if len(chain) == 0:
					# print('Process failed again. Moving on.')
					keep_going = False
		accepted_full = list(itertools.chain(accepted_full, last_accepted_full))
	if test:
		return [accepted_full, evidences]
	else:
		return accepted_full
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
	model_1612 = np.zeros(len(modified_data['tau_spectrum']['1612']))
	model_1665 = np.zeros(len(modified_data['tau_spectrum']['1665']))
	model_1667 = np.zeros(len(modified_data['tau_spectrum']['1667']))
	model_1720 = np.zeros(len(modified_data['tau_spectrum']['1720']))

	lnllh_tau_1612 = lnlikelihood(model = model_1612, 
		spectrum = modified_data['tau_spectrum']['1612'], 
		sigma = modified_data['tau_rms']['1612'])
	lnllh_tau_1665 = lnlikelihood(model = model_1665, 
		spectrum = modified_data['tau_spectrum']['1665'], 
		sigma = modified_data['tau_rms']['1665'])
	lnllh_tau_1667 = lnlikelihood(model = model_1667, 
		spectrum = modified_data['tau_spectrum']['1667'], 
		sigma = modified_data['tau_rms']['1667'])
	lnllh_tau_1720 = lnlikelihood(model = model_1720, 
		spectrum = modified_data['tau_spectrum']['1720'], 
		sigma = modified_data['tau_rms']['1720'])

	lnllh = np.sum([lnllh_tau_1612, lnllh_tau_1665, 
		lnllh_tau_1667, lnllh_tau_1720])

	if modified_data['Texp_spectrum']['1665'] != []:
		lnllh_Texp_1612 = lnlikelihood(model = model_1612, 
			spectrum = modified_data['Texp_spectrum']['1612'], 
			sigma = modified_data['Texp_rms']['1612'])
		lnllh_Texp_1665 = lnlikelihood(model = model_1665, 
			spectrum = modified_data['Texp_spectrum']['1665'], 
			sigma = modified_data['Texp_rms']['1665'])
		lnllh_Texp_1667 = lnlikelihood(model = model_1667, 
			spectrum = modified_data['Texp_spectrum']['1667'], 
			sigma = modified_data['Texp_rms']['1667'])
		lnllh_Texp_1720 = lnlikelihood(model = model_1720, 
			spectrum = modified_data['Texp_spectrum']['1720'], 
			sigma = modified_data['Texp_rms']['1720'])

		lnllh += np.sum([lnllh_Texp_1612, lnllh_Texp_1665, 
			lnllh_Texp_1667, lnllh_Texp_1720])
	return lnllh	
def lnlikelihood(model = None, spectrum = None, sigma = None): # returns lnlikelihood
	N = len(spectrum)
	sse = np.sum((np.array(model) - np.array(spectrum))**2.)
	# print('N:' + str(N) + '\tsigma: ' + str(sigma) + '\tsse: ' + str(sse) + '\tlikelihood: ' + str(-N * np.log(sigma * np.sqrt(2. * np.pi)) - (sse / (2. * (sigma**2.)))))
	return -N * np.log(sigma * np.sqrt(2. * np.pi)) - (sse / (2. * (sigma**2.)))
def bestparams(chain = None, lnprob = None): # returns ([-sig, med, +sig] for all variables in chain, evidence)
	'''
	Tested and verified 22/3/19
	'''
	# separate out walkers with very different positions
	(grouped_chains, grouped_lnprob) = splitwalkers(chain, lnprob)

	(final_results, final_evidence) = ([], -np.inf)

	for group in range(len(grouped_chains)):
		chain_subset = grouped_chains[group]
		lnprob_subset = grouped_lnprob[group]

		num_steps = len(chain_subset)
		num_param = len(chain_subset[0])

		final_array = [list(reversed(sorted(lnprob_subset)))]
		final_darray = [list(reversed(sorted(lnprob_subset)))]

		for param in range(num_param):
			param_chain = [chain_subset[x][param] for x in range(num_steps)]
			final_array = np.concatenate((final_array, [[x for _,x in list(reversed(sorted(zip(lnprob_subset, param_chain))))]])) # this makes an array with [[lnprob_subset], [param1 iterations], etc] all sorted in descending order of lnprob_subset (seems ok)
			zipped = sorted(zip(param_chain, lnprob_subset))
			sorted_param_chain, sorted_lnprob = zip(*zipped) # then sort all by this one parameter

			dparam_chain = [0] + [sorted_param_chain[x] - sorted_param_chain[x-1] for x in range(1, len(sorted_param_chain))] # calculate dparam
			sorted_dparam_chain = [[x for _,x in list(reversed(sorted(zip(sorted_lnprob, dparam_chain))))]] # put back into correct order
			final_darray = np.concatenate((final_darray, sorted_dparam_chain), axis = 0) # this makes an array with [[lnprob_subset], [dparam1 iterations], etc] all sorted in descending order of lnprob_subset (seems ok)

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
				accumulated_evidence[step] = np.logaddexp(
					accumulated_evidence[step - 1], contribution_to_lnevidence)

		total_evidence = accumulated_evidence[-1]
		
		if total_evidence > final_evidence:
			evidence_68 = total_evidence + np.log(0.6825)
			sigma_index = np.argmin(abs(accumulated_evidence - evidence_68))
			results = np.zeros([num_param, 3])

			for param in range(num_param):
				results[param][0]=np.amin(final_array[param+1][:sigma_index+1])
				results[param][1]=np.median(final_array[param+1])
				results[param][2]=np.amax(final_array[param+1][:sigma_index+1])
			final_results = results
			final_evidence = total_evidence

	print('Evidence and Preliminary results:'+'\t'+str(final_evidence)+
		'\t'+str(final_results))
	return (final_results, final_evidence)
# initial fit using mpfit
def p0gen(vel_range = None, num_gauss = None, modified_data = None, 
	accepted_params = [], last_accepted_params = [], nwalkers = None, 
	use_molex = True):
	# generate p0
	if use_molex:
		p0_2 = 0
		for walker in range(nwalkers):
			p0_1 = 0
			for comp in range(num_gauss):
				p0_0 = [
					np.random.uniform(np.min(vel_range), np.max(vel_range)), 
					np.random.uniform(np.min(logTgas_range), np.max(logTgas_range)), 
					np.random.uniform(np.min(lognH2_range), np.max(lognH2_range)), 
					np.random.uniform(np.min(logNOH_range), np.max(logNOH_range)), 
					np.random.uniform(np.min(fortho_range), np.max(fortho_range)), 
					np.random.uniform(np.min(FWHM_range), np.max(FWHM_range)), 
					np.random.uniform(np.min(Av_range), np.max(Av_range)), 
					np.random.uniform(np.min(logxOH_range), np.max(logxOH_range)), 
					np.random.uniform(np.min(logxHe_range), np.max(logxHe_range)), 
					np.random.uniform(np.min(logxe_range), np.max(logxe_range)), 
					np.random.uniform(np.min(logTdint_range), np.max(logTdint_range)), 
					np.random.uniform(np.min(logTd_range), np.max(logTd_range))]
				parameter_list = [True] + modified_data['parameter_list']
				len_param_list = len(parameter_list)
				if p0_1 == 0:
					p0_1 = [p0_0[a] for a in range(len(p0_0)) if 
						type(parameter_list[a%len_param_list]) == bool and 
						parameter_list[a%len_param_list] == True]
				else:
					p0_1 += [p0_0[a] for a in range(len(p0_0)) if 
						type(parameter_list[a%len_param_list]) == bool and 
						parameter_list[a%len_param_list] == True]
			
			if p0_2 == 0:
				p0_2 = [p0_1]
			else:
				p0_2 += [p0_1]

		return p0_2
	elif modified_data['Texp_spectrum']['1665'] != []:
		p0_2 = 0
		for walker in range(nwalkers):
			p0_1 = 0
			for comp in range(num_gauss):
				p0_0 = [np.random.uniform(np.min(vel_range), np.max(vel_range)), 
					np.random.uniform(np.min(FWHM_range), np.max(FWHM_range)), 
					np.random.uniform(np.min(modified_data['N_ranges'][0]), 
						np.max(modified_data['N_ranges'][0])), 
					np.random.uniform(np.min(modified_data['N_ranges'][1]), 
						np.max(modified_data['N_ranges'][1])), 
					np.random.uniform(np.min(modified_data['N_ranges'][2]), 
						np.max(modified_data['N_ranges'][2])), 
					np.random.uniform(np.min(modified_data['N_ranges'][3]), 
						np.max(modified_data['N_ranges'][3]))] * num_gauss
				if p0_1 == 0:
					p0_1 = p0_0
				else:
					p0_1 += p0_0

			if p0_2 == 0:
				p0_2 = [p0_1]
			else:
				p0_2 += [p0_1]

		return p0_2
	else:
		# only these are subject to the tau priors, so should be ok
		(tau_1612_range, tau_1665_range, tau_1667_range, tau_1720_range) = (
			[-2 * np.abs(np.amin(modified_data['tau_spectrum']['1612'])), 
				2 * np.abs(np.amax(modified_data['tau_spectrum']['1612']))], 
			[-1.5 * np.abs(np.amin(modified_data['tau_spectrum']['1665'])), 
				1.5 * np.abs(np.amax(modified_data['tau_spectrum']['1665']))], 
			[-1.5 * np.abs(np.amin(modified_data['tau_spectrum']['1667'])), 
				1.5 * np.abs(np.amax(modified_data['tau_spectrum']['1667']))], 
			[-2 * np.abs(np.amin(modified_data['tau_spectrum']['1720'])), 
				2 * np.abs(np.amax(modified_data['tau_spectrum']['1720']))]) 

		# define axes for meshgrid
		t1612 = np.arange(tau_1612_range[0], tau_1612_range[1], 
			(tau_1612_range[1] - tau_1612_range[0]+1e-10)/(100*num_gauss**(1./3.)))
		t1667 = np.arange(tau_1667_range[0], tau_1667_range[1], 
			(tau_1667_range[1] - tau_1667_range[0]+1e-10)/(100*num_gauss**(1./3.)))
		t1720 = np.arange(tau_1720_range[0], tau_1720_range[1], 
			(tau_1720_range[1] - tau_1720_range[0]+1e-10)/(100*num_gauss**(1./3.)))

		tt1612, tt1667, tt1720 = np.meshgrid(t1612, t1667, t1720, 
			indexing = 'ij')
		t1665 = 5.*tt1612 + 5.*tt1720 - (5.*tt1667/9.)

		good_values = np.argwhere((t1665 > tau_1665_range[0]) & 
			(t1665 < tau_1665_range[1]))

		if len(good_values) >= num_gauss * nwalkers:
			p0_indices = good_values[np.random.choice(
				np.arange(len(good_values)), nwalkers * num_gauss, replace = False)]
		else:
			p0_indices = good_values[np.random.choice(
				np.arange(len(good_values)), nwalkers * num_gauss, replace = True)]

		vel_guesses = [sorted([np.random.uniform(vel_range[0], vel_range[1]) for x in range(num_gauss)]) for y in range(nwalkers)]
		for comp in range(num_gauss):
			p0_comp = [[vel_guesses[x][comp], 
				np.random.uniform(FWHM_range[0], FWHM_range[1]), 
				t1612[p0_indices[comp*nwalkers + x][0]],t1667[p0_indices[comp*nwalkers + x][1]],t1720[p0_indices[comp*nwalkers + x][2]]] 
				for x in range(nwalkers)]
			if comp == 0:
				p0 = p0_comp
			else:
				p0 = np.concatenate((p0, p0_comp), axis = 1)
		return p0
	'''
	Might still not work, need to check. Errors are thrown below even though 
	it seems good.
	'''
	# success = 0
	# failure = 0
	# p0 = []

	# while success < nwalkers and failure < 1e2:
	# 	walker = [a * np.random.uniform(0.999, 1.001) for a in p0_0]

	# 	if use_molex:
	# 		p = plist(walker, modified_data, num_gauss)
	# 		prior = lnprprior(modified_data = modified_data, p = p, 
	# 			vel_range = vel_range, num_gauss = num_gauss)
	# 	else:
	# 		prior = lnprprior(modified_data = modified_data, params = walker, 
	# 			vel_range = vel_range, num_gauss = num_gauss, 
	# 			use_molex = use_molex)

	# 	if np.isnan(prior) == False and np.isinf(prior) == False:
	# 		if p0 == []:
	# 			p0 = [walker]
	# 		else:
	# 			p0 = np.concatenate((p0, [walker]), axis = 0)
	# 		success += 1
	# 	else:
	# 		failure += 1

	# if len(p0) == nwalkers:
	# 	return p0
	# elif len(p0) > 0:
	# 	p0 = (1. + 1e-8 * np.random.randn()) * np.array([p0[x] for x in 
	# 		np.random.choice(np.arange(len(p0)), nwalkers, replace = True)])
	# 	return p0
	# else:
	# 	print('Failed to find initial positions for walkers! Check allowed ' 
			# + 'ranges for parameters, they may be too restrictive.')
# converts between x <--> p --> params
def molex(p = [], x = None, modified_data = None, num_gauss = 1, return_Tex = False, molex_path = None, file_suffix = None): # returns params
	'''
	If return_Tex == True, Tex will be returned instead of N. Use with caution.
	'''
	if p == []:
		p = plist(x, modified_data, num_gauss)

	if return_Tex:
		output = list(np.zeros(int(10 * num_gauss))) 
			# returns v, fwhm, tau x 4, Tex x 4
	else:
		output = list(np.zeros(int(6 * num_gauss)))
			# returns v, fwhm, N x 4

	for gaussian in range(int(num_gauss)):
		[vel, logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe,
			logTdint, logTd] = p[int(gaussian * 12):int((gaussian + 1) * 12)]

		# check if temp.txt exists - erase
		# if file_suffix == None:
		# 	file_suffix = str(datetime.datetime.now())
		try: 
			subprocess.run('rm temp.txt', shell = True, 
				stderr=subprocess.DEVNULL)
		except:
			pass

		# Write ohslab.in file
		with open('oh_slab.in', 'w') as oh_slab:
			# Based on template! Don't touch!
			oh_slab.write('\'temp.txt\'\nF\nF\n' + str(10.**logTgas) + '\n' + 
				str(FWHM) + '\n' + str(fortho) + '\n')
			oh_slab.write(str(10.**logxOH) + '\n' + str(10.**logxHe) + '\n' + 
				str(10.**logxe) + '\nF\n' + str(10.**logTdint) + '\n' + 
				str(10.**logTd) + '\n')
			oh_slab.write(str(Av) + '\nF\n2\n' + str(lognH2) + '\n' + 
				str(lognH2) + '\n0.05\n' + str(logNOH) + '\n')
			oh_slab.write(str(logNOH) + '\n0.1\n3,2\n3,1\n4,2\n4,1\n -1, ' + 
				'-1\nF\nT\n1.0\n4\n0.1\n32\n1\n1.d-6\n1.d-16')
			oh_slab.write('\nT\n20\n1.d-6\nF\n1\n17\nF\nF\nF\nF\nF\nF\nF\nF' +
				'\nF\nF\nF\nF\nF\nT\nF\nF\nF\n')

		subprocess.call('make ohslab', shell = True, stdout=subprocess.DEVNULL)
		subprocess.run([molex_path], shell = True, stdout=subprocess.DEVNULL)
		try:
			with open('temp.txt', 'r') as f:
				for line in islice(f, 33, 34):
					Tex_1612, tau0_1612, Tex_1665, tau0_1665, Tex_1667, tau0_1667, Tex_1720, tau0_1720 = line.split()[3:11]
			[Tex_1612, tau0_1612, Tex_1665, tau0_1665, Tex_1667, tau0_1667, Tex_1720, tau0_1720] = [float(Tex_1612), float(tau0_1612), float(Tex_1665), float(tau0_1665), float(Tex_1667), float(tau0_1667), float(Tex_1720), float(tau0_1720)]
			if return_Tex:
				output[int(10 * gaussian):int(10 * (gaussian + 1))] = [vel, FWHM, tau0_1612, tau0_1665, tau0_1667, tau0_1720, Tex_1612, Tex_1665, Tex_1667, Tex_1720]
			else:
				[N1, N2, N3, N4] = NtauTex(tau_1612 = tau0_1612, tau_1665 = tau0_1665, tau_1667 = tau0_1667, tau_1720 = tau0_1720, Tex_1612 = Tex_1612, Tex_1665 = Tex_1665, Tex_1667 = Tex_1667, Tex_1720 = Tex_1720, fwhm = FWHM)
				output[int(6 * gaussian):int(6 * (gaussian + 1))] = [vel, FWHM, N1, N2, N3, N4]		



		except:
			[Tex_1612, tau0_1612, Tex_1665, tau0_1665, Tex_1667, tau0_1667, Tex_1720, tau0_1720] = np.zeros(8)
		
			if return_Tex:
				output[int(10 * gaussian):int(10 * (gaussian + 1))] = [vel, FWHM, 0,0,0,0,0,0,0,0]
			else:
				output[int(6 * gaussian):int(6 * (gaussian + 1))] = [vel, FWHM, 0,0,0,0]
		# erase temp.txt
		subprocess.run('rm temp.txt', shell = True, stderr = subprocess.DEVNULL)

	return output
def plist(x = None, data = None, num_gauss = None): # returns p
	p = ([True] + list(data['parameter_list'])) * int(num_gauss)
	x_counter = 0
	for a in range(len(p)):
		if type(p[a]) == bool and p[a] == True:
			p[a] = x[x_counter]
			x_counter += 1
	return p
def xlist(p = None, data = None, num_gauss = None): # returns x
	parameter_list = ([True] + list(data['parameter_list'])) * int(num_gauss)
	x = [p[a] for a in range(len(p)) if type(parameter_list[a]) == bool and parameter_list[a] == True]
	return x
# converts between tau, Tex, Texp, N
def Texp(tau = None, Tbg = None, Tex = None): # returns Texp
	Texp = (Tex - Tbg) * (1 - np.exp(-tau))
	return Texp
def tau3(tau_1612 = None, tau_1665 = None, tau_1667 = None, tau_1720 = None): # returns all 4 taus when 3 are input
	tau_list = np.array([tau_1612, tau_1665, tau_1667, tau_1720])
	if (tau_list == None).sum() == 1: # only one is left blank, can proceed
		if tau_1612 == None:
			tau_1612 = tau_1665/5 + tau_1667/9 - tau_1720
		elif tau_1665 == None:
			tau_1665 = 5 * (tau_1612 + tau_1720 - tau_1667/9)
		elif tau_1667 == None:
			tau_1667 = 9 * (tau_1612 + tau_1720 - tau_1665/5)
		elif tau_1720 == None:
			tau_1720 = tau_1665/5 + tau_1667/9 - tau_1612

		return np.array([tau_1612, tau_1665, tau_1667, tau_1720])
	elif (tau_list == None).sum() == 0: # none left blank, see if they follow the sum rule
		print('4 values of tau provided, confirming adherence to sum rule.')
		sum_res = np.abs(tau_1665/5 + tau_1667/9 - tau_1612 - tau_1720)
		print('Residual of sum rule = ' + str(sum_res))
		print('Percentages of supplied tau values: ' + str(np.abs(sum_res/tau_list)))
	else: # can't do anything
		print('Error, at least 3 values of tau needed to apply the sum rule.')
		return None
def tauTexN(logN1 = None, logN2 = None, logN3 = None, logN4 = None, fwhm = None): # returns peak tau and Tex values given 4 log column densities
	# mp.dps = 50
	# relevant constants (cgs):
	con = {
	'rest_freq': {'1612': 1612231000., '1665': 1665402000., 
		'1667': 1667359000., '1720': 1720530000.},
	'gu': {'1612': 3., '1665': 3., '1667': 5., '1720': 5.},
	'gl': {'1612': 5., '1665': 3., '1667': 5., '1720': 3.},
	'Aul': {'1612': 1.302E-11, '1665': 7.177E-11, '1667': 7.778E-11, 
		'1720': 9.496E-12},
	'logNu': {'1612': logN2, '1665': logN2, '1667': logN1, '1720': logN1},
	'logNl': {'1612': logN3, '1665': logN4, '1667': logN3, '1720': logN4}}
	h = 6.62607004E-27
	k = 1.38064852E-16
	c = 2.99792458E+10
	# calculating ln(Nlgu/Nugl) separately
	lnNg_1612 = np.log(10.)*(con['logNl']['1612']-con['logNu']['1612'])+np.log(con['gu']['1612'])-np.log(con['gl']['1612'])
	lnNg_1665 = np.log(10.)*(con['logNl']['1665']-con['logNu']['1665'])
	lnNg_1667 = np.log(10.)*(con['logNl']['1667']-con['logNu']['1667'])
	lnNg_1720 = np.log(10.)*(con['logNl']['1720']-con['logNu']['1720'])+np.log(con['gu']['1720'])-np.log(con['gl']['1720'])

	# compute Tex
	Tex_1612 = (h*con['rest_freq']['1612'])/(k*lnNg_1612)
	Tex_1665 = (h*con['rest_freq']['1665'])/(k*lnNg_1665)
	Tex_1667 = (h*con['rest_freq']['1667'])/(k*lnNg_1667)
	Tex_1720 = (h*con['rest_freq']['1720'])/(k*lnNg_1720)


	# coefficients
	coeff_1612 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1612']*
		con['rest_freq']['1612']**2)/(con['gu']['1612']*con['Aul']['1612']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))
	coeff_1665 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1665']*
		con['rest_freq']['1665']**2)/(con['gu']['1665']*con['Aul']['1665']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))
	coeff_1667 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1667']*
		con['rest_freq']['1667']**2)/(con['gu']['1667']*con['Aul']['1667']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))
	coeff_1720 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1720']*
		con['rest_freq']['1720']**2)/(con['gu']['1720']*con['Aul']['1720']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))

	# compute peak tau
	tau_1612 = (10.**con['logNl']['1612'])/(coeff_1612*Tex_1612*fwhm*(10**5))
	tau_1665 = (10.**con['logNl']['1665'])/(coeff_1665*Tex_1665*fwhm*(10**5))
	tau_1667 = (10.**con['logNl']['1667'])/(coeff_1667*Tex_1667*fwhm*(10**5))
	tau_1720 = (10.**con['logNl']['1720'])/(coeff_1720*Tex_1720*fwhm*(10**5))


	return [tau_1612, tau_1665, tau_1667, tau_1720, 
			Tex_1612, Tex_1665, Tex_1667, Tex_1720]
def NtauTex(tau_1612 = None, tau_1665 = None, tau_1667 = None, tau_1720 = None, Tex_1612 = None, Tex_1665 = None, Tex_1667 = None, Tex_1720 = None, fwhm = None): # returns 4 log column densities (assumes consistent input)
	# relevant constants (cgs):
	# con = {
	# 'rest_freq': {'1612': 1612231000., '1665': 1665402000., 
	# 	'1667': 1667359000., '1720': 1720530000.},
	# 'gu': {'1612': 3., '1665': 3., '1667': 5., '1720': 5.},
	# 'gl': {'1612': 5., '1665': 3., '1667': 5., '1720': 3.},
	# 'Aul': {'1612': 1.302E-11, '1665': 7.177E-11, '1667': 7.778E-11, 
	# 	'1720': 9.496E-12}}
	# h = 6.62607004E-27
	# k = 1.38064852E-16
	# c = 2.99792458E+10
	
	# N3 = ((con['gl']['1667']*con['rest_freq']['1667']**2)/
	# 	(con['gu']['1667']*con['Aul']['1667']))*(
	# 	(8.*np.pi*k*np.sqrt(np.pi)*Tex_1667*tau_1667*fwhm*(10**5))/
	# 	((c**3)*h*2.*np.sqrt(np.log(2.))))
	# N1 = N3*((con['gu']['1667']/con['gl']['1667'])*
	# 	np.exp(-h*con['rest_freq']['1667']/(k*Tex_1667)))
	# N2 = N3*((con['gu']['1612']/con['gl']['1612'])*
	# 	np.exp(-h*con['rest_freq']['1612']/(k*Tex_1612)))
	# N4 = N1/((con['gu']['1720']/con['gl']['1720'])*
	# 	np.exp(-h*con['rest_freq']['1720']/(k*Tex_1720)))

	N3 = Tex_1667*tau_1667*fwhm*7.39481592616533E+13
	N1 = N3*(np.exp(-0.0800206378074005/Tex_1667))
	N2 = N3*((3/5)*np.exp(-0.0773749102100167/Tex_1612))
	N4 = N1/((5/3)*np.exp(-0.0825724441867449/Tex_1720))

	return [np.log10(N1), np.log10(N2), np.log10(N3), np.log10(N4)]
def NrangetauTexp(modified_data = None): # returns 4 log column density ranges based on tau and Texp ranges
	# relevant constants (cgs):
	con = {
	'rest_freq': {'1612': 1612231000., '1665': 1665402000., 
		'1667': 1667359000., '1720': 1720530000.},
	'gu': {'1612': 3., '1665': 3., '1667': 5., '1720': 5.},
	'gl': {'1612': 5., '1665': 3., '1667': 5., '1720': 3.},
	'Aul': {'1612': 1.302E-11, '1665': 7.177E-11, '1667': 7.778E-11, 
		'1720': 9.496E-12}}
	h = 6.62607004E-27
	k = 1.38064852E-16
	c = 2.99792458E+10

	coeff_1612 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1612']*
		con['rest_freq']['1612']**2)/(con['gu']['1612']*con['Aul']['1612']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))
	coeff_1665 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1665']*
		con['rest_freq']['1665']**2)/(con['gu']['1665']*con['Aul']['1665']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))
	coeff_1667 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1667']*
		con['rest_freq']['1667']**2)/(con['gu']['1667']*con['Aul']['1667']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))
	coeff_1720 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1720']*
		con['rest_freq']['1720']**2)/(con['gu']['1720']*con['Aul']['1720']*
		(c**3)*h*2.*np.sqrt(np.log(2.))))

	Tex_1612 = modified_data['Tbg']['1612']+(np.array(modified_data[
		'Texp_spectrum']['1612'])/(1.-np.exp(-np.array(modified_data[
		'tau_spectrum']['1612']))))
	Tex_1665 = modified_data['Tbg']['1665']+(np.array(modified_data[
		'Texp_spectrum']['1665'])/(1.-np.exp(-np.array(modified_data[
		'tau_spectrum']['1665']))))
	Tex_1667 = modified_data['Tbg']['1667']+(np.array(modified_data[
		'Texp_spectrum']['1667'])/(1.-np.exp(-np.array(modified_data[
		'tau_spectrum']['1667']))))
	Tex_1720 = modified_data['Tbg']['1720']+(np.array(modified_data[
		'Texp_spectrum']['1720'])/(1.-np.exp(-np.array(modified_data[
		'tau_spectrum']['1720']))))
	
	logN3_1 = np.abs(np.log10(np.abs(coeff_1667*(10**5)*FWHM_range[1]*np.array(
		modified_data['tau_spectrum']['1667'])*(Tex_1667))))
	logN3_2 = np.abs(np.log10(np.abs(coeff_1612*(10**5)*FWHM_range[1]*np.array(
		modified_data['tau_spectrum']['1612'])*(Tex_1612))))
	logN3_3 = np.abs(np.log10(np.abs(coeff_1667*(10**5)*FWHM_range[0]*np.array(
		modified_data['tau_spectrum']['1667'])*(Tex_1667))))
	logN3_4 = np.abs(np.log10(np.abs(coeff_1612*(10**5)*FWHM_range[0]*np.array(
		modified_data['tau_spectrum']['1612'])*(Tex_1612))))
	logN4_1 = np.abs(np.log10(np.abs(coeff_1665*(10**5)*FWHM_range[1]*np.array(
		modified_data['tau_spectrum']['1665'])*(Tex_1665))))
	logN4_2 = np.abs(np.log10(np.abs(coeff_1720*(10**5)*FWHM_range[1]*np.array(
		modified_data['tau_spectrum']['1720'])*(Tex_1720))))
	logN4_3 = np.abs(np.log10(np.abs(coeff_1665*(10**5)*FWHM_range[0]*np.array(
		modified_data['tau_spectrum']['1665'])*(Tex_1665))))
	logN4_4 = np.abs(np.log10(np.abs(coeff_1720*(10**5)*FWHM_range[0]*np.array(
		modified_data['tau_spectrum']['1720'])*(Tex_1720))))

	logN3_min = np.nanmin([np.nanmin(logN3_1), np.nanmin(logN3_2), 
		np.nanmin(logN3_3), np.nanmin(logN3_4)])
	logN3_max = np.nanmax([np.nanmax(logN3_1), np.nanmax(logN3_2), 
		np.nanmax(logN3_3), np.nanmax(logN3_4)])

	logN4_min = np.nanmin([np.nanmin(logN4_1), np.nanmin(logN4_2), 
		np.nanmin(logN4_3), np.nanmin(logN4_4)])
	logN4_max = np.nanmax([np.nanmax(logN4_1), np.nanmax(logN4_2), 
		np.nanmax(logN4_3), np.nanmax(logN4_4)])


	logN1_1 = np.log10((10.**logN3_min)*(5./5.)*np.exp(-h*
			con['rest_freq']['1667']/(k*Tex_1667)))
	logN1_2 = np.log10((10.**logN4_min)*(5./3.)*np.exp(-h*
			con['rest_freq']['1720']/(k*Tex_1720)))
	logN1_3 = np.log10((10.**logN3_max)*(5./5.)*np.exp(-h*
			con['rest_freq']['1667']/(k*Tex_1667)))
	logN1_4 = np.log10((10.**logN4_max)*(5./3.)*np.exp(-h*
			con['rest_freq']['1720']/(k*Tex_1720)))
	logN2_1 = np.log10((10.**logN4_min)*(3./3.)*np.exp(-h*
			con['rest_freq']['1665']/(k*Tex_1665)))
	logN2_2 = np.log10((10.**logN3_min)*(3./5.)*np.exp(-h*
			con['rest_freq']['1612']/(k*Tex_1612)))
	logN2_3 = np.log10((10.**logN4_max)*(3./3.)*np.exp(-h*
			con['rest_freq']['1665']/(k*Tex_1665)))
	logN2_4 = np.log10((10.**logN3_max)*(3./5.)*np.exp(-h*
			con['rest_freq']['1612']/(k*Tex_1612)))

	logN1_min = np.nanmin([np.nanmin(logN1_1), np.nanmin(logN1_2), np.nanmin(logN1_3), np.nanmin(logN1_4)])
	logN1_max = np.nanmax([np.nanmax(logN1_1), np.nanmax(logN1_2), np.nanmax(logN1_3), np.nanmax(logN1_4)])

	logN2_min = np.nanmin([np.nanmin(logN2_1), np.nanmin(logN2_2), np.nanmin(logN2_3), np.nanmin(logN2_4)])
	logN2_max = np.nanmax([np.nanmax(logN2_1), np.nanmax(logN2_2), np.nanmax(logN2_3), np.nanmax(logN2_4)])	

	return [[logN1_min, logN1_max], [logN2_min, logN2_max], [logN3_min, logN3_max], [logN4_min, logN4_max]]
# makes/plots model from params
def makemodel(params = None, modified_data = None, accepted_params = [], num_gauss = None, use_molex = True): # returns tau and Texp models
	vel_1612 = modified_data['vel_axis']['1612']
	vel_1665 = modified_data['vel_axis']['1665']
	vel_1667 = modified_data['vel_axis']['1667']
	vel_1720 = modified_data['vel_axis']['1720']

	num_params = int(len(params) / num_gauss)
	# initialise models
	if accepted_params != []:
		if modified_data['Texp_spectrum']['1665'] != []:
			(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = makemodel(params = accepted_params, modified_data = modified_data, num_gauss = int(len(accepted_params) / 6), use_molex = use_molex)
		else:
			(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720) = makemodel(params = accepted_params, modified_data = modified_data, num_gauss = int(len(accepted_params) / 5), use_molex = use_molex)
	else:
		tau_m_1612 = np.zeros(len(vel_1612))
		tau_m_1665 = np.zeros(len(vel_1665))
		tau_m_1667 = np.zeros(len(vel_1667))
		tau_m_1720 = np.zeros(len(vel_1720))

		if modified_data['Texp_spectrum']['1665'] != []:
			Texp_m_1612 = np.zeros(len(vel_1612))
			Texp_m_1665 = np.zeros(len(vel_1665))
			Texp_m_1667 = np.zeros(len(vel_1667))
			Texp_m_1720 = np.zeros(len(vel_1720))
	
	# make models
	for comp in range(int(num_gauss)): 
		if modified_data['Texp_spectrum']['1665'] != []:
			[vel, FWHM, logN1, logN2, logN3, logN4] = params[comp * 
				num_params:(comp + 1) * num_params]
			[tau_1612, tau_1665, tau_1667, tau_1720, 
				Tex_1612, Tex_1665, Tex_1667, Tex_1720] = tauTexN(
					logN1 = logN1, logN2 = logN2, logN3 = logN3, 
					logN4 = logN4, fwhm = FWHM)
			[Texp_1612, Texp_1665, Texp_1667, Texp_1720] = [
				Texp(tau_1612, modified_data['Tbg']['1612'], Tex_1612), 
				Texp(tau_1665, modified_data['Tbg']['1665'], Tex_1665), 
				Texp(tau_1667, modified_data['Tbg']['1667'], Tex_1667), 
				Texp(tau_1720, modified_data['Tbg']['1720'], Tex_1720)]
		else:
			[vel, FWHM, tau_1612, tau_1667, tau_1720] = params[comp * 
				num_params:(comp + 1) * num_params]
			[tau_1612, tau_1665, tau_1667, tau_1720] = tau3(
				tau_1612 = tau_1612, tau_1667 = tau_1667, tau_1720 = tau_1720)
		tau_m_1612 += gaussian(vel, FWHM, tau_1612)(np.array(vel_1612))
		tau_m_1665 += gaussian(vel, FWHM, tau_1665)(np.array(vel_1665))
		tau_m_1667 += gaussian(vel, FWHM, tau_1667)(np.array(vel_1667))
		tau_m_1720 += gaussian(vel, FWHM, tau_1720)(np.array(vel_1720))
		if modified_data['Texp_spectrum']['1665'] != []:
			Texp_m_1612 += gaussian(vel, FWHM, Texp_1612)(np.array(vel_1612))
			Texp_m_1665 += gaussian(vel, FWHM, Texp_1665)(np.array(vel_1665))
			Texp_m_1667 += gaussian(vel, FWHM, Texp_1667)(np.array(vel_1667))
			Texp_m_1720 += gaussian(vel, FWHM, Texp_1720)(np.array(vel_1720))
	# return models
	if modified_data['Texp_spectrum']['1665'] != []:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, 
			Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720)	
	else:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720)
def gaussian(mean = None, FWHM = None, height = None, sigma = None, amp = None): # returns lambda gaussian
	'''
	Generates a gaussian profile with the given parameters.
	'''
	if sigma == None:
		sigma = FWHM / (2. * np.sqrt(2. * np.log(2.)))

	if height == None:
		height = amp / (sigma * np.sqrt(2.* np.pi))
	return lambda x: height * np.exp(-((x - mean)**2.) / (2.*sigma**2.))
# sample posterior using emcee
def sampleposterior(modified_data = None,  # returns (chain, lnprob)
	num_gauss = None, p0 = None, vel_range = None, accepted = [], 
	nwalkers = None, use_molex = True, molex_path = None, a = None, file_suffix = None): 
	if use_molex:
		ndim = num_gauss # starts with one for all velocities
		for a in modified_data['parameter_list']:
			if type(a) == bool and a == True:
				ndim += num_gauss # adds another parameter for each velocity
			
	elif modified_data['Texp_spectrum']['1665'] != []:
		ndim = 6 * num_gauss
	else:
		ndim = 5 * num_gauss
	
	num_param_per_gauss = ndim / num_gauss

	if use_molex and accepted != []:
		accepted_x = [x[1] for x in accepted]
		accepted_params = molex(x = accepted_x, modified_data = modified_data, 
			num_gauss = len(accepted_x) / num_param_per_gauss, molex_path = molex_path, file_suffix = file_suffix)
	elif accepted != []:
		accepted_params = []
	else:
		accepted_params = [x[1] for x in accepted]

	burn_iterations = 30
	final_iterations = 30
	
	args = [modified_data,[], vel_range, num_gauss, accepted_params, use_molex, molex_path, file_suffix]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args, a = a)
	# burn
	[burn_run, test_result] = [0, 'Fail']
	while burn_run <= 3 and test_result == 'Fail':
		try:
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		except ValueError: # sometimes there is an error within emcee
			print('emcee is throwing a value error for p0. Running again.')
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		# test convergence
		(test_result, p0) = convergencetest(sampler_chain = sampler.chain, 
			num_gauss = num_gauss, pos = pos)
		burn_run += 1
	# plotchain(modified_data = modified_data, chain = sampler.chain, 
		# phase = 'Burn')
	# final run
	sampler.reset()
	sampler.run_mcmc(pos, final_iterations)
	# remove steps where lnprob = -np.inf
	chain = [[sampler.chain[walker][step] for step in 
		range(len(sampler.chain[walker])) if 
		sampler.lnprobability[walker][step] != -np.inf] for walker in 
		range(len(sampler.chain))]
	if np.array(chain).shape[0] == 0:
		print('No finite members of posterior')
		return (np.array([]), np.array([]))
	else:
		lnprob_ = [[a for a in sampler.lnprobability[walker] if a != -np.inf] 
			for walker in range(len(sampler.lnprobability))]
		return (np.array(chain), np.array(lnprob_))
def lnprob(lnprobx = None, modified_data = None, p = [], vel_range = None, 
	num_gauss = None, accepted_params = [], use_molex = True, molex_path = None, file_suffix = None): # returns lnprob
	if use_molex:
		if p == []:
			p = plist(lnprobx, modified_data, num_gauss)
		prior = lnprprior(modified_data = modified_data, p = p, 
			vel_range = vel_range, num_gauss = num_gauss)
		params = molex(p = p, modified_data = modified_data, 
			num_gauss = num_gauss, molex_path = molex_path, 
			file_suffix = file_suffix)
	else:
		prior = lnprprior(modified_data = modified_data, params = lnprobx, 
			vel_range = vel_range, num_gauss = num_gauss, use_molex = False)
		params = lnprobx
	models = makemodel(params = params, modified_data = modified_data, 
		accepted_params = accepted_params, num_gauss = num_gauss, 
		use_molex = use_molex)
	
	if modified_data['Texp_spectrum']['1665'] != []:
		[tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, 
		Texp_m_1665, Texp_m_1667, Texp_m_1720] = models
		spectra = [	modified_data['tau_spectrum']['1612'], 
					modified_data['tau_spectrum']['1665'], 
					modified_data['tau_spectrum']['1667'], 
					modified_data['tau_spectrum']['1720'], 
					modified_data['Texp_spectrum']['1612'], 
					modified_data['Texp_spectrum']['1665'], 
					modified_data['Texp_spectrum']['1667'], 
					modified_data['Texp_spectrum']['1720']]
		rms = [		modified_data['tau_rms']['1612'], 
					modified_data['tau_rms']['1665'], 
					modified_data['tau_rms']['1667'], 
					modified_data['tau_rms']['1720'], 
					modified_data['Texp_rms']['1612'], 
					modified_data['Texp_rms']['1665'], 
					modified_data['Texp_rms']['1667'], 
					modified_data['Texp_rms']['1720']]
	else:
		[tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720] = models
		spectra = [	modified_data['tau_spectrum']['1612'], 
					modified_data['tau_spectrum']['1665'], 
					modified_data['tau_spectrum']['1667'], 
					modified_data['tau_spectrum']['1720']]
		rms = [		modified_data['tau_rms']['1612'], 
					modified_data['tau_rms']['1665'], 
					modified_data['tau_rms']['1667'], 
					modified_data['tau_rms']['1720']]
	lprob = prior 
	for a in range(len(spectra)):
		llh = lnlikelihood(model = models[a], spectrum = spectra[a], 
			sigma = rms[a])
		# print('prior: ' + str(prior) + '\tlikelihood: ' + str(llh))
		lprob += llh
	if np.isnan(lprob):
		# print('lnprob wanted to return NaN for parameters: ' + str(lnprobx))
		return -np.inf
	else:
		return lprob	
def lnprprior(modified_data = None, p = [], params = [], vel_range = None, num_gauss = None, use_molex = True): # returns lnprprior
	lnprprior = 0
	if use_molex:
		parameter_list = [True] + modified_data['parameter_list'] # add velocity
		vel_prev = vel_range[0]
		for gauss in range(num_gauss):
			# define params
			[vel, logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd] = p[int(gauss * 12):int((gauss + 1) * 12)]
			#calculate priors
			vel_prior = lnnaiveprior(value = vel, value_range = vel_range)
			logTgas_prior = lnnaiveprior(value = logTgas, value_range = logTgas_range)
			lognH2_prior = lnnaiveprior(value = lognH2, value_range = lognH2_range)
			logNOH_prior = lnnaiveprior(value = logNOH, value_range = logNOH_range)
			fortho_prior = lnnaiveprior(value = fortho, value_range = fortho_range)
			FWHM_prior = lnnaiveprior(value = FWHM, value_range = FWHM_range)
			Av_prior = lnnaiveprior(value = Av, value_range = Av_range)
			logxOH_prior = lnnaiveprior(value = logxOH, value_range = logxOH_range)
			logxHe_prior = lnnaiveprior(value = logxHe, value_range = logxHe_range)
			logxe_prior = lnnaiveprior(value = logxe, value_range = logxe_range)
			logTdint_prior = lnnaiveprior(value = logTdint, value_range = logTdint_range)
			logTd_prior = lnnaiveprior(value = logTd, value_range = logTd_range)

			priors = [vel_prior, logTgas_prior, lognH2_prior, logNOH_prior, 
				fortho_prior, FWHM_prior, Av_prior, logxOH_prior, logxHe_prior, 
				logxe_prior, logTdint_prior, logTd_prior]

			# print('values: ' + str([vel, logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd]))
			priors = [priors[a] for a in range(len(priors)) if type(parameter_list[a]) == bool and parameter_list[a] == True]
			lnprprior = lnprprior + np.sum(priors)
			vel_prev = vel

	else:
		if modified_data['Texp_spectrum']['1665'] != []:
			[logN1_range, logN2_range, logN3_range, logN4_range] = modified_data['N_ranges']
			# [logN1_range, logN2_range, logN3_range, logN4_range] = NrangetauTexp(modified_data)
		vel_prev = vel_range[0]
		for gauss in range(num_gauss):
			# define params
			if modified_data['Texp_spectrum']['1665'] != []:
				[vel, FWHM, logN1, logN2, logN3, logN4] = params[int(gauss * 6):int((gauss + 1) * 6)]
				[tau_1612, tau_1665, tau_1667, tau_1720, Tex_1612, Tex_1665, Tex_1667, Tex_1720] = tauTexN(logN1, logN2, logN3, logN4, FWHM)
			else:
				[vel, FWHM, tau_1612, tau_1667, tau_1720] = params[int(gauss * 5):int((gauss + 1) * 5)]
				[tau_1612, tau_1665, tau_1667, tau_1720] = tau3(tau_1612 = tau_1612, tau_1667 = tau_1667, tau_1720 = tau_1720)
			# calculate priors
			vel_prior = lnnaiveprior(value = vel, value_range = [vel_prev, vel_range[1]])
			FWHM_prior = lnnaiveprior(value = FWHM, value_range = FWHM_range)
			if modified_data['Texp_spectrum']['1665'] != []:
				logN1_prior = lnnaiveprior(value = logN1, value_range = logN1_range)
				logN2_prior = lnnaiveprior(value = logN2, value_range = logN2_range)
				logN3_prior = lnnaiveprior(value = logN3, value_range = logN3_range)
				logN4_prior = lnnaiveprior(value = logN4, value_range = logN4_range)
				lnprprior += vel_prior + FWHM_prior + logN1_prior + logN2_prior + logN3_prior + logN4_prior				
			else:
				(tau_1612_range, tau_1665_range, tau_1667_range, tau_1720_range) = (
					[-2 * np.abs(np.amin(modified_data['tau_spectrum']['1612'])), 2 * np.abs(np.amax(modified_data['tau_spectrum']['1612']))], 
					[-1.5 * np.abs(np.amin(modified_data['tau_spectrum']['1665'])), 1.5 * np.abs(np.amax(modified_data['tau_spectrum']['1665']))], 
					[-1.5 * np.abs(np.amin(modified_data['tau_spectrum']['1667'])), 1.5 * np.abs(np.amax(modified_data['tau_spectrum']['1667']))], 
					[-2 * np.abs(np.amin(modified_data['tau_spectrum']['1720'])), 2 * np.abs(np.amax(modified_data['tau_spectrum']['1720']))])
				tau_1612_prior = lnnaiveprior(value = tau_1612, value_range = tau_1612_range)
				tau_1665_prior = lnnaiveprior(value = tau_1665, value_range = tau_1665_range)
				tau_1667_prior = lnnaiveprior(value = tau_1667, value_range = tau_1667_range)
				tau_1720_prior = lnnaiveprior(value = tau_1720, value_range = tau_1720_range)
				lnprprior += vel_prior + FWHM_prior + tau_1612_prior + tau_1665_prior + tau_1667_prior + tau_1720_prior
	return lnprprior
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
	for comp in range(num_gauss):

		var_within_chains = np.median([np.var(sampler_chain[x,-25:-1,comp * model_dim]) for x in range(sampler_chain.shape[0])])
		var_across_chains = np.median([np.var(sampler_chain[:,-x-1,comp * model_dim]) for x in range(24)])
		ratio = max([var_within_chains, var_across_chains]) / min([var_within_chains, var_across_chains])
		max_var = max([var_within_chains, var_across_chains])

		if ratio > 15. and max_var < 1.:
			return ('Fail', sampler_chain[:,-1,:])

	return ('Pass', sampler_chain[:,-1,:])
def splitwalkers(chain, lnprob, tolerance = 10):
	'''
	Note: This may not be needed if I increase 'a' in the EnsembleSampler (a relates to step size or acceptance rate or similar)
	'''
	lnprob = np.array([[[x] for x in y] for y in lnprob])
	chain = np.array(chain)
	try:
		num_params = chain.shape[2]
		num_steps = chain.shape[1]
		num_walkers = chain.shape[0]
	except IndexError:
		return ([chain], [lnprob])

	# print(str([num_walkers, num_steps, num_params]))
	
	combined_chain = np.append(lnprob, chain, axis = 2)
	# print('combined chain: ' + str(combined_chain))
	groups = [combined_chain]
	
	for param in range(1, num_params):
		for full_chain in groups:
			full_chain = np.array(full_chain)
		
			# print('param: ' + str(param))
			b = full_chain[:, -1, param].argsort() # args that would sort all the walkers (:) by their final (-1) value of parameter 'param'
			# print('b: ' + str(b))
			param_spacings = [full_chain[b[x], -1, param] - full_chain[b[x-1], -1, param] for x in range(1, full_chain.shape[0])] # space between neighbouring walkers (x and x-1) in their final (-1) step in parameter 'param'
			# print('param_spacings: ' + str(param_spacings))
			med_spacing = np.median(param_spacings) # median spacing
			# print('med_spacing: ' + str(med_spacing))


			c = np.argwhere(param_spacings > tolerance * med_spacing) # args of spacing outliers
			c = [x[0] + 1 for x in c] # args of the first value in the new group
			# print('c: ' + str(c))
			if len(c) != 0:


				first_walker_index = b[c]
				# print('first_walker_index: ' + str(first_walker_index))
				first_walker_value = full_chain[first_walker_index, -1, param]
				# print('first_walker_value: ' + str(first_walker_value))
				
				group2 = np.array(full_chain)
				# print('group2: ' + str(group2))
				groups = []
				# print('groups: ' + str(groups))
				for g in range(len(first_walker_value)):
					group1 = [group2[x] for x in range(group2.shape[0]) if group2[x, -1, param] < first_walker_value[g]]
					group2 = [group2[x] for x in range(group2.shape[0]) if not group2[x, -1, param] < first_walker_value[g]]
					groups += [group1]
					group2 = np.array(group2)
					# print('groups: ' + str(groups))

				groups += [group2]

	if len(groups) == 1:
		return ([chain.reshape(num_walkers * num_steps, num_params)], 
				[lnprob.reshape(num_walkers * num_steps)])
	else:
		grouped_chains = []
		grouped_lnprob = []
		for group in groups:
			group = np.array(group)
			chain = group[:, :, 1:]
			# print('chain before: ' + str(chain))
			chain = chain.reshape(chain.shape[0] * chain.shape[1], chain.shape[2])
			# print('chain after: ' + str(chain))
			lnprob = group[:, :, 0]
			lnprob = lnprob.reshape(lnprob.shape[0] * lnprob.shape[1])
			grouped_chains += [chain]
			grouped_lnprob += [lnprob]
		# print('grouped_chains: ' + str(grouped_chains))
		# print('grouped_lnprob: ' + str(grouped_lnprob))
		return (grouped_chains, grouped_lnprob)
# priors
def lnnaiveprior(value = None, value_range = None): # returns lnpriorvalue for naive top-hat prior
	if value >= value_range[0] and value <= value_range[1]:
		return -np.log(np.abs(value_range[1] - value_range[0]))
	else:
		# print('returning infinite naive prior. Value of ' + str(value) + ' is outside range of ' + str(value_range))
		return -np.inf
# reports
def resultsreport(final_parameters = None, final_median_parameters = None, data = None, file_preamble = None):
	'''
	Generates a nice report of results
	'''
	short_source_name_dict = {'g003.74+0.64.':'g003', 
			'g006.32+1.97.':'g006', 
			'g007.47+0.06.':'g007', 
			'g334.72-0.65.':'g334', 
			'g336.49-1.48.':'g336', 
			'g340.79-1.02a.':'g340a', 
			'g340.79-1.02b.':'g340b', 
			'g344.43+0.05.':'g344', 
			'g346.52+0.08.':'g346', 
			'g347.75-1.14.':'g347', 
			'g348.44+2.08.':'g348', 
			'g349.73+1.67.':'g349', 
			'g350.50+0.96.':'g350', 
			'g351.61+0.17a.':'g351a', 
			'g351.61+0.17b.':'g351b', 
			'g353.411-0.3.':'g353', 
			'g356.91+0.08.':'g356'}

	try:
		os.makedir('pickles')
	except:
		pass

	pickle.dump(final_parameters, open('pickles/RESULTS_' + str(file_preamble) + '_' + short_source_name_dict[data['source_name']] + '.pickle', 'w'))
	
	# PlotFinalModel(final_parameters = final_parameters, data = data, file_preamble = file_preamble)
	print('\nResults for ' + data['source_name'])
	
	if data['Texp_spectrum']['1665'] != []:
		print('\tNumber of features identified: ' + str(len(final_parameters) / 10) + '\n')
	else:
		print('\tNumber of features identified: ' + str(len(final_parameters) / 6) + '\n'	)

	if len(final_parameters) > 5:

		print('\tBackground Temperatures [1612, 1665, 1667, 1720MHz] = ' + str([data['Tbg']['1612'], data['Tbg']['1665'], data['Tbg']['1667'], data['Tbg']['1720']]))
		final_parameters = final_parameters
		print('\tFeature Parameters [16th, 50th, 84th quantiles]:')
		
		if data['Texp_spectrum']['1665'] != []:
			for feature in range(int(len(final_parameters) / 10)):
				print('\t\tfeature number ' + str(feature + 1) + ':')
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84], [Texp_1612_16, Texp_1612_50, Texp_1612_84], [Texp_1665_16, Texp_1665_50, Texp_1665_84], [Texp_1667_16, Texp_1667_50, Texp_1667_84], [Texp_1720_16, Texp_1720_50, Texp_1720_84]] = final_parameters[feature * 10:feature * 10 + 10]

				print('\t\t\tcentroid velocity = ' + str([vel_16, vel_50, vel_84]) + ' km/sec')
				print('\t\t\tfwhm = ' + str([fwhm_16, fwhm_50, fwhm_84]) + ' km/sec')
				print('\n\t\t\t1612MHz peak tau = ' + str([tau_1612_16, tau_1612_50, tau_1612_84]))
				print('\t\t\t1665MHz peak tau = ' + str([tau_1665_16, tau_1665_50, tau_1665_84]))
				print('\t\t\t1667MHz peak tau = ' + str([tau_1667_16, tau_1667_50, tau_1667_84]))
				print('\t\t\t1720MHz peak tau = ' + str([tau_1720_16, tau_1720_50, tau_1720_84]))
				print('\n\t\t\t1612MHz Texp = ' + str([Texp_1612_16, Texp_1612_50, Texp_1612_84]) + ' K')
				print('\t\t\t1665MHz Texp = ' + str([Texp_1665_16, Texp_1665_50, Texp_1665_84]) + ' K')
				print('\t\t\t1667MHz Texp = ' + str([Texp_1667_16, Texp_1667_50, Texp_1667_84]) + ' K')
				print('\t\t\t1720MHz Texp = ' + str([Texp_1720_16, Texp_1720_50, Texp_1720_84]) + ' K')

		else:
			for feature in range(int(len(final_parameters) / 6)):
				print('\t\tfeature number ' + str(feature + 1) + ':')
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84]] = final_parameters[feature * 6:feature * 6 + 6]

				print('\t\t\tcentroid velocity = ' + str([vel_16, vel_50, vel_84]) + ' km/sec')
				print('\t\t\tfwhm = ' + str([fwhm_16, fwhm_50, fwhm_84]) + ' km/sec')
				print('\n\t\t\t1612MHz peak tau = ' + str([tau_1612_16, tau_1612_50, tau_1612_84]))
				print('\t\t\t1665MHz peak tau = ' + str([tau_1665_16, tau_1665_50, tau_1665_84]))
				print('\t\t\t1667MHz peak tau = ' + str([tau_1667_16, tau_1667_50, tau_1667_84]))
				print('\t\t\t1720MHz peak tau = ' + str([tau_1720_16, tau_1720_50, tau_1720_84]))
def resultstable(f_param = None, data = None, use_molex = None):
	'''
	make a latex table
	'''
	if len(f_param) > 0:
		if data['Texp_spectrum']['1665'] != []:
			for comp in range(int(len(f_param) / 6)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], 
				[logN1_16, logN1_50, logN1_84], [logN2_16, logN2_50, logN2_84], 
				[logN3_16, logN3_50, logN3_84], 
				[logN4_16, logN4_50, logN4_84]] = f_param[comp*6:comp*6+6]
				[vel_16, vel_50, vel_84] = [round(x, 1) 
					for x in [vel_16, vel_50, vel_84]]
				[fwhm_16, fwhm_50, fwhm_84] = [round(x, 2) 
					for x in [fwhm_16, fwhm_50, fwhm_84]]
				[tau_1612_16,tau_1665_16,tau_1667_16,tau_1720_16,
					Tex_1612_16,Tex_1665_16,Tex_1667_16,Tex_1720_16] = tauTexN(
						logN1 = logN1_16, logN2 = logN2_16, logN3 = logN3_16, 
						logN4 = logN4_16, fwhm = fwhm_16)
				[tau_1612_50,tau_1665_50,tau_1667_50,tau_1720_50,
					Tex_1612_50,Tex_1665_50,Tex_1667_50,Tex_1720_50] = tauTexN(
						logN1 = logN1_50, logN2 = logN2_50, logN3 = logN3_50, 
						logN4 = logN4_50, fwhm = fwhm_50)
				[tau_1612_84,tau_1665_84,tau_1667_84,tau_1720_84,
					Tex_1612_84,Tex_1665_84,Tex_1667_84,Tex_1720_84] = tauTexN(
						logN1 = logN1_84, logN2 = logN2_84, logN3 = logN3_84, 
						logN4 = logN4_84, fwhm = fwhm_84)
				Texp_1612_16 = Texp(tau = tau_1612_16, 
						Tbg = data['Tbg']['1612'], Tex = Tex_1612_16)
				Texp_1665_16 = Texp(tau = tau_1665_16, 
						Tbg = data['Tbg']['1665'], Tex = Tex_1665_16)
				Texp_1667_16 = Texp(tau = tau_1667_16, 
						Tbg = data['Tbg']['1667'], Tex = Tex_1667_16)
				Texp_1720_16 = Texp(tau = tau_1720_16, 
						Tbg = data['Tbg']['1720'], Tex = Tex_1720_16)
				Texp_1612_50 = Texp(tau = tau_1612_50, 
						Tbg = data['Tbg']['1612'], Tex = Tex_1612_50)
				Texp_1665_50 = Texp(tau = tau_1665_50, 
						Tbg = data['Tbg']['1665'], Tex = Tex_1665_50)
				Texp_1667_50 = Texp(tau = tau_1667_50, 
						Tbg = data['Tbg']['1667'], Tex = Tex_1667_50)
				Texp_1720_50 = Texp(tau = tau_1720_50, 
						Tbg = data['Tbg']['1720'], Tex = Tex_1720_50)
				Texp_1612_84 = Texp(tau = tau_1612_84, 
						Tbg = data['Tbg']['1612'], Tex = Tex_1612_84)
				Texp_1665_84 = Texp(tau = tau_1665_84, 
						Tbg = data['Tbg']['1665'], Tex = Tex_1665_84)
				Texp_1667_84 = Texp(tau = tau_1667_84, 
						Tbg = data['Tbg']['1667'], Tex = Tex_1667_84)
				Texp_1720_84 = Texp(tau = tau_1720_84, 
						Tbg = data['Tbg']['1720'], Tex = Tex_1720_84)

				[tau_1612_16, tau_1612_50, tau_1612_84, 
				tau_1665_16, tau_1665_50, tau_1665_84, 
				tau_1667_16, tau_1667_50, tau_1667_84, 
				tau_1720_16, tau_1720_50, tau_1720_84, 
				Texp_1612_16, Texp_1612_50, Texp_1612_84, 
				Texp_1665_16, Texp_1665_50, Texp_1665_84, 
				Texp_1667_16, Texp_1667_50, Texp_1667_84, 
				Texp_1720_16, Texp_1720_50, Texp_1720_84] = [round(x, 3) 
					for x in [tau_1612_16, tau_1612_50, tau_1612_84, 
								tau_1665_16, tau_1665_50, tau_1665_84, 
								tau_1667_16, tau_1667_50, tau_1667_84, 
								tau_1720_16, tau_1720_50, tau_1720_84, 
								Texp_1612_16, Texp_1612_50, Texp_1612_84, 
								Texp_1665_16, Texp_1665_50, Texp_1665_84, 
								Texp_1667_16, Texp_1667_50, Texp_1667_84, 
								Texp_1720_16, Texp_1720_50, Texp_1720_84]]
				print(data['source_name'] + '&' + str(vel_50) + '$^{+' + str(np.abs(vel_84 - vel_50)) + '}_{-' + str(np.abs(vel_50 - vel_16)) + '}$' + '&' + 
						str(fwhm_50) + '$^{+' + str(np.abs(fwhm_84 - fwhm_50)) + '}_{-' + str(np.abs(fwhm_50 - fwhm_16)) + '}$' + '&' + 

						str(logN1_50) + '$^{+' + str(np.abs(logN1_84 - logN1_50)) + '}_{-' + str(np.abs(logN1_50 - logN1_16)) + '}$' + '&' + 
						str(logN2_50) + '$^{+' + str(np.abs(logN2_84 - logN2_50)) + '}_{-' + str(np.abs(logN2_50 - logN2_16)) + '}$' + '&' + 
						str(logN3_50) + '$^{+' + str(np.abs(logN3_84 - logN3_50)) + '}_{-' + str(np.abs(logN3_50 - logN3_16)) + '}$' + '&' + 
						str(logN4_50) + '$^{+' + str(np.abs(logN4_84 - logN4_50)) + '}_{-' + str(np.abs(logN4_50 - logN4_16)) + '}$' + '&' + 

						str(Tex_1612_50) + '$^{+' + str(np.abs(Tex_1612_84 - Tex_1612_50)) + '}_{-' + str(np.abs(Tex_1612_50 - Tex_1612_16)) + '}$' + '&' + 
						str(Tex_1665_50) + '$^{+' + str(np.abs(Tex_1665_84 - Tex_1665_50)) + '}_{-' + str(np.abs(Tex_1665_50 - Tex_1665_16)) + '}$' + '&' + 
						str(Tex_1667_50) + '$^{+' + str(np.abs(Tex_1667_84 - Tex_1667_50)) + '}_{-' + str(np.abs(Tex_1667_50 - Tex_1667_16)) + '}$' + '&' + 
						str(Tex_1720_50) + '$^{+' + str(np.abs(Tex_1720_84 - Tex_1720_50)) + '}_{-' + str(np.abs(Tex_1720_50 - Tex_1720_16)) + '}$' + '&' + 

						str(tau_1612_50) + '$^{+' + str(np.abs(tau_1612_84 - tau_1612_50)) + '}_{-' + str(np.abs(tau_1612_50 - tau_1612_16)) + '}$' + '&' + 
						str(tau_1665_50) + '$^{+' + str(np.abs(tau_1665_84 - tau_1665_50)) + '}_{-' + str(np.abs(tau_1665_50 - tau_1665_16)) + '}$' + '&' + 
						str(tau_1667_50) + '$^{+' + str(np.abs(tau_1667_84 - tau_1667_50)) + '}_{-' + str(np.abs(tau_1667_50 - tau_1667_16)) + '}$' + '&' + 
						str(tau_1720_50) + '$^{+' + str(np.abs(tau_1720_84 - tau_1720_50)) + '}_{-' + str(np.abs(tau_1720_50 - tau_1720_16)) + '}$' + '&' + 

						str(Texp_1612_50) + '$^{+' + str(np.abs(Texp_1612_84 - Texp_1612_50)) + '}_{-' + str(np.abs(Texp_1612_50 - Texp_1612_16)) + '}$' + '&' + 
						str(Texp_1665_50) + '$^{+' + str(np.abs(Texp_1665_84 - Texp_1665_50)) + '}_{-' + str(np.abs(Texp_1665_50 - Texp_1665_16)) + '}$' + '&' + 
						str(Texp_1667_50) + '$^{+' + str(np.abs(Texp_1667_84 - Texp_1667_50)) + '}_{-' + str(np.abs(Texp_1667_50 - Texp_1667_16)) + '}$' + '&' + 
						str(Texp_1720_50) + '$^{+' + str(np.abs(Texp_1720_84 - Texp_1720_50)) + '}_{-' + str(np.abs(Texp_1720_50 - Texp_1720_16)) + '}$' + '\\\\')

		else:
			for comp in range(int(len(f_param) / 5)):
				[[vel_16,vel_50,vel_84],[fwhm_16,fwhm_50,fwhm_84], 
				[tau_1612_16,tau_1612_50,tau_1612_84],
				[tau_1667_16,tau_1667_50,tau_1667_84],
				[tau_1720_16,tau_1720_50,tau_1720_84]]=f_param[comp*5:comp*5+5]
				
				[_, tau_1665_16, _, _] = tau3(tau_1612 = tau_1612_16, 
					tau_1667 = tau_1667_16, tau_1720 = tau_1720_16)
				[_, tau_1665_50, _, _] = tau3(tau_1612 = tau_1612_50, 
					tau_1667 = tau_1667_50, tau_1720 = tau_1720_50)
				[_, tau_1665_84, _, _] = tau3(tau_1612 = tau_1612_84, 
					tau_1667 = tau_1667_84, tau_1720 = tau_1720_84)

				[velm, velp] = [vel_50 - vel_16, vel_84 - vel_50]
				[fwhmm, fwhmp] = [fwhm_50 - fwhm_16, fwhm_84 - fwhm_50]
				[tau_1612m, tau_1612p] = [tau_1612_50 - tau_1612_16, 
											tau_1612_84 - tau_1612_50]
				[tau_1665m, tau_1665p] = [tau_1665_50 - tau_1665_16, 
											tau_1665_84 - tau_1665_50]
				[tau_1667m, tau_1667p] = [tau_1667_50 - tau_1667_16, 
											tau_1667_84 - tau_1667_50]
				[tau_1720m, tau_1720p] = [tau_1720_50 - tau_1720_16, 
											tau_1720_84 - tau_1720_50]
				[velm, vel_50, velp] = [round(x, 2) for x in [velm, vel_50, velp]]
				[fwhmm, fwhm_50, fwhmp] = [round(x, 2) for x in [fwhmm, fwhm_50, fwhmp]]
				[tau_1612m, tau_1612_50, tau_1612p, 
					tau_1665m, tau_1665_50, tau_1665p, 
					tau_1667m, tau_1667_50, tau_1667p, 
					tau_1720m, tau_1720_50, tau_1720p] = [int(round(1000*x, 0))
					for x in [tau_1612m, tau_1612_50, tau_1612p, 
						tau_1665m, tau_1665_50, tau_1665p, 
						tau_1667m, tau_1667_50, tau_1667p, 
						tau_1720m, tau_1720_50, tau_1720p]]
				print(data['source_name'] + '&' + 
					str(vel_50) + '$^{+' + str(np.abs(velp)) + '}_{-' + str(np.abs(velm)) + '}$' + '&' + 
					str(fwhm_50) + '$^{+' + str(np.abs(fwhmp)) + '}_{-' + str(np.abs(fwhmm)) + '}$' + '&' + 
					str(tau_1612_50) + '$^{+' + str(np.abs(tau_1612p)) + '}_{-' + str(np.abs(tau_1612m)) + '}$' + '&' + 
					str(tau_1665_50) + '$^{+' + str(np.abs(tau_1665p)) + '}_{-' + str(np.abs(tau_1665m)) + '}$' + '&' + 
					str(tau_1667_50) + '$^{+' + str(np.abs(tau_1667p)) + '}_{-' + str(np.abs(tau_1667m)) + '}$' + '&' + 
					str(tau_1720_50) + '$^{+' + str(np.abs(tau_1720p)) + '}_{-' + str(np.abs(tau_1720m)) + '}$' + '\\\\')
def resultstableexcel(final_parameters = None, final_median_parameters = None, data = None):
	'''
	make an excel table
	'''
	if len(final_parameters) > 5:

		final_parameters = final_parameters
		
		if data['Texp_spectrum']['1665'] != []:
			for feature in range(int(len(final_parameters) / 10)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84], [Texp_1612_16, Texp_1612_50, Texp_1612_84], [Texp_1665_16, Texp_1665_50, Texp_1665_84], [Texp_1667_16, Texp_1667_50, Texp_1667_84], [Texp_1720_16, Texp_1720_50, Texp_1720_84]] = final_parameters[feature * 10:feature * 10 + 10]
				print(data['source_name'] + ' \t ' + str(vel_50) + ' \t ' + str(vel_84 - vel_50) + ' \t ' + str(vel_50 - vel_16) + ' \t ' + str(fwhm_50) + ' \t ' + str(fwhm_84 - fwhm_50) + ' \t ' + str(fwhm_50 - fwhm_16) + ' \t ' + str(tau_1612_50) + ' \t ' + str(tau_1612_84 - tau_1612_50) + ' \t ' + str(tau_1612_50 - tau_1612_16) + ' \t ' + str(tau_1665_50) + ' \t ' + str(tau_1665_84 - tau_1665_50) + ' \t ' + str(tau_1665_50 - tau_1665_16) + ' \t ' + str(tau_1667_50) + ' \t ' + str(tau_1667_84 - tau_1667_50) + ' \t ' + str(tau_1667_50 - tau_1667_16) + ' \t ' + str(tau_1720_50) + ' \t ' + str(tau_1720_84 - tau_1720_50) + ' \t ' + str(tau_1720_50 - tau_1720_16) + ' \t ' + str(Texp_1612_50) + ' \t ' + str(Texp_1612_84 - Texp_1612_50) + ' \t ' + str(Texp_1612_50 - Texp_1612_16) + ' \t ' + str(Texp_1665_50) + ' \t ' + str(Texp_1665_84 - Texp_1665_50) + ' \t ' + str(Texp_1665_50 - Texp_1665_16) + ' \t ' + str(Texp_1667_50) + ' \t ' + str(Texp_1667_84 - Texp_1667_50) + ' \t ' + str(Texp_1667_50 - Texp_1667_16) + ' \t ' + str(Texp_1720_50) + ' \t ' + str(Texp_1720_84 - Texp_1720_50) + ' \t ' + str(Texp_1720_50 - Texp_1720_16))

		else:
			for feature in range(int(len(final_parameters) / 6)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84]] = final_parameters[feature * 6:feature * 6 + 6]
				print(data['source_name'] + ' \t ' + str(vel_50) + ' \t ' + str(vel_84 - vel_50) + ' \t ' + str(vel_50 - vel_16) + ' \t ' + str(fwhm_50) + ' \t ' + str(fwhm_84 - fwhm_50) + ' \t ' + str(fwhm_50 - fwhm_16) + ' \t ' + str(tau_1612_50) + ' \t ' + str(tau_1612_84 - tau_1612_50) + ' \t ' + str(tau_1612_50 - tau_1612_16) + ' \t ' + str(tau_1665_50) + ' \t ' + str(tau_1665_84 - tau_1665_50) + ' \t ' + str(tau_1665_50 - tau_1665_16) + ' \t ' + str(tau_1667_50) + ' \t ' + str(tau_1667_84 - tau_1667_50) + ' \t ' + str(tau_1667_50 - tau_1667_16) + ' \t ' + str(tau_1720_50) + ' \t ' + str(tau_1720_84 - tau_1720_50) + ' \t ' + str(tau_1720_50 - tau_1720_16))

#############################
#                           #
#   M   M        i          #
#   MM MM   aa      n nn    #
#   M M M   aaa  i  nn  n   #
#   M   M  a  a  i  n   n   #
#   M   M   aaa  i  n   n   #
#                           #
#############################

def main( # prints final_p
	source_name = None,
	vel_axes = None, 
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
	use_molex = True, 
	a = 2.0, 
	file_suffix = None, 
	molex_path = None, 
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
	vel_axes - list of velocity alogxes: [vel_axis_1612, vel_axis_1665, vel_axis_1667, vel_axis_1720]
	spectra - list of spectra (brightness temperature or tau): [spectrum_1612, spectrum_1665, spectrum_1667, 
			spectrum_1720]
	rms - list of estimates of rms error in spectra: [rms_1612, rms_1665, rms_1667, rms_1720]. Used by AGD
	expected_min_FWHM - estimate of the minimum full width at half-maximum of features expected in the data 
			in km/sec. Used when categorising features as isolated or blended.
	Returns parameters of gaussian comp(s): [vel_1, FWHM_1, height_1612_1, height_1665_1, height_1667_1, 
			height_1720_1, ..., _N] for N comps
	'''
	print('source name: ' + str(source_name))
	if file_suffix == None:
		file_suffix = str(datetime.datetime.now())
	# initialise data dictionary
	# print('starting ' + source_name)
	data = {'source_name': source_name, 'vel_axis': {'1612': [], '1665': [], '1667': [], '1720': []}, 'tau_spectrum': {'1612': [], '1665': [], '1667': [], '1720': []}, 'tau_rms': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Texp_spectrum': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Texp_rms': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Tbg': {'1612': [], '1665': [], '1667': [], '1720': []}, 'parameter_list': [logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd], 'sig_vel_ranges': [[]], 'interesting_vel': []}

		###############################################
		#                                             #
		#   Load Data into 'data' dictionary object   #
		#                                             #	
		###############################################

	data['vel_axis']['1612']		= vel_axes[0]
	data['tau_spectrum']['1612']	= tau_spectra[0]
	data['tau_rms']['1612']			= tau_rms[0]

	data['vel_axis']['1665']		= vel_axes[1]
	data['tau_spectrum']['1665']	= tau_spectra[1]
	data['tau_rms']['1665']			= tau_rms[1]

	data['vel_axis']['1667']		= vel_axes[2]
	data['tau_spectrum']['1667']	= tau_spectra[2]
	data['tau_rms']['1667']			= tau_rms[2]

	data['vel_axis']['1720']		= vel_axes[3]
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
	
	final_p = placegaussians(data, Bayes_threshold, use_molex = use_molex, molex_path = molex_path, a = a, test = test, file_suffix = file_suffix)

	return final_p








