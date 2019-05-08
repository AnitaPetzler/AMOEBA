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
lognH2_range    = [-2, 7]
logNOH_range    = [11, 16]
fortho_range    = [0, 1]
FWHM_range      = [0.1, 15]
Av_range        = [0, 10]
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
def placegaussians(data = None, Bayes_threshold = 10, use_molex = True, a = None, test = False): # returns final_p
	'''
	mpfit, emcee, decide whether or not to add another gaussian
	'''
	accepted_full = []
	total_num_gauss = 0
	plot_num = 0
	for vel_range in data['sig_vel_ranges']:
	# for vel_range in [[-5, 10]]:
		# print('vel_range: ' + str(vel_range))
		last_accepted_full = []
		[min_vel, max_vel] = vel_range
		modified_data = trimdata(data, min_vel, max_vel)

		num_gauss = 1
		keep_going = True
		(null_evidence, _, _) = nullevidence(modified_data)
		# print('null_evidence = ' + str(null_evidence))
		prev_evidence = null_evidence
		evidences = [prev_evidence]
		print(data['source_name'] + '\t' + str(vel_range) + '\t' + str(null_evidence))

		while keep_going == True:
			nwalkers = 30 * num_gauss
			p0 = p0gen(	vel_range = vel_range, 
						num_gauss = num_gauss, 
						modified_data = modified_data, 
						accepted = accepted_full,
						last_accepted = last_accepted_full, 
						nwalkers = nwalkers, 
						use_molex = use_molex)
			(chain, lnprob_) = sampleposterior(	modified_data = modified_data, 
												num_gauss = num_gauss, 
												p0 = p0, 
												vel_range = [min_vel, max_vel],  
												accepted = accepted_full, 
												nwalkers = nwalkers, 
												use_molex = use_molex, 
												a = a)

			if len(chain) != 0:
				(current_full, current_evidence) = bestparams(chain, lnprob_)
				# if use_molex:
				# 	plotmodel(data = modified_data, x = [x[1] for x in current_full], num_gauss = num_gauss, plot_num = plot_num)
				# else:
				# 	plotmodel(data = modified_data, params = [x[1] for x in current_full], num_gauss = num_gauss, plot_num = plot_num)
				plot_num += 1
				# print('evidence for ' + str(num_gauss) + ' = ' + str(current_evidence))
				# print('evidences: ' + str(evidences))
				evidences += [current_evidence]
				if current_evidence - prev_evidence > Bayes_threshold: # working on some tests to refine this
					last_accepted_full = current_full
					prev_evidence = current_evidence
				# if num_gauss < 5:
					num_gauss += 1
					total_num_gauss += 1
				else:
					keep_going = False
			else:
				keep_going = False
		accepted_full = list(itertools.chain(accepted_full, last_accepted_full))
	# if len(accepted_full) != 0:
	# 	if use_molex:
	# 		plotmodel(data = modified_data, x = [x[1] for x in accepted_full], num_gauss = total_num_gauss, plot_num = 'Final')
	# 	else:
	# 		plotmodel(data = modified_data, params = [x[1] for x in accepted_full], num_gauss = total_num_gauss, plot_num = 'Final')
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

	# plotchain(modified_data = modified_data, chain = sampler.chain, phase = 'Null')
	(params, evidence) = bestparams(chain = sampler.chain, lnprob = sampler.lnprobability)
	# print('params for null: ' + str(params))

	# plt.figure()
	# plt.plot(modified_data['vel_axis']['1612'], modified_data['tau_spectrum']['1612'], color = 'blue')
	# plt.plot(modified_data['vel_axis']['1665'], modified_data['tau_spectrum']['1665'], color = 'green')
	# plt.plot(modified_data['vel_axis']['1667'], modified_data['tau_spectrum']['1667'], color = 'red')
	# plt.plot(modified_data['vel_axis']['1720'], modified_data['tau_spectrum']['1720'], color = 'cyan')
	# plt.hlines(params[0][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'blue')
	# plt.hlines(params[1][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'green')
	# plt.hlines(params[2][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'red')
	# plt.hlines(params[3][1], np.amin(modified_data['vel_axis']['1612']), np.amax(modified_data['vel_axis']['1612']), color = 'cyan')
	# plt.title('Null Model')
	# plt.savefig('Plots/null_model.pdf')
	# plt.close()
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
				accumulated_evidence[step] = np.logaddexp(accumulated_evidence[step - 1], contribution_to_lnevidence)

		total_evidence = accumulated_evidence[-1]
		
		if total_evidence > final_evidence:
			evidence_68 = total_evidence + np.log(0.6825)
			sigma_index = np.argmin(abs(accumulated_evidence - evidence_68))
			results = np.zeros([num_param, 3])

			for param in range(num_param):
				results[param][0] = np.amin(final_array[param + 1][:sigma_index + 1])
				results[param][1] = np.median(final_array[param + 1])
				results[param][2] = np.amax(final_array[param + 1][:sigma_index + 1])
			final_results = results
			final_evidence = total_evidence

	print('Preliminary results:')
	print(final_results)
	print('Evidence = ' + str(final_evidence))
	return (final_results, final_evidence)
# initial fit using mpfit
def p0gen(vel_range = None, num_gauss = None, modified_data = None, accepted = [], last_accepted = [], nwalkers = None, use_molex = True): # returns p0
	if use_molex:
		num_full_param = 12
	else:
		if modified_data['Texp_spectrum']['1665'] != []:
			num_full_param = 10
		else:
			num_full_param = 6
	
	# if using molex, (last) accepted will be 'x', otherwise 'params'
	if use_molex:
		[accepted_x, last_accepted_x] = [accepted, last_accepted]
	else:
		[accepted_params, last_accepted_params] = [accepted, last_accepted]
	
	# find params
	if accepted != [] and use_molex:
		ndim = np.sum([type(a) == bool and a == True for a in modified_data['parameter_list']]) + 1
		accepted_x = [x[1] for x in accepted]
		accepted_params = molex(x = accepted_x, modified_data = modified_data, num_gauss = int(len(accepted_x) / ndim))
	elif accepted == []:
		accepted_params = []
	else:
		accepted_params = [x[1] for x in accepted]
	
	if last_accepted != [] and use_molex:
		ndim = np.sum([type(a) == bool and a == True for a in modified_data['parameter_list']]) + 1
		last_accepted_x = [x[1] for x in last_accepted]
		last_accepted_params = molex(x = last_accepted_x, modified_data = modified_data, num_gauss = int(len(last_accepted_x) / ndim))
	elif last_accepted == []:
		last_accepted_params = []
	else:
		last_accepted_params = [x[1] for x in last_accepted]
	
	# get velocity combos
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

	# set parinfo
	if use_molex:
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
	
	else:
		(tau_1612_range, tau_1665_range, tau_1667_range, tau_1720_range) = ([-5 * np.abs(np.amin(modified_data['tau_spectrum']['1612'])), 5 * np.abs(np.amax(modified_data['tau_spectrum']['1612']))], [-5 * np.abs(np.amin(modified_data['tau_spectrum']['1665'])), 5 * np.abs(np.amax(modified_data['tau_spectrum']['1665']))], [-5 * np.abs(np.amin(modified_data['tau_spectrum']['1667'])), 5 * np.abs(np.amax(modified_data['tau_spectrum']['1667']))], [-5 * np.abs(np.amin(modified_data['tau_spectrum']['1720'])), 5 * np.abs(np.amax(modified_data['tau_spectrum']['1720']))]) 
		standard_parinfo = [{'parname': 'vel', 'fixed': False, 'step': 1.e-1, 'limited': [1,1], 'limits': vel_range}, 
						{'parname': 'FWHM', 'fixed': False, 'step': 1.e-1, 'limited': [1,1], 'limits': FWHM_range},
						{'parname': 'tau_1612', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': tau_1612_range}, 
						{'parname': 'tau_1665', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': tau_1665_range},
						{'parname': 'tau_1667', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': tau_1667_range},
						{'parname': 'tau_1720', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': tau_1720_range}]
		guess_base = [np.mean(vel_range), np.mean(FWHM_range), np.mean(tau_1612_range), np.mean(tau_1665_range), np.mean(tau_1667_range), np.mean(tau_1720_range)]
		if modified_data['Texp_spectrum']['1665'] != []:
			(Texp_1612_range, Texp_1665_range, Texp_1667_range, Texp_1720_range) = ([-5 * np.abs(np.amin(modified_data['Texp_spectrum']['1612'])), 5 * np.abs(np.amax(modified_data['Texp_spectrum']['1612']))], [-5 * np.abs(np.amin(modified_data['Texp_spectrum']['1665'])), 5 * np.abs(np.amax(modified_data['Texp_spectrum']['1665']))], [-5 * np.abs(np.amin(modified_data['Texp_spectrum']['1667'])), 5 * np.abs(np.amax(modified_data['Texp_spectrum']['1667']))], [-5 * np.abs(np.amin(modified_data['Texp_spectrum']['1720'])), 5 * np.abs(np.amax(modified_data['Texp_spectrum']['1720']))]) 
			standard_parinfo += [{'parname': 'Texp_1612', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': Texp_1612_range}, 
						{'parname': 'Texp_1665', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': Texp_1665_range},
						{'parname': 'Texp_1667', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': Texp_1667_range},
						{'parname': 'Texp_1720', 'fixed': False, 'step': 1.e-2, 'limited': [1,1], 'limits': Texp_1720_range}]
			guess_base += [np.mean(Texp_1612_range), np.mean(Texp_1665_range), np.mean(Texp_1667_range), np.mean(Texp_1720_range)]

	guess = guess_base * num_gauss
	parinfo = standard_parinfo * num_gauss
	[best_p, best_params, best_llh] = [[], [], -np.inf]

	if use_molex:
		full_param_list = ([True] + modified_data['parameter_list']) * num_gauss
		for a in range(len(parinfo)):
			if type(full_param_list[a]) != bool:
				parinfo[a]['fixed'] = True

	for vel_combo in vel_combos:
		for vel_ind in range(len(vel_combo)):
			guess[int(vel_ind * num_full_param)] = vel_combo[vel_ind]
		if use_molex:
			for c in range(len(full_param_list)):
				if type(full_param_list[c]) != bool:
					guess[c] = full_param_list[c]
		
		fa = {'modified_data': modified_data, 'vel_range': vel_range, 'num_gauss': num_gauss, 'accepted_params': accepted_params, 'use_molex': use_molex}
		mp = mpfit(mpfitp0, guess, parinfo = parinfo, functkw = fa, maxiter = 1000, quiet = True)
		if use_molex:
			fitted_p = mp.params
			llh = lnprob(	modified_data = modified_data, 
						p = fitted_p, 
						vel_range = vel_range, 
						num_gauss = num_gauss, 
						accepted_params = accepted_params, 
						use_molex = use_molex)
			if llh > best_llh:
				[best_p, best_llh] = [fitted_p, llh]
		else:
			fitted_params = mp.params
			llh = lnprob(	modified_data = modified_data, 
						x = fitted_params, 
						vel_range = vel_range, 
						num_gauss = num_gauss, 
						accepted_params = accepted_params, 
						use_molex = use_molex)
			if llh > best_llh:
				[best_params, best_llh] = [fitted_params, llh]

	if use_molex:
		if best_p != []:
			p0_0 = xlist(p = best_p, data = modified_data, num_gauss = num_gauss)
		else:
			p0_0 = [np.random.uniform(vel_range[0], vel_range[1]), 
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
	else:
		if best_params != []:
			p0_0 = best_params
		else:
			p0_0 = [np.random.uniform(vel_range[0], vel_range[1]), 
					np.random.uniform(FWHM_range[0], FWHM_range[1]), 
					np.random.uniform(tau_1612_range[0], tau_1612_range[1]), 
					np.random.uniform(tau_1665_range[0], tau_1665_range[1]), 
					np.random.uniform(tau_1667_range[0], tau_1667_range[1]), 
					np.random.uniform(tau_1720_range[0], tau_1720_range[1])]					
			if modified_data['Texp_spectrum']['1665'] != []:
				p0_0 += [np.random.uniform(Texp_1612_range[0], Texp_1612_range[1]), 
					np.random.uniform(Texp_1665_range[0], Texp_1665_range[1]), 
					np.random.uniform(Texp_1667_range[0], Texp_1667_range[1]), 
					np.random.uniform(Texp_1720_range[0], Texp_1720_range[1])]
			p0_0 = p0_0 * num_gauss		
	
	p0 = [[a * np.random.uniform(0.999, 1.001) for a in p0_0] for b in range(nwalkers)]
	return p0
def mpfitp0(p = None, fjac = None, modified_data = None, vel_range = None, num_gauss = None, accepted_params = [], use_molex = True): # returns [0, residuals]
	if use_molex:
		params = molex(p = p, modified_data = modified_data, num_gauss = num_gauss)
	else:
		params = p
	if modified_data['Texp_spectrum']['1665'] != []:
		(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = makemodel(params = params, modified_data = modified_data, accepted_params = accepted_params, num_gauss = num_gauss)
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
		(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720) = makemodel(params = params, modified_data = modified_data, accepted_params = accepted_params, num_gauss = num_gauss)
		residuals = np.concatenate((
					(tau_m_1612 - modified_data['tau_spectrum']['1612']) / modified_data['tau_rms']['1612'], 
					(tau_m_1665 - modified_data['tau_spectrum']['1665']) / modified_data['tau_rms']['1665'], 
					(tau_m_1667 - modified_data['tau_spectrum']['1667']) / modified_data['tau_rms']['1667'], 
					(tau_m_1720 - modified_data['tau_spectrum']['1720']) / modified_data['tau_rms']['1720']))
	return [0, residuals]
# converts between x <--> p --> params
def molex(p = [], x = None, modified_data = None, num_gauss = 1, calc_Texp = True): # returns params
	'''
	If calc_Texp == False, Tex will be returned instead of Texp. Use with caution.
	'''
	if p == []:
		p = plist(x, modified_data, num_gauss)

	output = list(np.zeros(int(10 * num_gauss)))


	for gaussian in range(int(num_gauss)):

		[vel, logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd] = p[int(gaussian * 12):int((gaussian + 1) * 12)]

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
		
		if calc_Texp:
			Texp_1612 = Texp(tau = tau0_1612, Tbg = modified_data['Tbg']['1612'], Tex = Tex_1612)
			Texp_1665 = Texp(tau = tau0_1665, Tbg = modified_data['Tbg']['1665'], Tex = Tex_1665)
			Texp_1667 = Texp(tau = tau0_1667, Tbg = modified_data['Tbg']['1667'], Tex = Tex_1667)
			Texp_1720 = Texp(tau = tau0_1720, Tbg = modified_data['Tbg']['1720'], Tex = Tex_1720)
			output[int(10 * gaussian):int(10 * (gaussian + 1))] = [vel, FWHM, tau0_1612, tau0_1665, tau0_1667, tau0_1720, Texp_1612, Texp_1665, Texp_1667, Texp_1720]
		else:
			output[int(10 * gaussian):int(10 * (gaussian + 1))] = [vel, FWHM, tau0_1612, tau0_1665, tau0_1667, tau0_1720, Tex_1612, Tex_1665, Tex_1667, Tex_1720]

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
# converts between tau, Tex, Tbg --> Texp
def Texp(tau = None, Tbg = None, Tex = None): # returns Texp
	Texp = (Tex - Tbg) * (1 - np.exp(-tau))
	return Texp
# makes/plots model from params
def makemodel(params = None, modified_data = None, accepted_params = [], num_gauss = None): # returns tau and Texp models
	vel_1612 = modified_data['vel_axis']['1612']
	vel_1665 = modified_data['vel_axis']['1665']
	vel_1667 = modified_data['vel_axis']['1667']
	vel_1720 = modified_data['vel_axis']['1720']

	num_params = int(len(params) / num_gauss)

	if accepted_params != []:
		if modified_data['Texp_spectrum']['1665'] != []:
			(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = makemodel(params = accepted_params, modified_data = modified_data, num_gauss = int(len(accepted_params) / 10))
		else:
			(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720) = makemodel(params = accepted_params, modified_data = modified_data, num_gauss = int(len(accepted_params) / 6))

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
		
	for comp in range(int(num_gauss)): 
		if modified_data['Texp_spectrum']['1665'] != []:
			[vel, FWHM, tau_1612, tau_1665, tau_1667, tau_1720, Texp_1612, Texp_1665, Texp_1667, Texp_1720] = params[comp * num_params:(comp + 1) * num_params]
		else:
			[vel, FWHM, tau_1612, tau_1665, tau_1667, tau_1720] = params[comp * num_params:(comp + 1) * num_params]
		tau_m_1612 += gaussian(mean = vel, FWHM = FWHM, height = tau_1612)(np.array(vel_1612))
		tau_m_1665 += gaussian(mean = vel, FWHM = FWHM, height = tau_1665)(np.array(vel_1665))
		tau_m_1667 += gaussian(mean = vel, FWHM = FWHM, height = tau_1667)(np.array(vel_1667))
		tau_m_1720 += gaussian(mean = vel, FWHM = FWHM, height = tau_1720)(np.array(vel_1720))

		if modified_data['Texp_spectrum']['1665'] != []:
			Texp_m_1612 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1612)(np.array(vel_1612))
			Texp_m_1665 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1665)(np.array(vel_1665))
			Texp_m_1667 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1667)(np.array(vel_1667))
			Texp_m_1720 += gaussian(mean = vel, FWHM = FWHM, height = Texp_1720)(np.array(vel_1720))
	if modified_data['Texp_spectrum']['1665'] != []:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720)	
	else:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720)
def gaussian(mean = None, FWHM = None, height = None, sigma = None, amp = None): # returns lambda gaussian
	'''
	Generates a gaussian profile with the given parameters.
	'''
	if sigma == None:
		sigma = FWHM / (2. * math.sqrt(2. * np.log(2.)))

	if height == None:
		height = amp / (sigma * math.sqrt(2.* math.pi))
	return lambda x: height * np.exp(-((x - mean)**2.) / (2.*sigma**2.))
# def plotmodel(data = None, x = None, p = [], params = None, accepted_params = [], num_gauss = None, plot_num = None):
# 	'''
# 	Still needs axis labels!
# 	'''
# 	if params == None:
# 		params = molex(p = p, x = x, modified_data = data, num_gauss = num_gauss)
# 	if data['Texp_spectrum']['1665'] != []:
# 		(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720) = makemodel(params = params, modified_data = data, accepted_params = accepted_params, num_gauss = num_gauss)
# 	else:
# 		(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720) = makemodel(params = params, modified_data = data, accepted_params = accepted_params, num_gauss = num_gauss)
# 	fig, axes = plt.subplots(nrows = 5, ncols = 2, sharex = True)

# 	axes[0,0].plot(data['vel_axis']['1612'], data['tau_spectrum']['1612'], color = 'blue', label = '1612 MHz', linewidth = 1)
# 	axes[0,0].plot(data['vel_axis']['1612'], tau_m_1612, color = 'black', linewidth = 1)
# 	axes[1,0].plot(data['vel_axis']['1665'], data['tau_spectrum']['1665'], color = 'green', label = '1665 MHz', linewidth = 1)
# 	axes[1,0].plot(data['vel_axis']['1665'], tau_m_1665, color = 'black', linewidth = 1)
# 	axes[2,0].plot(data['vel_axis']['1667'], data['tau_spectrum']['1667'], color = 'red', label = '1667 MHz', linewidth = 1)
# 	axes[2,0].plot(data['vel_axis']['1667'], tau_m_1667, color = 'black', linewidth = 1)
# 	axes[3,0].plot(data['vel_axis']['1720'], data['tau_spectrum']['1720'], color = 'cyan', label = '1720 MHz', linewidth = 1)
# 	axes[3,0].plot(data['vel_axis']['1720'], tau_m_1720, color = 'black', linewidth = 1)
# 	axes[4,0].plot(data['vel_axis']['1612'], data['tau_spectrum']['1612'] - tau_m_1612, color = 'blue', linewidth = 1)
# 	axes[4,0].plot(data['vel_axis']['1665'], data['tau_spectrum']['1665'] - tau_m_1665, color = 'green', linewidth = 1)
# 	axes[4,0].plot(data['vel_axis']['1667'], data['tau_spectrum']['1667'] - tau_m_1667, color = 'red', linewidth = 1)
# 	axes[4,0].plot(data['vel_axis']['1720'], data['tau_spectrum']['1720'] - tau_m_1720, color = 'cyan', linewidth = 1)
# 	if data['Texp_spectrum']['1665'] != []:
# 		axes[0,1].plot(data['vel_axis']['1612'], Texp_m_1612, color = 'black', linewidth = 1)
# 		axes[1,1].plot(data['vel_axis']['1665'], Texp_m_1665, color = 'black', linewidth = 1)
# 		axes[2,1].plot(data['vel_axis']['1667'], Texp_m_1667, color = 'black', linewidth = 1)
# 		axes[3,1].plot(data['vel_axis']['1720'], Texp_m_1720, color = 'black', linewidth = 1)
# 		axes[0,1].plot(data['vel_axis']['1612'], data['Texp_spectrum']['1612'], color = 'blue', label = '1612 MHz', linewidth = 1)
# 		axes[1,1].plot(data['vel_axis']['1665'], data['Texp_spectrum']['1665'], color = 'green', label = '1665 MHz', linewidth = 1)
# 		axes[2,1].plot(data['vel_axis']['1667'], data['Texp_spectrum']['1667'], color = 'red', label = '1667 MHz', linewidth = 1)
# 		axes[3,1].plot(data['vel_axis']['1720'], data['Texp_spectrum']['1720'], color = 'cyan', label = '1720 MHz', linewidth = 1)
# 		axes[4,1].plot(data['vel_axis']['1612'], data['Texp_spectrum']['1612'] - Texp_m_1612, color = 'blue', linewidth = 1)
# 		axes[4,1].plot(data['vel_axis']['1665'], data['Texp_spectrum']['1665'] - Texp_m_1665, color = 'green', linewidth = 1)
# 		axes[4,1].plot(data['vel_axis']['1667'], data['Texp_spectrum']['1667'] - Texp_m_1667, color = 'red', linewidth = 1)
# 		axes[4,1].plot(data['vel_axis']['1720'], data['Texp_spectrum']['1720'] - Texp_m_1720, color = 'cyan', linewidth = 1)

	# plt.show()
	# plt.savefig('Plots/' + data['source_name'] + '_plot_' + str(plot_num) + '.pdf')
	# plt.close()
# sample posterior using emcee
def sampleposterior(modified_data = None, num_gauss = None, p0 = None, vel_range = None, accepted = [], nwalkers = None, use_molex = True, a = None): # returns (chain, lnprob)
	if use_molex:
		ndim = num_gauss
		for a in modified_data['parameter_list']:
			if type(a) == bool and a == True:
				ndim += num_gauss
			
	elif modified_data['Texp_spectrum']['1665'] != []:
		ndim = 10 * num_gauss
	else:
		ndim = 6 * num_gauss

	num_param_per_gauss = ndim / num_gauss

	if use_molex and accepted != []:
		accepted_x = [x[1] for x in accepted]
		accepted_params = molex(x = accepted_x, modified_data = modified_data, num_gauss = len(accepted_x) / num_param_per_gauss)
	elif accepted != []:
		accepted_params = []
	else:
		accepted_params = [x[1] for x in accepted]

	burn_iterations = 2000
	final_iterations = 1000
	
	args = [modified_data, [], vel_range, num_gauss, accepted_params, use_molex]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args, a = a)
	# burn
	[burn_run, test_result] = [0, 'Fail']
	while burn_run <= 5 and test_result == 'Fail':
		try:
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		except ValueError: # sometimes there is an error within emcee (random.randint)
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		# test convergence
		(test_result, p0) = convergencetest(sampler_chain = sampler.chain, num_gauss = num_gauss, pos = pos)
		burn_run += 1
	# plotchain(modified_data = modified_data, chain = sampler.chain, phase = 'Burn')
	# final run
	sampler.reset()
	sampler.run_mcmc(pos, final_iterations)
	# remove steps where lnprob = -np.inf
	chain = [[sampler.chain[walker][step] for step in range(len(sampler.chain[walker])) if sampler.lnprobability[walker][step] != -np.inf] for walker in range(len(sampler.chain))]
	if np.array(chain).shape[1] == 0:
		chain, lnprob_ = [], []
	else:
		lnprob_ = [[a for a in sampler.lnprobability[walker] if a != -np.inf] for walker in range(len(sampler.lnprobability))]
	return (np.array(chain), np.array(lnprob_))
def lnprob(x = None, modified_data = None, p = [], vel_range = None, num_gauss = None, accepted_params = [], use_molex = True): # returns lnprob
	'''
	need to add a test of whether or not the velocities are in order
	'''
	if use_molex:
		if p == []:
			p = plist(x, modified_data, num_gauss)
		prior = lnprprior(modified_data = modified_data, p = p, vel_range = vel_range, num_gauss = num_gauss)
		# if prior != -np.inf:
		params = molex(p = p, modified_data = modified_data, num_gauss = num_gauss)
	else:
		prior = lnprprior(modified_data = modified_data, params = x, vel_range = vel_range, num_gauss = num_gauss, use_molex = False)
		params = x
	models = makemodel(params = params, modified_data = modified_data, accepted_params = accepted_params, num_gauss = num_gauss)
	
	if modified_data['Texp_spectrum']['1665'] != []:
		[tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720] = models
		spectra = [	modified_data['tau_spectrum']['1612'], modified_data['tau_spectrum']['1665'], 
					modified_data['tau_spectrum']['1667'], modified_data['tau_spectrum']['1720'], 
					modified_data['Texp_spectrum']['1612'], modified_data['Texp_spectrum']['1665'], 
					modified_data['Texp_spectrum']['1667'], modified_data['Texp_spectrum']['1720']]
		rms = [		modified_data['tau_rms']['1612'], modified_data['tau_rms']['1665'], 
					modified_data['tau_rms']['1667'], modified_data['tau_rms']['1720'], 
					modified_data['Texp_rms']['1612'], modified_data['Texp_rms']['1665'], 
					modified_data['Texp_rms']['1667'], modified_data['Texp_rms']['1720']]
	else:
		[tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720] = models
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
def lnprprior(modified_data = None, p = [], params = [], vel_range = None, num_gauss = None, use_molex = True): # returns lnprprior
	if use_molex:
		lnprprior = 0
		parameter_list = [True] + modified_data['parameter_list'] # add velocity
		vel_prev = vel_range[0]
		for gauss in range(num_gauss):
			[vel, logTgas, lognH2, logNOH, fortho, FWHM, Av, logxOH, logxHe, logxe, logTdint, logTd] = p[int(gauss * 12):int((gauss + 1) * 12)]
			if vel >= vel_prev and vel <= vel_range[1]:
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

				priors = [vel_prior, logTgas_prior, lognH2_prior, logNOH_prior, fortho_prior, 
						FWHM_prior, Av_prior, logxOH_prior, logxHe_prior, logxe_prior, logTdint_prior, logTd_prior]
				priors = [priors[a] for a in range(len(priors)) if type(parameter_list[a]) == bool and parameter_list[a] == True]
				lnprprior = lnprprior + np.sum(priors)
				vel_prev = vel
			else:
				return -np.inf
	else:
		lnprprior = lnpriortausum(params = params, num_gauss = num_gauss, modified_data = modified_data)
		vel_prev = vel_range[0]
		for gauss in range(num_gauss):
			if modified_data['Texp_spectrum']['1665'] != []:
				[vel, FWHM, tau_1612, tau_1665, tau_1667, tau_1720, Texp_1612, Texp_1665, Texp_1667, Texp_1720] = params[int(gauss * 10):int((gauss + 1) * 10)]
			else:
				[vel, FWHM, tau_1612, tau_1665, tau_1667, tau_1720] = params[int(gauss * 6):int((gauss + 1) * 6)]
			if vel >= vel_prev and vel <= vel_range[1]:
				vel_prior = lnnaiveprior(value = vel, value_range = [vel_prev, vel_range[1]])
				FWHM_prior = lnnaiveprior(value = FWHM, value_range = FWHM_range)
				(tau_1612_range, tau_1665_range, tau_1667_range, tau_1720_range) = (
					[-5 * np.abs(np.amin(modified_data['tau_spectrum']['1612'])), 5 * np.abs(np.amax(modified_data['tau_spectrum']['1612']))], 
					[-1.5 * np.abs(np.amin(modified_data['tau_spectrum']['1665'])), 1.5 * np.abs(np.amax(modified_data['tau_spectrum']['1665']))], 
					[-1.5 * np.abs(np.amin(modified_data['tau_spectrum']['1667'])), 1.5 * np.abs(np.amax(modified_data['tau_spectrum']['1667']))], 
					[-5 * np.abs(np.amin(modified_data['tau_spectrum']['1720'])), 5 * np.abs(np.amax(modified_data['tau_spectrum']['1720']))])
				tau_1612_prior = lnnaiveprior(value = tau_1612, value_range = tau_1612_range)
				tau_1665_prior = lnnaiveprior(value = tau_1665, value_range = tau_1665_range)
				tau_1667_prior = lnnaiveprior(value = tau_1667, value_range = tau_1667_range)
				tau_1720_prior = lnnaiveprior(value = tau_1720, value_range = tau_1720_range)
				lnprprior += vel_prior + FWHM_prior + tau_1612_prior + tau_1665_prior + tau_1667_prior + tau_1720_prior

				if modified_data['Texp_spectrum']['1665'] != []:
					(Texp_1612_range, Texp_1665_range, Texp_1667_range, Texp_1720_range) = (
						[-5 * np.abs(np.amin(modified_data['Texp_spectrum']['1612'])), 5 * np.abs(np.amax(modified_data['Texp_spectrum']['1612']))], 
						[-1.5 * np.abs(np.amin(modified_data['Texp_spectrum']['1665'])), 1.5 * np.abs(np.amax(modified_data['Texp_spectrum']['1665']))], 
						[-1.5 * np.abs(np.amin(modified_data['Texp_spectrum']['1667'])), 1.5 * np.abs(np.amax(modified_data['Texp_spectrum']['1667']))], 
						[-5 * np.abs(np.amin(modified_data['Texp_spectrum']['1720'])), 5 * np.abs(np.amax(modified_data['Texp_spectrum']['1720']))]) 
					Texp_1612_prior = lnnaiveprior(value = Texp_1612, value_range = Texp_1612_range)
					Texp_1665_prior = lnnaiveprior(value = Texp_1665, value_range = Texp_1665_range)
					Texp_1667_prior = lnnaiveprior(value = Texp_1667, value_range = Texp_1667_range)
					Texp_1720_prior = lnnaiveprior(value = Texp_1720, value_range = Texp_1720_range)
					lnprprior += Texp_1612_prior + Texp_1665_prior + Texp_1667_prior + Texp_1720_prior
			else:
				return -np.inf
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
# def plotchain(modified_data = None, chain = None, phase = None):
# 	ndim = chain.shape[2]
# 	for parameter in range(ndim):
# 		# plt.figure()
		# for walker in range(chain.shape[0]):
		# 	plt.plot(range(chain.shape[1]), chain[walker,:,parameter])
		# plt.title(modified_data['source_name'] + ' for param ' + str(parameter) + ': ' + str(phase))
		# plt.show()
		# plt.savefig('Plots/Chain_plot_' + str(phase) + '_' + modified_data['source_name'] + '_' + str(parameter) + '.pdf')
		# plt.close()
def splitwalkers(chain, lnprob, tolerance = 10):
	'''
	Note: This may not be needed if I increase 'a' in the EnsembleSampler (a relates to step size or acceptance rate or similar)
	'''
	lnprob = np.array([[[x] for x in y] for y in lnprob])
	chain = np.array(chain)
	
	num_params = chain.shape[2]
	num_steps = chain.shape[1]
	num_walkers = chain.shape[0]

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
		return -np.log(value_range[1] - value_range[0])
	else:
		return -np.inf
def lnpriortausum(params = None, num_gauss = None, modified_data = None): # returns lnprior for tau sum rule
	lnprior = 0
	num_params = int(len(params) / num_gauss)
	[tau_rms_1612, tau_rms_1665, tau_rms_1667, tau_rms_1720] = [modified_data['tau_rms']['1612'], modified_data['tau_rms']['1665'], modified_data['tau_rms']['1667'], modified_data['tau_rms']['1720']]
	for gauss in range(num_gauss):
		[tau_1612, tau_1665, tau_1667, tau_1720] = params[num_params * gauss + 2:num_params * gauss + 6]
		tau_sum = tau_1665/5. + tau_1667/9. - tau_1612 - tau_1720
		tau_sum_rms = np.sqrt((tau_rms_1665/5.)**2. + (tau_rms_1667/9.)**2. + (tau_rms_1612)**2. + (tau_rms_1720)**2.)
		lnprior += -np.log(tau_sum_rms * np.sqrt(2.*math.pi)) - (tau_sum**2) / (2.*tau_sum_rms**2)
	return lnprior
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
def resultstable(final_parameters = None, data = None):
	'''
	make a latex table
	'''

	if len(final_parameters) > 5:
		if data['Texp_spectrum']['1665'] != []:
			for feature in range(int(len(final_parameters) / 10)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84], [Texp_1612_16, Texp_1612_50, Texp_1612_84], [Texp_1665_16, Texp_1665_50, Texp_1665_84], [Texp_1667_16, Texp_1667_50, Texp_1667_84], [Texp_1720_16, Texp_1720_50, Texp_1720_84]] = final_parameters[feature * 10:feature * 10 + 10]
				[vel_16, vel_50, vel_84] = [round(x, 1) for x in [vel_16, vel_50, vel_84]]
				[fwhm_16, fwhm_50, fwhm_84] = [round(x, 2) for x in [fwhm_16, fwhm_50, fwhm_84]]
				[tau_1612_16, tau_1612_50, tau_1612_84, tau_1665_16, tau_1665_50, tau_1665_84, tau_1667_16, tau_1667_50, tau_1667_84, tau_1720_16, tau_1720_50, tau_1720_84, Texp_1612_16, Texp_1612_50, Texp_1612_84, Texp_1665_16, Texp_1665_50, Texp_1665_84, Texp_1667_16, Texp_1667_50, Texp_1667_84, Texp_1720_16, Texp_1720_50, Texp_1720_84] = [round(x, 3) for x in [tau_1612_16, tau_1612_50, tau_1612_84, tau_1665_16, tau_1665_50, tau_1665_84, tau_1667_16, tau_1667_50, tau_1667_84, tau_1720_16, tau_1720_50, tau_1720_84, Texp_1612_16, Texp_1612_50, Texp_1612_84, Texp_1665_16, Texp_1665_50, Texp_1665_84, Texp_1667_16, Texp_1667_50, Texp_1667_84, Texp_1720_16, Texp_1720_50, Texp_1720_84]]
				print(data['source_name'] + '&' + str(vel_50) + '$^{+' + str(np.abs(vel_84 - vel_50)) + '}_{-' + str(np.abs(vel_50 - vel_16)) + '}$' + '&' + 
						str(fwhm_50) + '$^{+' + str(np.abs(fwhm_84 - fwhm_50)) + '}_{-' + str(np.abs(fwhm_50 - fwhm_16)) + '}$' + '&' + 
						str(tau_1612_50) + '$^{+' + str(np.abs(tau_1612_84 - tau_1612_50)) + '}_{-' + str(np.abs(tau_1612_50 - tau_1612_16)) + '}$' + '&' + 
						str(tau_1665_50) + '$^{+' + str(np.abs(tau_1665_84 - tau_1665_50)) + '}_{-' + str(np.abs(tau_1665_50 - tau_1665_16)) + '}$' + '&' + 
						str(tau_1667_50) + '$^{+' + str(np.abs(tau_1667_84 - tau_1667_50)) + '}_{-' + str(np.abs(tau_1667_50 - tau_1667_16)) + '}$' + '&' + 
						str(tau_1720_50) + '$^{+' + str(np.abs(tau_1720_84 - tau_1720_50)) + '}_{-' + str(np.abs(tau_1720_50 - tau_1720_16)) + '}$' + '&' + 
						str(Texp_1612_50) + '$^{+' + str(np.abs(Texp_1612_84 - Texp_1612_50)) + '}_{-' + str(np.abs(Texp_1612_50 - Texp_1612_16)) + '}$' + '&' + 
						str(Texp_1665_50) + '$^{+' + str(np.abs(Texp_1665_84 - Texp_1665_50)) + '}_{-' + str(np.abs(Texp_1665_50 - Texp_1665_16)) + '}$' + '&' + 
						str(Texp_1667_50) + '$^{+' + str(np.abs(Texp_1667_84 - Texp_1667_50)) + '}_{-' + str(np.abs(Texp_1667_50 - Texp_1667_16)) + '}$' + '&' + 
						str(Texp_1720_50) + '$^{+' + str(np.abs(Texp_1720_84 - Texp_1720_50)) + '}_{-' + str(np.abs(Texp_1720_50 - Texp_1720_16)) + '}$' + '\\\\')

		else:
			for feature in range(int(len(final_parameters) / 6)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84]] = final_parameters[feature * 6:feature * 6 + 6]
				[[tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84]] = [[tau_1612_16*1000, tau_1612_50*1000, tau_1612_84*1000], [tau_1665_16*1000, tau_1665_50*1000, tau_1665_84*1000], [tau_1667_16*1000, tau_1667_50*1000, tau_1667_84*1000], [tau_1720_16*1000, tau_1720_50*1000, tau_1720_84*1000]]
				[velm, velp] = [vel_50 - vel_16, vel_84 - vel_50]
				[fwhmm, fwhmp] = [fwhm_50 - fwhm_16, fwhm_84 - fwhm_50]
				[tau_1612m, tau_1612p] = [tau_1612_50 - tau_1612_16, tau_1612_84 - tau_1612_50]
				[tau_1665m, tau_1665p] = [tau_1665_50 - tau_1665_16, tau_1665_84 - tau_1665_50]
				[tau_1667m, tau_1667p] = [tau_1667_50 - tau_1667_16, tau_1667_84 - tau_1667_50]
				[tau_1720m, tau_1720p] = [tau_1720_50 - tau_1720_16, tau_1720_84 - tau_1720_50]
				[velm, vel_50, velp] = [round(x, 1) for x in [velm, vel_50, velp]]
				[fwhmm, fwhm_50, fwhmp] = [round(x, 2) for x in [fwhmm, fwhm_50, fwhmp]]
				[tau_1612m, tau_1612_50, tau_1612p, tau_1665m, tau_1665_50, tau_1665p, tau_1667m, tau_1667_50, tau_1667p, tau_1720m, tau_1720_50, tau_1720p] = [round(x, 0) for x in [tau_1612m, tau_1612_50, tau_1612p, tau_1665m, tau_1665_50, tau_1665p, tau_1667m, tau_1667_50, tau_1667p, tau_1720m, tau_1720_50, tau_1720p]]
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
# def plotfinalmodel(final_parameters = None, data = None, file_preamble = None):
# 	'''
# 	final_parameters is an Xx3 array, axis 1 has the 0.16, 0.50 and 0.84 quantiles from the posterior distribution, which approximate +/- 1 sigma
# 	THIS ONLY WORKS FOR NON-MOLEX!!
# 	'''

# 	parameters_16 = [x[0] for x in final_parameters]
# 	parameters_50 = [x[1] for x in final_parameters]
# 	parameters_84 = [x[2] for x in final_parameters]



# 	source_name = data['source_name']

# 	vel_1612 = data['vel_axis']['1612']
# 	vel_1665 = data['vel_axis']['1665']
# 	vel_1667 = data['vel_axis']['1667']
# 	vel_1720 = data['vel_axis']['1720']

# 	tau_1612 = data['tau_spectrum']['1612']
# 	tau_1665 = data['tau_spectrum']['1665']
# 	tau_1667 = data['tau_spectrum']['1667']
# 	tau_1720 = data['tau_spectrum']['1720']
	
# 	if data['Texp_spectrum']['1665'] != []:
# 		Texp_1612 = data['Texp_spectrum']['1612']
# 		Texp_1665 = data['Texp_spectrum']['1665']
# 		Texp_1667 = data['Texp_spectrum']['1667']
# 		Texp_1720 = data['Texp_spectrum']['1720']
# 		num_gauss = len(parameters_50)/10
		
# 		(tau_model_1612_min, tau_model_1665_min, tau_model_1667_min, tau_model_1720_min, Texp_model_1612_min, Texp_model_1665_min, Texp_model_1667_min, Texp_model_1720_min) = makemodel(params = parameters_16, modified_data = data, num_gauss = num_gauss)
# 		(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = makemodel(params = parameters_50, modified_data = data, num_gauss = num_gauss)
# 		(tau_model_1612_max, tau_model_1665_max, tau_model_1667_max, tau_model_1720_max, Texp_model_1612_max, Texp_model_1665_max, Texp_model_1667_max, Texp_model_1720_max) = makemodel(params = parameters_84, modified_data = data, num_gauss = num_gauss)

# 		fig, axes = plt.subplots(nrows = 5, ncols = 2, sharex = True)
# 		# tau
# 		axes[0,0].plot(vel_1612, tau_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
# 		axes[0,0].plot(vel_1612, tau_model_1612, color = 'black', linewidth = 1)
# 		axes[0,0].fill_between(vel_1612, tau_model_1612_min, tau_model_1612_max, color='0.7', zorder=-1)
# 		axes[1,0].plot(vel_1665, tau_1665, color = 'green', label = '1665 MHz', linewidth = 1)
# 		axes[1,0].plot(vel_1665, tau_model_1665, color = 'black', linewidth = 1)
# 		axes[1,0].fill_between(vel_1665, tau_model_1665_min, tau_model_1665_max, color='0.7', zorder=-1)
# 		axes[2,0].plot(vel_1667, tau_1667, color = 'red', label = '1667 MHz', linewidth = 1)
# 		axes[2,0].plot(vel_1667, tau_model_1667, color = 'black', linewidth = 1)
# 		axes[2,0].fill_between(vel_1667, tau_model_1667_min, tau_model_1667_max, color='0.7', zorder=-1)
# 		axes[3,0].plot(vel_1720, tau_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
# 		axes[3,0].plot(vel_1720, tau_model_1720, color = 'black', linewidth = 1)
# 		axes[3,0].fill_between(vel_1720, tau_model_1720_min, tau_model_1720_max, color='0.7', zorder=-1)
# 		# tau residuals
# 		axes[4,0].plot(vel_1612, tau_1612 - tau_model_1612, color = 'blue', linewidth = 1)
# 		axes[4,0].plot(vel_1665, tau_1665 - tau_model_1665, color = 'green', linewidth = 1)
# 		axes[4,0].plot(vel_1667, tau_1667 - tau_model_1667, color = 'red', linewidth = 1)
# 		axes[4,0].plot(vel_1720, tau_1720 - tau_model_1720, color = 'cyan', linewidth = 1)
# 		# Texp
# 		axes[0,1].plot(vel_1612, Texp_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
# 		axes[0,1].plot(vel_1612, Texp_model_1612, color = 'black', linewidth = 1)
# 		axes[0,1].fill_between(vel_1612, Texp_model_1612_min, Texp_model_1612_max, color='0.7', zorder=-1)
# 		axes[1,1].plot(vel_1665, Texp_1665, color = 'green', label = '1665 MHz', linewidth = 1)
# 		axes[1,1].plot(vel_1665, Texp_model_1665, color = 'black', linewidth = 1)
# 		axes[1,1].fill_between(vel_1665, Texp_model_1665_min, Texp_model_1665_max, color='0.7', zorder=-1)
# 		axes[2,1].plot(vel_1667, Texp_1667, color = 'red', label = '1667 MHz', linewidth = 1)
# 		axes[2,1].plot(vel_1667, Texp_model_1667, color = 'black', linewidth = 1)
# 		axes[2,1].fill_between(vel_1667, Texp_model_1667_min, Texp_model_1667_max, color='0.7', zorder=-1)
# 		axes[3,1].plot(vel_1720, Texp_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
# 		axes[3,1].plot(vel_1720, Texp_model_1720, color = 'black', linewidth = 1)
# 		axes[3,1].fill_between(vel_1720, Texp_model_1720_min, Texp_model_1720_max, color='0.7', zorder=-1)
# 		# Texp residuals
# 		axes[4,1].plot(vel_1612, Texp_1612 - Texp_model_1612, color = 'blue', linewidth = 1)
# 		axes[4,1].plot(vel_1665, Texp_1665 - Texp_model_1665, color = 'green', linewidth = 1)
# 		axes[4,1].plot(vel_1667, Texp_1667 - Texp_model_1667, color = 'red', linewidth = 1)
# 		axes[4,1].plot(vel_1720, Texp_1720 - Texp_model_1720, color = 'cyan', linewidth = 1)

# 		for row in range(5):
# 			for col in range(2):
# 				if any([axes[row, col].get_yticks()[::2][x] == 0. for x in axes[row, col].get_yticks()[::2]]):
# 					axes[row, col].set_yticks(axes[row, col].get_yticks()[::2])
# 				else:
# 					axes[row, col].set_yticks(axes[row, col].get_yticks()[1::2])

# 	else:
# 		num_gauss = len(parameters_50)/6
# 		(tau_model_1612_min, tau_model_1665_min, tau_model_1667_min, tau_model_1720_min) = makemodel(params = parameters_16, modified_data = data, num_gauss = num_gauss)
# 		(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = makemodel(params = parameters_50, modified_data = data, num_gauss = num_gauss)
# 		(tau_model_1612_max, tau_model_1665_max, tau_model_1667_max, tau_model_1720_max) = makemodel(params = parameters_84, modified_data = data, num_gauss = num_gauss)
# 		fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex = True)
# 		# tau
# 		axes[0].plot(vel_1612, tau_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
# 		axes[0].plot(vel_1612, tau_model_1612, color = 'black', linewidth = 1)
# 		axes[0].fill_between(vel_1612, tau_model_1612_min, tau_model_1612_max, color='0.7', zorder=-1)
# 		axes[1].plot(vel_1665, tau_1665, color = 'green', label = '1665 MHz', linewidth = 1)
# 		axes[1].plot(vel_1665, tau_model_1665, color = 'black', linewidth = 1)
# 		axes[1].fill_between(vel_1665, tau_model_1665_min, tau_model_1665_max, color='0.7', zorder=-1)
# 		axes[2].plot(vel_1667, tau_1667, color = 'red', label = '1667 MHz', linewidth = 1)
# 		axes[2].plot(vel_1667, tau_model_1667, color = 'black', linewidth = 1)
# 		axes[2].fill_between(vel_1667, tau_model_1667_min, tau_model_1667_max, color='0.7', zorder=-1)
# 		axes[3].plot(vel_1720, tau_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
# 		axes[3].plot(vel_1720, tau_model_1720, color = 'black', linewidth = 1)
# 		axes[3].fill_between(vel_1720, tau_model_1720_min, tau_model_1720_max, color='0.7', zorder=-1)
# 		# tau residuals
# 		axes[4].plot(vel_1612, tau_1612 - tau_model_1612, color = 'blue', linewidth = 1)
# 		axes[4].plot(vel_1665, tau_1665 - tau_model_1665, color = 'green', linewidth = 1)
# 		axes[4].plot(vel_1667, tau_1667 - tau_model_1667, color = 'red', linewidth = 1)
# 		axes[4].plot(vel_1720, tau_1720 - tau_model_1720, color = 'cyan', linewidth = 1)
# 		# labels
# 		axes[4].set_xlabel('Velocity (km/s)')
# 		axes[2].set_ylabel('Optical Depth', labelpad = 15)
# 		axes[0].set_title(source_name)
# 		axes[0].text(0.01, 0.75, '1612 MHz', transform=axes[0].transAxes)
# 		axes[1].text(0.01, 0.75, '1665 MHz', transform=axes[1].transAxes)
# 		axes[2].text(0.01, 0.75, '1667 MHz', transform=axes[2].transAxes)
# 		axes[3].text(0.01, 0.75, '1720 MHz', transform=axes[3].transAxes)
# 		axes[4].text(0.01, 0.75, 'Residuals', transform=axes[4].transAxes)
		
# 		for row in range(5):
# 			if any([axes[row].get_yticks()[::2][x] == 0. for x in axes[row].get_yticks()[::2]]):
# 				axes[row].set_yticks(axes[row].get_yticks()[::2])
# 			else:
# 				axes[row].set_yticks(axes[row].get_yticks()[1::2])

	
# 	plt.savefig(str(file_preamble) + '_' + source_name + '_Final_model.pdf')
# 	# plt.show()
# 	plt.close()

#############################
#                           #
#   M   M        i          #
#   MM MM   aa      n nn    #
#   M M M   aaa  i  nn  n   #
#   M   M  a  a  i  n   n   #
#   M   M   aaa  i  n   n   #
#                           #
#############################

def main(source_name = None, # prints final_p
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

	# print('starting Main(source_')
	# initialise data dictionary
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
	
	final_p = placegaussians(data, Bayes_threshold, use_molex = use_molex, a = a, test = test)
	print('Final array for ' + str(source_name) + ':')
	print(final_p)