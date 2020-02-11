from mpfit import mpfit
from statistics import mean
import copy
import emcee
import itertools
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from itertools import islice


##################################################
#                                                #
#   Set expected ranges for target environment   #
#                                                #
##################################################

FWHM_range      = [0.1, 15.]


# Variables used throughout:
#
# parameter_list = list of parameters for molex found in dictionary object. Those =True are to be fit
# p = full set of parameters for molex for all gaussians (including vel!)
# x = subset of parameters for molex for all gaussians (the molex parameters we're fitting)
# params = full set of v, FWHM, [tau(x3) or N(x4)] for all gaussians (parameters that relate to the spectra)




'''
To do list before publication:
- change arguments to have 'inputs' and 'keyword' arguments
- explanatory blurbs before each function
- explanations of how to modify priors to include other molecules, sum rule analogs etc.
'''




##############################
#                            #
#           Spectra          #
#                            #
##############################
#              |
#              |
#            \ | /
#             \|/
#              V
def findranges(data, sigma_tolerance, num_chan = 1): 
	'''
	Identifies velocity ranges likely to contain features. Adds 
	'interesting_vel' (velocity channels with maxima/minima/inflection points) 
	and 'sig_vel_ranges' (velocity ranges that satisfy a significance test) to 
	the 'data' dictionary.
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	Keywords:
	num_chan - (int) number of channels over which to bin summed spectra in order to 
	check for significance
	sigma_tolerance - (float) signal to noise ratio threshold for a channel of the 
	'summed spectra' to be identified as 'significant'.
	Returns:
	data - (dictionary) 'data' dictionary with updated 'sig_vel_ranges' keyword
	'''
	data = interestingvel(data)
	if data['interesting_vel'] != []:
		sig_vel_list = data['interesting_vel']
	else:
		sig_vel_list = []

	summed_spectra = sumspectra(data)
	if data['misc'] != None:
		bool_summed = [np.mean(summed_spectra[x:x+num_chan]) >= sigma_tolerance 
			for x in range(len(data['misc'][0][0]) - num_chan)]
		sig_vel_list += [data['misc'][0][0][x] for x in range(len(bool_summed)) 
			if bool_summed[x] == True]
	else:
		bool_summed = [np.mean(summed_spectra[x:x+num_chan]) >= sigma_tolerance 
			for x in range(len(data['vel_axis']['1612']) - num_chan)]
		sig_vel_list += [data['vel_axis']['1612'][x] for x in 
			range(len(bool_summed)) if bool_summed[x] == True]

	# merges closely spaced velocities, groups moderately spaced velocities
	sig_vel_list = reducelist(sig_vel_list)
	sig_vel_ranges = [[x[0], x[-1]] for x in sig_vel_list if x[0] != x[-1]]
	data['sig_vel_ranges'] = sig_vel_ranges
	return data
def interestingvel(data):
	'''
	Identifies the location of potential features (maxima, minima, 
	inflections, etc) and places them in the list 'interesting_vel' in the 
	'data' dictionary.
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	Returns:
	data - (dictionary) 'data' dictionary with updated 'interesting_vel' 
		keyword
	'''
	id_vel_list = []
	# Flag features
	
	if data['misc'] != None:
		vel_axes = [x[0] for x in data['misc']]
		spectra = [x[1] for x in data['misc']]
		spectra_rms = [findrms(x) for x in spectra]

	elif 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		vel_axes = [data['vel_axis']['1612'], data['vel_axis']['1665'], 
					data['vel_axis']['1667'], data['vel_axis']['1720']] * 2
		spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720'], 
					data['Texp_spectrum']['1612'], 
					data['Texp_spectrum']['1665'], 
					data['Texp_spectrum']['1667'], 
					data['Texp_spectrum']['1720']]
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

	dv = np.abs(vel_axes[0][1] - vel_axes[0][0])
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
		
		dx_zero = zeros(vel_axis, dx, y_rms = rms_dx)
		dx2_zero = zeros(vel_axis, dx2, y_rms = rms_dx2)		

		vel_list1 = [vel_axis[z] for z in range(int(len(dx_pos) - 1)) if 
			dx_zero[z] == True and spectrum_pos[z] != dx2_pos[z]]
		vel_list2 = [vel_axis[z] for z in range(1,int(len(dx2_zero) - 1)) if 
			dx2_zero[z] == True and spectrum_zero[z] == False and np.any([x <= 
			dx2_zero[z+1] and x >= dx2_zero[z-1] for x in dx_zero]) == False]
		
		vel_list = np.concatenate((vel_list1, vel_list2))
		
		id_vel_list.append(vel_list)
	
	id_vel_list = sorted([val for sublist in id_vel_list for val in sublist])

	if len(id_vel_list) != 0:
		id_vel_list = reducelist(id_vel_list, 3 * dv, 1) # removes duplicates
		id_vel_list = sorted([val for sublist 
				in id_vel_list for val in sublist])
		data['interesting_vel'] = id_vel_list
		return data
	return data
def derivative(vel_axis, spectrum, _range = 20): # returns dx
	'''
	Calculates a simple numerical derivative of 'spectrum' by fitting a 
	straight line to ranges of '_range' channels in 'vel_axis'.
	Parameters:
	vel_axis - (array-like) x-axis corresponding to 'spectrum' (assumed to be velocity but 
	not necessary)
	spectrum - (array-like) y-axis values
	Keywords:
	_range - (int) number of channels over which to fit a straight line
	Returns:
	dy/dx array (numpy array)
	'''
	extra = [0] * int(_range / 2) # _range will be even
	dx = []

	for start in range(int(len(vel_axis) - _range)):
		x = vel_axis[start:int(start + _range + 1)]
		y = spectrum[start:int(start + _range + 1)]

		guess = [(y[0] - y[-1]) / (x[0] - x[-1]), 0]
		parinfo = [	{'parname':'gradient','step':0.0001, 'limited': [1, 1], 
					'limits': [-20., 20.]}, 
					{'parname':'y intercept','step':0.001, 'limited': [1, 1], 
					'limits': [-1000., 1000.]}]
		fa = {'x': x, 'y': y}
		
		mp = mpfit(mpfitlinear, guess, parinfo = parinfo, functkw = fa, 
			maxiter = 10000, quiet = True)
		gradient = mp.params[0]
		dx.append(gradient)
	dx = np.concatenate((extra, dx, extra))
	return dx
def mpfitlinear(a, fjac, x, y): # returns [0, residuals]
	'''
	x and y should be small arrays of length '_range' (from parent function). 
	'''
	[m, c] = a # gradient and y intercept of line
	model_y = m * np.array(x) + c
	residuals = (np.array(y) - model_y)

	return [0, residuals]
def findrms(spectrum): # returns rms
	'''
	Calculates the root mean square of 'spectrum'. This is intended as a 
	measure of noise only, so rms is calculated for several ranges, and the 
	median is returned.
	Parameters:
	spectrum - (array-like) 1d array of values for which to compute rms
	Returns:
	(float) Median rms of those calculated 
	'''
	x = len(spectrum)
	a = int(x / 10)
	rms_list = []
	for _set in range(9):
		rms = np.std(spectrum[(_set * a):(_set * a) + (2 * a)])
		rms_list.append(rms)
	median_rms = np.median(rms_list)
	return median_rms
def zeros(x_axis, y_axis, y_rms = None):
	'''
	Produces a boolean array of length = len(x_axis) of whether or not an x 
	value is a zero, determined by the behaviour of a linear fit to surrounding 
	points.
	Parameters:
	x_axis - (array-like) 1d array of x values
	y_axis - (array-like) 1d array of y values
	Keywords:
	y_rms - (float) rms noise of y_axis
	Returns:
	(array) Boolean array of length = len(x_axis) of whether or not an x value is a 
	zero.
	'''	
	if y_rms == None:
		y_rms = findrms(y_axis)

	gradient_min = abs(2.* y_rms / (x_axis[10] - x_axis[0]))

	zeros = np.zeros(len(x_axis))
	for x in range(5, int(len(x_axis) - 5)):
		x_axis_subset = x_axis[x-5:x+6]
		y_axis_subset = y_axis[x-5:x+6]

		guess = [1., 1.]
		parinfo = [	{'parname':'gradient','step':0.001}, 
					{'parname':'y intercept','step':0.001}]
		fa = {'x': x_axis_subset, 'y': y_axis_subset}

		mp = mpfit(mpfitlinear, guess, parinfo = parinfo, functkw = fa, 
			maxiter = 10000, quiet = True)
		[grad_fit, y_int_fit] = mp.params

		if abs(grad_fit) >= gradient_min:

			# find y values on either side of x to test sign. True = pos
			if (grad_fit * x_axis[x-1] + y_int_fit > 0 and grad_fit * 
				x_axis[x+1] + y_int_fit < 0):
				zeros[x] = 1
			elif (grad_fit * x_axis[x-1] + y_int_fit < 0 and grad_fit * 
				x_axis[x+1] + y_int_fit > 0):
				zeros[x] = 1
	return zeros
def sumspectra(data):
	'''
	Regrids, normalises and finds the summed root mean square a set of spectra 
	to later identify regions of significance.
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	Returns:
	(numpy array) The summed root mean square of the spectra.
	'''
	if data['misc'] != None:
		vel_axes = np.array([x[0] for x in data['misc']])
		spectra = np.array([x[1] for x in data['misc']])
		spectra_rms = np.array([findrms(x) for x in spectra])
		num_spec = len(data['misc'])
	else:
		vel_axes = [data['vel_axis']['1612'], data['vel_axis']['1665'],
					data['vel_axis']['1667'], data['vel_axis']['1720']]
		spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'],
					data['tau_spectrum']['1667'], data['tau_spectrum']['1720']]
		spectra_rms = [ data['tau_rms']['1612'], data['tau_rms']['1665'], 
						data['tau_rms']['1667'], data['tau_rms']['1720']]
		num_spec = 4
	
	for a in range(num_spec):
		if not np.all(np.diff(vel_axes[a])):
			vel_axes[a], spectra[a] = zip(*sorted(zip(vel_axes[a],spectra[a])))

	regridded_spectra = [np.interp(vel_axes[0], vel_axes[x], spectra[x]) for x 
		in range(len(spectra))]
	norm_spectra = [regridded_spectra[x] / spectra_rms[x] for x in 
		range(len(spectra))]
	sqrd_spectra = [norm_spectra[x]**2 for x in range(len(spectra))]
	summed = np.zeros(len(sqrd_spectra[0]))

	for b in range(num_spec):
		summed += sqrd_spectra[b]

	if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		num_spec = 8
		Texp_spectra = [	data['Texp_spectrum']['1612'], 
							data['Texp_spectrum']['1665'],
							data['Texp_spectrum']['1667'], 
							data['Texp_spectrum']['1720']]
		Texp_spectra_rms = [data['Texp_rms']['1612'], data['Texp_rms']['1665'], 
							data['Texp_rms']['1667'], data['Texp_rms']['1720']]
		for a in range(4):
			if np.all(np.diff(vel_axes[a])):
				vel_axes[a], Texp_spectra[a] = zip(*sorted(zip(vel_axes[a], 
					Texp_spectra[a])))
		
		regridded_Texp_spectra = [np.interp(vel_axes[0], vel_axes[x], 
			Texp_spectra[x]) for x in range(4)]
		norm_Texp_spectra = [regridded_Texp_spectra[x] / Texp_spectra_rms[x] 
			for x in range(4)]	
		sqrd_Texp_spectra = [norm_Texp_spectra[x]**2 for x in range(4)]
		summed += sqrd_Texp_spectra[b]

	root_mean_sum = np.sqrt(summed/num_spec)

	return root_mean_sum
def reducelist(master_list, merge_size = 0.5, 
	group_spacing = 0.5*FWHM_range[1]):
	'''
	Merges values in master_list separated by less than merge_size, and groups 
	features separated by less than group_spacing into blended features. 
	Parameters:
	master_list - list of velocities
	merge_size - any velocities separated by less than 'merge_size' will be 
		merged into one velocity. This is performed in 4 stages, first using 
		merge_size / 4 so that the closest velocities are merged first. Merged 
		velocities are replaced by their mean. 
	group_spacing - any velocities separated by less than 'group_spacing' will 
		be grouped together so they can be fit as blended features. Smaller 
		values are likely to prevent the accurate identification of blended 
		features, while larger values will increase running time.
	Returns: 
	(list) Nested list of velocities = [[v1], [v2, v3], [v4]] where v1 and v4 are 
		isolated features that can be fit independently, but v2 and v3 are close 
		enough in velocity that they must be fit together as a blended feature.
	'''
	try:
		master_list = sorted([val for sublist in master_list for val in 
			sublist])
	except TypeError:
		pass

	master_list = np.array(master_list)
	
	# Step 1: merge based on merge_size
	new_merge_list = np.sort(master_list.flatten())

	for merge in [merge_size / 4, 2 * merge_size / 4, 3 * merge_size / 4, 
		merge_size]:
		new_merge_list = mergefeatures(new_merge_list, merge, 'merge')
	
	# Step 2: identify comps likely to overlap to be fit together
	final_merge_list = mergefeatures(new_merge_list, group_spacing, 'group')

	return final_merge_list
def mergefeatures(master_list, size, action):
	'''
	Does the work for ReduceList
	Parameters:
	master_list - list of velocities generated by AGD()
	size - Distance in km/sec for the given action
	action - Action to perform: 'merge' or 'group'
	Returns: 
	Nested list of velocities = [[v1], [v2, v3], [v4]] where v1 and v4 are 
	isolated features that can be fit independently, but v2 and v3 are close 
	enough in velocity that they must be fit together as a blended feature.
	'''	
	new_merge_list = []
	check = 0
	while check < len(master_list):
		skip = 1
		single = True

		if action == 'merge':
			while (check + skip < len(master_list) and 
				master_list[check + skip] - master_list[check] < size):
				skip += 1
				single = False
			if single == True:
				new_merge_list = np.append(new_merge_list, master_list[check])
			else:
				new_merge_list = np.append(new_merge_list, 
					mean(master_list[check:check + skip]))
			check += skip

		elif action == 'group':
			while (check + skip < len(master_list) and 
				master_list[check + skip] - 
				master_list[check + skip - 1] < size):
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
def placegaussians(data, Bayes_threshold = 10, quiet = False): 
	'''
	Manages the process of placing and evaluating gaussian features. 
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	Keywords:
	Bayes_threshold - (float) Bayes Factor required for a model to be preferred
	quiet - (boolean) Whether or not to print outputs to terminal
	Returns:
	(2D array) Full list of accepted parameters with upper and lower bounds of
		credibile intervals. 
		Format: [[1min, 1median, 1max], [2min, 2median, 2max],...] for 
		parameters 1, 2, etc.
	'''
	accepted_full = []
	total_num_gauss = 0
	plot_num = 0
	for vel_range in data['sig_vel_ranges']:
		last_accepted_full = []
		[min_vel, max_vel] = vel_range
		mod_data = trimdata(data, min_vel, max_vel)
		if 'Texp_spectrum' in mod_data and mod_data['Texp_spectrum']['1665'] != []:
			N_ranges = NrangetauTexp(mod_data)
			mod_data['N_ranges'] = N_ranges
		num_gauss = 1
		keep_going = True
		extra = 0
		null_evidence = nullevidence(mod_data)
		prev_evidence = null_evidence
		evidences = [prev_evidence]
		if not quiet:
			print(data['source_name'] + '\t' + str(vel_range) + '\t' + 
				str(null_evidence))

		while keep_going == True:
			if data['misc'] != None:
				nwalkers = int(20 * len(data['misc']) * num_gauss) 
				# in case 'misc' is big
			else:
				nwalkers = 30 * num_gauss
			p0 = p0gen(vel_range, num_gauss, mod_data, accepted_full, 
				nwalkers)
			(chain, lnprob_) = sampleposterior(mod_data, num_gauss, p0, 
				[min_vel, max_vel], nwalkers, accepted = accepted_full)
			if len(chain) != 0:
				(current_full, current_evidence) = bestparams(chain, lnprob_, 
					quiet = quiet)
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
				nwalkers = 30 * num_gauss
				p0 = p0gen(vel_range, num_gauss, mod_data, accepted_full, 
					nwalkers)
				(chain, lnprob_) = sampleposterior(mod_data, num_gauss, p0, 
					[min_vel, max_vel], nwalkers,  accepted = accepted_full)

				if len(chain) == 0:
					# print('Process failed again. Moving on.')
					keep_going = False
		accepted_full = list(itertools.chain(accepted_full, 
			last_accepted_full))
	return accepted_full
def trimdata(data, min_vel, max_vel):
	'''
	Trims the spectra within the 'data' dictionary to the velocity ranges 
	given by min_vel and max_vel.
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	min_vel - (float) minimum spectrum x-axis value (eg. velocity, freq)
	max_vel - (float) maximum spectrum x-axis value (eg. velocity, freq)
	Returns:
	(dictionary) Dictionary containing trimmed spectra
	'''
	data_temp = copy.deepcopy(data)
	data_temp['interesting_vel'] = [x for x in data_temp['interesting_vel'] 
		if x >= min_vel and x <= max_vel]

	if data['misc'] != None:
		misc_data = data['misc']
		for spec in range(len(misc_data)):
			vel = np.array(misc_data[spec][0])
			mini = np.amin([np.argmin(np.abs(vel - min_vel)), 
				np.argmin(np.abs(vel - max_vel))])
			maxi = np.amax([np.argmin(np.abs(vel - min_vel)), 
				np.argmin(np.abs(vel - max_vel))])
			misc_data[spec][0] = misc_data[spec][0][mini:maxi + 1]
			misc_data[spec][1] = misc_data[spec][1][mini:maxi + 1]
		data_temp['misc'] = misc_data

	else:
		for f in ['1612','1665','1667','1720']:
			vel = np.array(data_temp['vel_axis'][f])
			tau = np.array(data_temp['tau_spectrum'][f])
			
			if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
				Texp = np.array(data_temp['Texp_spectrum'][f])

			min_vel -= 10.
			max_vel += 10.

			mini = np.amin([np.argmin(np.abs(vel - min_vel)), 
				np.argmin(np.abs(vel - max_vel))])
			maxi = np.amax([np.argmin(np.abs(vel - min_vel)), 
				np.argmin(np.abs(vel - max_vel))])

			data_temp['vel_axis'][f] = vel[mini:maxi + 1]
			data_temp['tau_spectrum'][f] = tau[mini:maxi + 1]

			if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
				data_temp['Texp_spectrum'][f] = Texp[mini:maxi + 1]

	return data_temp
# Find null evidence
def nullevidence(data): 
	'''
	Calculates the evidence of the null model: a flat spectrum.
	Parameter:
	data - (dictionary) dictionary of trimmed spectra
	Returns:
	(float) Natural log of the evidence of the null model.
	'''
	lnllh = 0
	if data['misc'] != None:
		for spec in range(len(data['misc'])):
			model = np.zeros(len(data['misc'][0][0]))

			lnllh = lnlikelihood(model, data['misc'][0][1], 
				findrms(data['misc'][0][1]))

			lnllh += lnllh
	else:
		for f in ['1612','1665','1667','1720']:

			model = np.zeros(len(data['tau_spectrum'][f]))

			lnllh_tau = lnlikelihood(model, data['tau_spectrum'][f], 
				data['tau_rms'][f])

			lnllh += lnllh_tau

			if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
				lnllh_Texp = lnlikelihood(model, data['Texp_spectrum'][f], 
					data['Texp_rms'][f])

				lnllh += lnllh_Texp
	return lnllh	
def lnlikelihood(model, spectrum, sigma):
	'''
	Calculates the likelihood of a spectrum given a model and spectrum noise.
	Parameters:
	model - (array-like) Spectrum generalted by the model
	spectrum - Observed spectrum
	sigma - (float) rms noise level of observed spectrum
	Returns:
	(float) Natural log of the likelihood
	''' 
	N = len(spectrum)
	sse = np.sum((np.array(model) - np.array(spectrum))**2.)
	return -N*np.log(sigma*np.sqrt(2.*np.pi))-(sse/(2.*(sigma**2.)))
def bestparams(chain, lnprob, quiet = False): # 
	'''
	Finds the median of a Markov chain, and the range of each parameter that 
	bounds 68.25% of the accumulated evidence (~+/- 1 sigma range).
	Parameters:
	chain - (array-like) flattened markov chain
	lnprob - (array-like) flattened lnprob
	Keywords:
	quiet - (boolean) prevents printing of evidence and results to terminal
	Returns:
	Tuple with two items:
	(array-like) Array of three values for each parameter: lower sigma bound, median, 
	upper sigma bound.
	(float) Natural log of the evidence.
	'''

	(final_results, final_evidence) = ([], -np.inf)

	num_steps = len(chain)
	num_param = len(chain[0])

	final_array = [list(reversed(sorted(lnprob)))]
	final_darray = [list(reversed(sorted(lnprob)))]

	for param in range(num_param):
		param_chain = [chain[x][param] for x in range(num_steps)]
		final_array = np.concatenate((final_array, [[x for _,x in 
			list(reversed(sorted(zip(lnprob, param_chain))))]]))
		zipped = sorted(zip(param_chain, lnprob))
		sorted_param_chain, sorted_lnprob = zip(*zipped)

		dparam_chain = [0] + [sorted_param_chain[x] - sorted_param_chain[x-1] 
			for x in range(1, len(sorted_param_chain))]
		sorted_dparam_chain = [[x for _,x in 
			list(reversed(sorted(zip(sorted_lnprob, dparam_chain))))]]
		final_darray = np.concatenate((final_darray, sorted_dparam_chain), 
			axis = 0) 

	accumulated_evidence = np.zeros(num_steps)
	for step in range(num_steps):
		# multiply all dparam values
		param_volume = 1
		for param in range(1, len(final_darray)):
			param_volume *= final_darray[param][step]
		if param_volume != 0:
			contribution_to_lnevidence = (np.log(param_volume) + 
				final_darray[0][step])
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
	if not quiet:	
		print('Evidence and Preliminary results:'+'\t'+str(final_evidence)+
			'\t'+str(final_results))
	return (final_results, final_evidence)
# initial pos for walkers
def p0gen(vel_range, num_gauss, data, nwalkers):
	'''
	Generates initial positions of walkers for the MCMC simulation.
	Parameters:
	vel_range - (1D array-like) allowed velocity range
	num_gauss - (int) number of gaussian components in model
	data - (dictionary) dictionary of trimmed spectra
	nwalkers - (int) number of walkers
	Returns:
	(numpy array) Initial positions of all walkers.
	'''
	if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		p0_2 = 0
		for walker in range(nwalkers):
			p0_1 = 0
			for comp in range(num_gauss):
				p0_0 = [np.random.uniform(np.min(vel_range), 
					np.max(vel_range)), np.random.uniform(np.min(FWHM_range), 
					np.max(FWHM_range)), 
					np.random.uniform(np.min(data['N_ranges'][0]), 
						np.max(data['N_ranges'][0])), 
					np.random.uniform(np.min(data['N_ranges'][1]), 
						np.max(data['N_ranges'][1])), 
					np.random.uniform(np.min(data['N_ranges'][2]), 
						np.max(data['N_ranges'][2])), 
					np.random.uniform(np.min(data['N_ranges'][3]), 
						np.max(data['N_ranges'][3]))]
				if p0_1 == 0:
					p0_1 = p0_0
				else:
					p0_1 += p0_0

			if p0_2 == 0:
				p0_2 = [p0_1]
			else:
				p0_2 += [p0_1]

		return p0_2
	elif data['misc'] != None:
		num_spec = len(data['misc'])
		p0_2 = 0
		for walker in range(nwalkers):
			p0_1 = 0
			for comp in range(num_gauss):
				p0_0 = [np.random.uniform(np.min(vel_range), 
					np.max(vel_range)), np.random.uniform(np.min(FWHM_range), 
					np.max(FWHM_range))]
				for spec in range(num_spec):
					p0_0 += [np.random.uniform(
						np.min(data['misc'][spec][1]), 
						np.max(data['misc'][spec][1]))]
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
			[-2 * np.abs(np.amin(data['tau_spectrum']['1612'])), 
				2 * np.abs(np.amax(data['tau_spectrum']['1612']))], 
			[-1.5 * np.abs(np.amin(data['tau_spectrum']['1665'])), 
				1.5 * np.abs(np.amax(data['tau_spectrum']['1665']))], 
			[-1.5 * np.abs(np.amin(data['tau_spectrum']['1667'])), 
				1.5 * np.abs(np.amax(data['tau_spectrum']['1667']))], 
			[-2 * np.abs(np.amin(data['tau_spectrum']['1720'])), 
				2 * np.abs(np.amax(data['tau_spectrum']['1720']))]) 

		# define axes for meshgrid
		t1612 = np.arange(tau_1612_range[0], 
			tau_1612_range[1], (tau_1612_range[1] - 
			tau_1612_range[0]+1e-10)/(100*num_gauss**(1./3.)))
		t1667 = np.arange(tau_1667_range[0], 
			tau_1667_range[1], (tau_1667_range[1] - 
			tau_1667_range[0]+1e-10)/(100*num_gauss**(1./3.)))
		t1720 = np.arange(tau_1720_range[0], 
			tau_1720_range[1], (tau_1720_range[1] - 
			tau_1720_range[0]+1e-10)/(100*num_gauss**(1./3.)))

		tt1612, tt1667, tt1720 = np.meshgrid(t1612, t1667, t1720, 
			indexing = 'ij')
		t1665 = 5.*tt1612 + 5.*tt1720 - (5.*tt1667/9.)

		good_values = np.argwhere((t1665 > tau_1665_range[0]) & 
			(t1665 < tau_1665_range[1]))

		if len(good_values) >= num_gauss * nwalkers:
			p0_indices = good_values[np.random.choice(
				np.arange(len(good_values)), nwalkers * num_gauss, 
					replace = False)]
		else:
			p0_indices = good_values[np.random.choice(
				np.arange(len(good_values)), nwalkers * num_gauss, 
					replace = True)]

		vel_guesses = [sorted([np.random.uniform(vel_range[0], vel_range[1]) 
			for x in range(num_gauss)]) for y in range(nwalkers)]
		for comp in range(num_gauss):
			p0_comp = [[vel_guesses[x][comp], 
				np.random.uniform(FWHM_range[0], FWHM_range[1]), 
					t1612[p0_indices[comp*nwalkers + x][0]],
					t1667[p0_indices[comp*nwalkers + x][1]],
					t1720[p0_indices[comp*nwalkers + x][2]]] 
				for x in range(nwalkers)]
			if comp == 0:
				p0 = p0_comp
			else:
				p0 = np.concatenate((p0, p0_comp), axis = 1)
		return p0
# converts between tau, Tex, Texp, N
def Texp(tau, Tbg, Tex): 
	'''
	Calculated the expected brightness temperature given the excitation 
	temperature, background brightness temperature and optical depth.
	Parameters:
	tau - (float) optical depth
	Tbg - (float) background brightness temperature (kelvin)
	Tex - (float) excitation temperature (kelvin)
	Returns:
	(float) Expected brightness temperature in kelvin
	'''
	Texp = (Tex - Tbg) * (1 - np.exp(-tau))
	return Texp
def tau3(tau_1612=None, tau_1665=None, tau_1667=None, tau_1720=None): 
	'''
	Applies the OH optical depth sum rule:
		tau_1612 + tau_1720 = tau_1665/5 + tau_1667/9
	to either:
		- a set of 3 optical depths to find the fourth, or
		- a set of 4 optical depths to evaluate adherance to the sum rule
	Parameters:
	tau_(1612, 1665, 1667, 1720) - (float) Optical depth of OH ground 
		rotational state transition
	Returns:
	either:
	(list) List of all four optical depths, or
	prints residual of sum rule
	'''
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
	elif (tau_list == None).sum() == 0: 
		print('4 values of tau provided, confirming adherence to sum rule.')
		sum_res = np.abs(tau_1665/5 + tau_1667/9 - tau_1612 - tau_1720)
		print('Residual of sum rule = ' + str(sum_res))
		print('Percentages of supplied tau values: ' + 
			str(np.abs(sum_res/tau_list)))
	else: # can't do anything
		print('Error, at least 3 values of tau needed to apply the sum rule.')
		return None
def tauTexN(logN1, logN2, logN3, logN4, fwhm): 
	'''
	Calculates optical depths and excitation temperatures from the column 
		densities of the four ground rotaitonal state levels of OH:


									  1612 1665 1667 1720
									 ____________________ N1
								 ___/			 |	  |
								/   \____________|____|__ N2
			2Pi 3/2 J = 3/2 ---<	 __|____|____|____|__ N3
								\___/		|		  |
									\_______|_________|__ N4
												  

	Parameters:
	logN(1,2,3,4) - (float) log (base 10) of the column densities of the four 
		ground rotational state levels of OH.
	fwhm - (float) Full width at half-maximum of the gaussian feature (km/sec)
	'''
	con = {
	'rest_freq': {'1612': 1612231000., '1665': 1665402000., 
		'1667': 1667359000., '1720': 1720530000.},
	'gu': {'1612': 3., '1665': 3., '1667': 5., '1720': 5.},
	'gl': {'1612': 5., '1665': 3., '1667': 5., '1720': 3.},
	'logNu': {'1612': logN2, '1665': logN2, '1667': logN1, '1720': logN1},
	'logNl': {'1612': logN3, '1665': logN4, '1667': logN3, '1720': logN4}}
	h = 6.62607004E-27
	k = 1.38064852E-16
	# calculating ln(Nlgu/Nugl) separately
	lnNg_1612 = (np.log(10.)*(con['logNl']['1612']-con['logNu']['1612'])+
		np.log(con['gu']['1612'])-np.log(con['gl']['1612']))
	lnNg_1665 = np.log(10.)*(con['logNl']['1665']-con['logNu']['1665'])
	lnNg_1667 = np.log(10.)*(con['logNl']['1667']-con['logNu']['1667'])
	lnNg_1720 = (np.log(10.)*(con['logNl']['1720']-con['logNu']['1720'])+
		np.log(con['gu']['1720'])-np.log(con['gl']['1720']))

	# compute Tex
	Tex_1612 = (h*con['rest_freq']['1612'])/(k*lnNg_1612)
	Tex_1665 = (h*con['rest_freq']['1665'])/(k*lnNg_1665)
	Tex_1667 = (h*con['rest_freq']['1667'])/(k*lnNg_1667)
	Tex_1720 = (h*con['rest_freq']['1720'])/(k*lnNg_1720)

	# coefficients
	[coeff_1612, coeff_1665, coeff_1667, coeff_1720] = [6883817863.273635, 
		799525441.7995559, 739481592.616534, 3869653815.260256]
	# where:
	# coeff_1612 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1612']*
	# 	con['rest_freq']['1612']**2)/(con['gu']['1612']*con['Aul']['1612']*
	# 	(c**3)*h*2.*np.sqrt(np.log(2.)))), etc.
	# 'Aul': {'1612': 1.302E-11, '1665': 7.177E-11, '1667': 7.778E-11, 
	# 	'1720': 9.496E-12}
	# c = 2.99792458E+10

	# compute peak tau
	tau_1612 = (10.**con['logNl']['1612'])/(coeff_1612*Tex_1612*fwhm*(10**5))
	tau_1665 = (10.**con['logNl']['1665'])/(coeff_1665*Tex_1665*fwhm*(10**5))
	tau_1667 = (10.**con['logNl']['1667'])/(coeff_1667*Tex_1667*fwhm*(10**5))
	tau_1720 = (10.**con['logNl']['1720'])/(coeff_1720*Tex_1720*fwhm*(10**5))

	return [tau_1612, tau_1665, tau_1667, tau_1720, 
			Tex_1612, Tex_1665, Tex_1667, Tex_1720]
def NtauTex(tau_1612, tau_1665, tau_1667, tau_1720, 
	Tex_1612, Tex_1665, Tex_1667, Tex_1720, fwhm): 
	'''
	Determines column densities of the 4 OH ground state levels from optical 
		depth and excitation temperature of the four ground state transitions:

									  1612 1665 1667 1720
									 ____________________ N1
								 ___/			 |	  |
								/   \____________|____|__ N2
			2Pi 3/2 J = 3/2 ---<	 __|____|____|____|__ N3
								\___/		|		  |
									\_______|_________|__ N4
	
	Parameters:
	tau_(freq) - (float) optical depth
	Tex_(freq) - (float) excitation temperature (K)
	fwhm - (freq) full width at half-maximum (km/s)
	Returns:
	(list) log (base 10) of the column densities of the four 
		ground rotational state levels of OH.
	'''
	N3 = Tex_1667*tau_1667*fwhm*7.39481592616533E+13
	N1 = N3*(np.exp(-0.0800206378074005/Tex_1667))
	N2 = N3*((3/5)*np.exp(-0.0773749102100167/Tex_1612))
	N4 = N1/((5/3)*np.exp(-0.0825724441867449/Tex_1720))

	return [np.log10(N1), np.log10(N2), np.log10(N3), np.log10(N4)]
def NrangetauTexp(data): 
	'''
	Determines the ranges of possible column densities that correspond to the 
		range of tau and Texp in the data.
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	Returns:
	(array-line) 2D array with [[min log(10) N1, max log(10) N1], ...]
	'''
	# relevant constants (cgs):
	con = {
	'rest_freq': {'1612': 1612231000., '1665': 1665402000., 
		'1667': 1667359000., '1720': 1720530000.},
	'gu': {'1612': 3., '1665': 3., '1667': 5., '1720': 5.},
	'gl': {'1612': 5., '1665': 3., '1667': 5., '1720': 3.}}
	h = 6.62607004E-27
	k = 1.38064852E-16

	[coeff_1612, coeff_1665, coeff_1667, coeff_1720] = [6883817863.273635, 
		799525441.7995559, 739481592.616534, 3869653815.260256]
	# where:
	# coeff_1612 = ((8.*np.pi*k*np.sqrt(np.pi)*con['gl']['1612']*
	# 	con['rest_freq']['1612']**2)/(con['gu']['1612']*con['Aul']['1612']*
	# 	(c**3)*h*2.*np.sqrt(np.log(2.)))), etc.
	# 'Aul': {'1612': 1.302E-11, '1665': 7.177E-11, '1667': 7.778E-11, 
	# 	'1720': 9.496E-12}
	# c = 2.99792458E+10

	Tex_1612 = data['Tbg']['1612']+(np.array(data[
		'Texp_spectrum']['1612'])/(1.-np.exp(-np.array(data[
		'tau_spectrum']['1612']))))
	Tex_1665 = data['Tbg']['1665']+(np.array(data[
		'Texp_spectrum']['1665'])/(1.-np.exp(-np.array(data[
		'tau_spectrum']['1665']))))
	Tex_1667 = data['Tbg']['1667']+(np.array(data[
		'Texp_spectrum']['1667'])/(1.-np.exp(-np.array(data[
		'tau_spectrum']['1667']))))
	Tex_1720 = data['Tbg']['1720']+(np.array(data[
		'Texp_spectrum']['1720'])/(1.-np.exp(-np.array(data[
		'tau_spectrum']['1720']))))
	
	logN3_1 = np.abs(np.log10(np.abs(coeff_1667*(10**5)*FWHM_range[1]*np.array(
		data['tau_spectrum']['1667'])*(Tex_1667))))
	logN3_2 = np.abs(np.log10(np.abs(coeff_1612*(10**5)*FWHM_range[1]*np.array(
		data['tau_spectrum']['1612'])*(Tex_1612))))
	logN3_3 = np.abs(np.log10(np.abs(coeff_1667*(10**5)*FWHM_range[0]*np.array(
		data['tau_spectrum']['1667'])*(Tex_1667))))
	logN3_4 = np.abs(np.log10(np.abs(coeff_1612*(10**5)*FWHM_range[0]*np.array(
		data['tau_spectrum']['1612'])*(Tex_1612))))
	logN4_1 = np.abs(np.log10(np.abs(coeff_1665*(10**5)*FWHM_range[1]*np.array(
		data['tau_spectrum']['1665'])*(Tex_1665))))
	logN4_2 = np.abs(np.log10(np.abs(coeff_1720*(10**5)*FWHM_range[1]*np.array(
		data['tau_spectrum']['1720'])*(Tex_1720))))
	logN4_3 = np.abs(np.log10(np.abs(coeff_1665*(10**5)*FWHM_range[0]*np.array(
		data['tau_spectrum']['1665'])*(Tex_1665))))
	logN4_4 = np.abs(np.log10(np.abs(coeff_1720*(10**5)*FWHM_range[0]*np.array(
		data['tau_spectrum']['1720'])*(Tex_1720))))

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

	logN1_min = np.nanmin([np.nanmin(logN1_1), np.nanmin(logN1_2), 
		np.nanmin(logN1_3), np.nanmin(logN1_4)])
	logN1_max = np.nanmax([np.nanmax(logN1_1), np.nanmax(logN1_2), 
		np.nanmax(logN1_3), np.nanmax(logN1_4)])

	logN2_min = np.nanmin([np.nanmin(logN2_1), np.nanmin(logN2_2), 
		np.nanmin(logN2_3), np.nanmin(logN2_4)])
	logN2_max = np.nanmax([np.nanmax(logN2_1), np.nanmax(logN2_2), 
		np.nanmax(logN2_3), np.nanmax(logN2_4)])	

	return [[logN1_min, logN1_max], [logN2_min, logN2_max], 
		[logN3_min, logN3_max], [logN4_min, logN4_max]]
# makes/plots model from params
def makemodel(params, data, num_gauss, accepted_params = []): 
	'''
	Generates model spectra from input parameters.
	Parameters:
	params - (list) parameters for the models. If 'data' contains tau and Texp 
		spectra, these will be [mean vel(1), fwhm(1), N1(1), N2(1), N3(1), 
		N4(1), ...(num_gauss)] for 'num_gauss' gaussian components. If 'data'
		contains only tau spectra, these will be [mean vel(1), fwhm(1), 
		tau_1(1), tau_2(1), tau_3(1), tau_4(1), ...(num_gauss)]. If 'misc' 
		spectra are provided, these will be [mean vel(1), fwhm(1), h1(1), 
		h2(1), ... hN(1), ... hN(num_gauss)] for N provided spectra and 
		'num_gauss' gaussian components.
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	num_gauss - (int) number of gaussian components.
	Keywords:
	accepted_params - (list) parameters of any previously accepted components 
		(i.e. from another velocity range). The same form as 'params'. These 
		are included to account for the tail of a neighbouring gaussian that 
		may influence the current velocity range.
	Returns:
		(tuple of arrays) Tuple of model spectra. If 'data' contains tau and Texp 
		spectra, these will be (tau_1612, tau_1665, tau_1667, tau_1720, tau_1612, tau_1665, tau_1667, tau_1720). If 'data' 
		contains only tau spectra, these will be [mean vel(1), fwhm(1), 
		tau_1(1), tau_2(1), tau_3(1), tau_4(1), ...(num_gauss)]. If 'misc' 
		spectra are provided, these will be [mean vel(1), fwhm(1), h1(1), 
		h2(1), ... hN(1), ... hN(num_gauss)] for N provided spectra and 
		'num_gauss' gaussian components.
	'''
	# initialise models
	if accepted_params != []:
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, 
				Texp_m_1612, Texp_m_1665, 
				Texp_m_1667, Texp_m_1720) = makemodel(accepted_params, data, 
					int(len(accepted_params) / 6))
		elif data['misc'] != None:
			models = makemodel(accepted_params, data, 
				int(len(accepted_params) / len(data['misc'])))
		else:
			(tau_m_1612, tau_m_1665, 
				tau_m_1667, tau_m_1720) = makemodel(accepted_params, data, 
					int(len(accepted_params) / 5))
	else:
		if data['misc'] != None:
			models = [np.zeros(int(len(data['misc'][x][0]))) for x in 
				range(len(data['misc']))]
		else:
			vel_1612 = data['vel_axis']['1612']
			vel_1665 = data['vel_axis']['1665']
			vel_1667 = data['vel_axis']['1667']
			vel_1720 = data['vel_axis']['1720']

			num_params = int(len(params) / num_gauss)

			tau_m_1612 = np.zeros(len(vel_1612))
			tau_m_1665 = np.zeros(len(vel_1665))
			tau_m_1667 = np.zeros(len(vel_1667))
			tau_m_1720 = np.zeros(len(vel_1720))

			if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
				Texp_m_1612 = np.zeros(len(vel_1612))
				Texp_m_1665 = np.zeros(len(vel_1665))
				Texp_m_1667 = np.zeros(len(vel_1667))
				Texp_m_1720 = np.zeros(len(vel_1720))
	
	# make models
	if data['misc'] != None:
		for comp in range(int(num_gauss)): 
			num_spec = len(data['misc'])
			for spec in range(num_spec):
				models[spec] += gaussian(mean = params[comp*(num_spec+2)], 
					FWHM = params[comp*(num_spec+2) + 1], 
					height = params[comp*(num_spec+2) + 
						spec + 2])(np.array(data['misc'][spec][0]))
	elif 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		for comp in range(int(num_gauss)): 
			[vel, FWHM, logN1, logN2, logN3, logN4] = params[comp * 
				num_params:(comp + 1) * num_params]
			[tau_1612, tau_1665, tau_1667, tau_1720, 
				Tex_1612, Tex_1665, Tex_1667, Tex_1720] = tauTexN(
					logN1 = logN1, logN2 = logN2, logN3 = logN3, 
					logN4 = logN4, fwhm = FWHM)
			[Texp_1612, Texp_1665, Texp_1667, Texp_1720] = [
				Texp(tau_1612, data['Tbg']['1612'], Tex_1612), 
				Texp(tau_1665, data['Tbg']['1665'], Tex_1665), 
				Texp(tau_1667, data['Tbg']['1667'], Tex_1667), 
				Texp(tau_1720, data['Tbg']['1720'], Tex_1720)]
			tau_m_1612 += gaussian(vel, FWHM, tau_1612)(np.array(vel_1612))
			tau_m_1665 += gaussian(vel, FWHM, tau_1665)(np.array(vel_1665))
			tau_m_1667 += gaussian(vel, FWHM, tau_1667)(np.array(vel_1667))
			tau_m_1720 += gaussian(vel, FWHM, tau_1720)(np.array(vel_1720))
			Texp_m_1612 += gaussian(vel, FWHM, Texp_1612)(np.array(vel_1612))
			Texp_m_1665 += gaussian(vel, FWHM, Texp_1665)(np.array(vel_1665))
			Texp_m_1667 += gaussian(vel, FWHM, Texp_1667)(np.array(vel_1667))
			Texp_m_1720 += gaussian(vel, FWHM, Texp_1720)(np.array(vel_1720))
	else:
		for comp in range(int(num_gauss)): 
			[vel, FWHM, tau_1612, tau_1667, tau_1720] = params[comp * 
				num_params:(comp + 1) * num_params]
			[tau_1612, tau_1665, tau_1667, tau_1720] = tau3(
				tau_1612 = tau_1612, tau_1667 = tau_1667, tau_1720 = tau_1720)
			tau_m_1612 += gaussian(vel, FWHM, tau_1612)(np.array(vel_1612))
			tau_m_1665 += gaussian(vel, FWHM, tau_1665)(np.array(vel_1665))
			tau_m_1667 += gaussian(vel, FWHM, tau_1667)(np.array(vel_1667))
			tau_m_1720 += gaussian(vel, FWHM, tau_1720)(np.array(vel_1720))

	# return models
	if data['misc'] != None:
		return models
	elif 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, 
			Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720)	
	else:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720)
def gaussian(mean, FWHM = None, height = None, sigma = None, 
	amp = None): 
	'''
	Generates a gaussian profile with the given parameters. Mean must be 
		provided, as well as FWHM or sigma, and height or amp. Must be called 
		with a 'x' axis array (i.e. velocity axis)
	Parameters:
	mean - (float) mean of gaussian (line centre)
	Keywords:
	FWHM - (float) full width at half maximum
	height - (float) maximum 'y' value
	sigma - (float) standard deviation
	amp - (float) integral of full gaussian profile (= height * sigma * 
		sqrt(2*pi))
	Returns:
	(lambda) Value of gaussian function at 'x'
	'''
	if sigma == None:
		sigma = FWHM / (2. * np.sqrt(2. * np.log(2.)))

	if height == None:
		height = amp / (sigma * np.sqrt(2.* np.pi))
	return lambda x: height * np.exp(-((x - mean)**2.) / (2.*sigma**2.))
# sample posterior using emcee
def sampleposterior(data, num_gauss, p0, vel_range, nwalkers, accepted = []): 
	'''
	Samples the posterior probability distribution.
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	num_gauss - (int) number of gaussian components.
	p0 - (numpy array) initial positions of all walkers.
	vel_range - (1D array-like) allowed velocity range
	nwalkers - (even int) the number of walkers
	Keywords:
	accepted_params - (list) parameters of any previously accepted components 
		(i.e. from another velocity range). If 'data' contains tau and Texp 
		spectra, these will be [mean vel(1), fwhm(1), N1(1), N2(1), N3(1), 
		N4(1), ...(num_gauss)] for 'num_gauss' gaussian components. If 'data'
		contains only tau spectra, these will be [mean vel(1), fwhm(1), 
		tau_1(1), tau_2(1), tau_3(1), tau_4(1), ...(num_gauss)]. If 'misc' 
		spectra are provided, these will be [mean vel(1), fwhm(1), h1(1), 
		h2(1), ... hN(1), ... hN(num_gauss)] for N provided spectra and 
		'num_gauss' gaussian components. These are included to account for the 
		tail of a neighbouring gaussian that may influence the current 
		velocity range.
	Returns:
	(tuple) Two array-like: (2D array, 1D array). First is the flattened 
		Markov chain: the positions in parameter space visited by all walkers 
		for all iterations. Second is the flattened ln probability chain: the 
		value of the posterior probability distribution at all the points in 
		parameter space indicated by the flattened Markov chain.
	'''
	if data['misc'] != None:
		ndim = (2 + len(data['misc'])) * num_gauss
	elif 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		ndim = 6 * num_gauss
	else:
		ndim = 5 * num_gauss
	
	if accepted != []:
		accepted_params = []
	else:
		accepted_params = [x[1] for x in accepted]

	burn_iterations = 500
	final_iterations = 100
	
	args = [data, vel_range, num_gauss, accepted_params]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args)
	# burn
	[burn_run, test_result] = [0, 'Fail']

	while burn_run <= 6 and test_result == 'Fail':
		try:
			sampler.reset()
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		except ValueError: # sometimes there is an error within emcee
			print('emcee is throwing a value error for p0. Running again.')
			# sampler.reset()
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
		# test convergence
		(test_result, p0) = convergencetest(sampler_chain = sampler.chain, 
			num_gauss = num_gauss, pos = pos)
		print('Convergence test result: ' + test_result)
		if test_result == 'Fail':
			# plotchain(sampler.chain)
			p0 = bestwalkers(sampler.flatchain, sampler.flatlnprobability, 
				nwalkers)
		burn_run += 1

	# final run
	sampler.reset()
	sampler.run_mcmc(pos, final_iterations)
	# remove steps where lnprob = -np.inf
	flatchain = sampler.flatchain
	flatlnprob = sampler.flatlnprobability

	chain = [flatchain[x] for x in range(len(flatchain)) 
		if flatlnprob[x] != -np.inf]

	if np.array(chain).shape[0] == 0:
		print('No finite members of posterior')
		return (np.array([]), np.array([]))
	else:
		lnprob_ = [x for x in flatlnprob if x != -np.inf]
		return (np.array(chain), np.array(lnprob_))
def plotchain(chain):
	'''
	Generates a set of simple plots of the Markov chain. A plot is produced 
		for each parameter, and shows the paths taken through parameter space 
		of each walker over time (i.e. over all iterations). Useful for 
		diagnosing non-convergence or degeneracy.
	Parameters:
	chain - (array-like) 3D array (i.e. not flattened) of Markov chain. 
		Dimensions: [number of walkers, number of iterations, number of 
		parameters].
	'''
	nwalkers = chain.shape[0]
	nstep = chain.shape[1]
	ndim = chain.shape[2]
	for parameter in range(ndim):
		plt.figure()
		for walker in range(nwalkers):
			plt.plot(range(nstep), chain[walker,:,parameter])
		plt.title('Chains for param ' + str(parameter) + ': burn in')
		plt.xlabel('Iterations')
		plt.ylabel('Parameter value')
		plt.show()
		plt.close()
def lnprob(lnprobx, data, vel_range, num_gauss, accepted_params = []):
	'''
	Returns the value of the posterior probability at the point in parameter 
		space given by 'lnprobx'.
	Parameters:
	lnprobx - (array-like) 1D position vector in parameter space assigned by 
		emcee
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	vel_range - (1D array-like) allowed velocity range
	num_gauss - (int) number of gaussian components
	Keywords:
	accepted_params - (list) parameters of any previously accepted components 
		(i.e. from another velocity range). If 'data' contains tau and Texp 
		spectra, these will be [mean vel(1), fwhm(1), N1(1), N2(1), N3(1), 
		N4(1), ...(num_gauss)] for 'num_gauss' gaussian components. If 'data'
		contains only tau spectra, these will be [mean vel(1), fwhm(1), 
		tau_1(1), tau_2(1), tau_3(1), tau_4(1), ...(num_gauss)]. If 'misc' 
		spectra are provided, these will be [mean vel(1), fwhm(1), h1(1), 
		h2(1), ... hN(1), ... hN(num_gauss)] for N provided spectra and 
		'num_gauss' gaussian components. These are included to account for the 
		tail of a neighbouring gaussian that may influence the current 
		velocity range.
	Returns:
	(float) natural log of the posterior probability distribution at a the 
		sampled point in parameter space.
	'''
	prior = lnprprior(data, lnprobx, vel_range, num_gauss)
	params = lnprobx
	models = makemodel(params, data, num_gauss, accepted_params)
	if data['misc'] != None:
		spectra = [x[1] for x in data['misc']]
		rms = [findrms(x) for x in spectra]
	elif 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		[tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, Texp_m_1612, 
		Texp_m_1665, Texp_m_1667, Texp_m_1720] = models
		spectra = [	data['tau_spectrum']['1612'], 
					data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], 
					data['tau_spectrum']['1720'], 
					data['Texp_spectrum']['1612'], 
					data['Texp_spectrum']['1665'], 
					data['Texp_spectrum']['1667'], 
					data['Texp_spectrum']['1720']]
		rms = [		data['tau_rms']['1612'], 
					data['tau_rms']['1665'], 
					data['tau_rms']['1667'], 
					data['tau_rms']['1720'], 
					data['Texp_rms']['1612'], 
					data['Texp_rms']['1665'], 
					data['Texp_rms']['1667'], 
					data['Texp_rms']['1720']]
	else:
		[tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720] = models
		spectra = [	data['tau_spectrum']['1612'], 
					data['tau_spectrum']['1665'], 
					data['tau_spectrum']['1667'], 
					data['tau_spectrum']['1720']]
		rms = [		data['tau_rms']['1612'], 
					data['tau_rms']['1665'], 
					data['tau_rms']['1667'], 
					data['tau_rms']['1720']]
	lprob = prior 
	for a in range(len(spectra)):
		llh = lnlikelihood(models[a], spectra[a], rms[a])
		# print('prior: ' + str(prior) + '\tlikelihood: ' + str(llh))
		lprob += llh
	if np.isnan(lprob):
		# print('lnprob wanted to return NaN for parameters: ' + str(lnprobx))
		return -np.inf
	else:
		return lprob	
def lnprprior(data, params, vel_range, num_gauss): 
	'''
	Returns the value of the pprior probability at the point in parameter 
		space given by 'params'.
	Parameters:
	data - (dictionary) observed spectra and other observed quantities (see 
		Main function)
	params - (array-like) 1D position vector in parameter space
	vel_range - (1D array-like) allowed velocity range
	num_gauss - (int) number of gaussian components
	Returns:
	(float) natural log of the prior probability distribution at a the sampled 
		point in parameter space.
	'''
	lnprprior = 0
	if data['misc'] != None:
		vel_prev = vel_range[0]
		num_params = len(data['misc']) + 2
		for gauss in range(num_gauss):
			lnprprior += lnnaiveprior(value = params[int(gauss*num_params)], 
				value_range = [vel_prev, vel_range[1]])
			lnprprior += lnnaiveprior(value = params[int(gauss*num_params)+1], 
				value_range = FWHM_range)
			for spec in range(len(data['misc'])):
				lnprprior += lnnaiveprior(
					value = params[int(gauss*num_params+spec+2)], 
					value_range = [-1.5*np.abs(
						np.min(data['misc'][spec][1])),
					1.5*np.abs(np.max(data['misc'][spec][1]))])
	else:
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			[logN1_range, logN2_range, 
				logN3_range, logN4_range] = data['N_ranges']
		vel_prev = vel_range[0]
		for gauss in range(num_gauss):
			# define params
			if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
				[vel, FWHM, logN1, logN2, logN3, 
					logN4] = params[int(gauss * 6):int((gauss + 1) * 6)]
				[tau_1612, tau_1665, tau_1667, tau_1720, Tex_1612, Tex_1665, 
					Tex_1667, Tex_1720] = tauTexN(logN1, logN2, logN3, 
						logN4, FWHM)
			else:
				[vel, FWHM, tau_1612, tau_1667, 
					tau_1720] = params[int(gauss * 5):int((gauss + 1) * 5)]
				[tau_1612, tau_1665, tau_1667, 
					tau_1720] = tau3(tau_1612 = tau_1612, 
							tau_1667 = tau_1667, 
							tau_1720 = tau_1720)
			# calculate priors
			vel_prior = lnnaiveprior(value = vel, 
				value_range = [vel_prev, vel_range[1]])
			FWHM_prior = lnnaiveprior(value = FWHM, value_range = FWHM_range)
			if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
				logN1_prior = lnnaiveprior(value = logN1, 
					value_range = logN1_range)
				logN2_prior = lnnaiveprior(value = logN2, 
					value_range = logN2_range)
				logN3_prior = lnnaiveprior(value = logN3, 
					value_range = logN3_range)
				logN4_prior = lnnaiveprior(value = logN4, 
					value_range = logN4_range)
				lnprprior += np.sum(vel_prior,FWHM_prior,logN1_prior,
					logN2_prior,logN3_prior,logN4_prior)				
			else:
				(tau_1612_range, tau_1665_range, 
					tau_1667_range, tau_1720_range) = (
					[-2 * np.abs(np.amin(data['tau_spectrum']['1612'])), 
					2 * np.abs(np.amax(data['tau_spectrum']['1612']))], 
					[-1.5 * np.abs(np.amin(data['tau_spectrum']['1665'])), 
					1.5 * np.abs(np.amax(data['tau_spectrum']['1665']))], 
					[-1.5 * np.abs(np.amin(data['tau_spectrum']['1667'])), 
					1.5 * np.abs(np.amax(data['tau_spectrum']['1667']))], 
					[-2 * np.abs(np.amin(data['tau_spectrum']['1720'])), 
					2 * np.abs(np.amax(data['tau_spectrum']['1720']))])
				tau_1612_prior = lnnaiveprior(value = tau_1612, 
					value_range = tau_1612_range)
				tau_1665_prior = lnnaiveprior(value = tau_1665, 
					value_range = tau_1665_range)
				tau_1667_prior = lnnaiveprior(value = tau_1667, 
					value_range = tau_1667_range)
				tau_1720_prior = lnnaiveprior(value = tau_1720, 
					value_range = tau_1720_range)
				lnprprior += np.sum(vel_prior,FWHM_prior,tau_1612_prior,
					tau_1665_prior,tau_1667_prior,tau_1720_prior)
	return lnprprior
def convergencetest(sampler_chain, num_gauss, pos): 
	'''
	Tests the convergence of a Markov chain. If the chain has not converged, 
		will replace the last position of the walkers with a 'better' position 
		in parameter space.
	Parameters:
	sampler_chain - (array-like) 3D array (i.e. not flattened) of Markov chain.
		Dimensions: [number of walkers, number of iterations, number of 
		parameters].
	num_gauss - (int) number of gaussian components
	pos - (array-like) final position in parameter space
	Returns:
	(tuple) Tuple with two elements: first the string 'Pass' or 'Fail', second 
		the position in parameter space for each walker to continue from.

	'''
	model_dim = int(sampler_chain.shape[2] / num_gauss)
	orig_num_walkers = sampler_chain.shape[0]
	counter = 0

	# remove dead walkers
	for walker in reversed(range(sampler_chain.shape[0])):
		if sampler_chain[walker,0,0] == sampler_chain[walker,-1,0]:
			sampler_chain = np.delete(sampler_chain, walker, 0)
			counter += 1
	# replace removed walkers
	if counter > 0 and counter < orig_num_walkers / 2:
		for x in range(counter):
			sampler_chain = np.concatenate((sampler_chain, [sampler_chain[0]]), 
				axis = 0)
	elif counter >= orig_num_walkers / 2:
		print('Convergence test failed due to too many dead walkers')
		return ('Fail', pos)

	# test convergence in velocity
	for comp in range(num_gauss):

		var_within_chains = np.median([np.var(sampler_chain[x,-25:-1,
			comp * model_dim]) for x in range(sampler_chain.shape[0])])
		var_across_chains = np.median([np.var(sampler_chain[:,-x-1,
			comp * model_dim]) for x in range(24)])
		ratio = (max([var_within_chains, var_across_chains]) / 
			min([var_within_chains, var_across_chains]))
		max_var = max([var_within_chains, var_across_chains])

		if ratio > 5. and max_var < 1.:
			print('Convergence test failed due to component number ' + 
				str(comp+1) + ' with variance ratio = ' + 
				str(round(ratio, 2)) + ' and max variance = ' + 
				str(round(max_var, 2)))
			return ('Fail', sampler_chain[:,-1,:])

	return ('Pass', sampler_chain[:,-1,:])
def bestwalkers(flatchain, flatlnprob, nwalkers):
	'''
	Returns 'improved' initial positions for walkers based on a previous 
		Markov chain.
	Parameters:
	flatchain - (array-like) flattened (2D) Markov chain
	flatlnprob - (array-like) flattened (1D) log probability chain
	nwalkers - (int) number of walkers
	Returns:
	(array-like) (2D) positions of nwalkers in parameter space
	'''
	flatchain = np.array(flatchain)
	flatlnprob = np.array(flatlnprob)
	
	inds = (-flatlnprob).argsort()
	sorted_chain = flatchain[inds]

	if len(sorted_chain) >= nwalkers:
		p0 = sorted_chain[0:nwalkers]
	else:
		needed = nwalkers - len(sorted_chain)
		new_raw = np.random.choice(sorted_chain, needed, replace = False)
		new = [[x*(0.01*(np.random.randn()) + 1) for x in y] for y in new_raw]
		p0 = np.concatenate((sorted_chain, new), axis = 0)
	return p0
# priors
def lnnaiveprior(value, value_range): 
	'''
	Calculates the natural log of a naive (flat) prior for the given value 
		assuming the given value range (inclusive).
	Parameters:
	value - (float) value of parameter
	value_range - (array-like) min and max allowed value of parameter
	Returns:
	(float) natural log of the prior probability distribution for this 
		parameter
	'''
	if value >= value_range[0] and value <= value_range[1]:
		return -np.log(np.abs(value_range[1] - value_range[0]))
	else:
		return -np.inf

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
	source_name,
	vel_axes = None, 
	tau_spectra = None, 
	Texp_spectra = None, 
	Tbg = None, 
	misc = None,
	quiet = True, 
	Bayes_threshold = 10., 
	sigma_tolerance = 1.5):
	'''
	Performs fully automated gaussian decomposition.
	Parameters:
	source_name - (string) name of source to be included with results reports
	Keywords:
	vel_axes - (array-like) set of 4 1D arrays corresponding to the 4 spectra 
		of the ground state transitions of OH
	tau_spectra - (array-like) set of 4 1D arrays of optical depth in the 4 
		ground rotational state transitions of OH
	Texp_spectra - (array-like) set of 4 1D arrays of expected brightness 
		temperature in the 4 ground rotational state transitions of OH
	Tbg - (array-like) set of 4 background brightness temperatures at 1612, 
		1665, 1667 and 1720 MHz (K)
	misc - (array-like) set of spectra to be fitted simultaneously. Format: 
		[[vel_axis_1, spectrum_1], [vel_axis_2, spectrum_2], ...]
	quiet - (boolean) whether or not to provide various outputs to terminal
	Bayes_threshold - (float) minimum Bayes' Factor (K) required for the 
		fitting routine to attempt to fit an additional gaussian feature
	sigma_tolerance - (float) factor used to identify 'significant' ranges in 
		the supplied spectra. Roughly related to sigma level but *NOT* 
		equivalent. Increase sigma_tolerance to reduce the range of spectra 
		identified.
	Returns:
	(2D array) Full list of accepted parameters with upper and lower bounds of
		credibile intervals. 
		Format: [[1min, 1median, 1max], [2min, 2median, 2max],...] for 
		parameters 1, 2, etc.
	'''
	if not quiet:
		print('source name: ' + str(source_name))

	# initialise data dictionary
	# print('starting ' + source_name)
	data = {'source_name': source_name, 
		'vel_axis':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'tau_spectrum':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'tau_rms':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'Texp_spectrum':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'Texp_rms':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'Tbg':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'misc':misc,
		'sig_vel_ranges':[[]],
		'interesting_vel':[]}

		###############################################
		#                                             #
		#   Load Data into 'data' dictionary object   #
		#                                             #	
		###############################################
	
	if misc == None:
		
		data['tau_rms']['1612'] = findrms(tau_spectra[0])
		data['tau_rms']['1665'] = findrms(tau_spectra[1])
		data['tau_rms']['1667'] = findrms(tau_spectra[2])
		data['tau_rms']['1720'] = findrms(tau_spectra[3])

		data['vel_axis']['1612']		= vel_axes[0]
		data['tau_spectrum']['1612']	= tau_spectra[0]

		data['vel_axis']['1665']		= vel_axes[1]
		data['tau_spectrum']['1665']	= tau_spectra[1]

		data['vel_axis']['1667']		= vel_axes[2]
		data['tau_spectrum']['1667']	= tau_spectra[2]

		data['vel_axis']['1720']		= vel_axes[3]
		data['tau_spectrum']['1720']	= tau_spectra[3]

		if Texp_spectra != None: #absorption and emission spectra are available 
				# (i.e. on-off observations)
			
			data['Texp_rms']['1612'] = findrms(Texp_spectra[0])
			data['Texp_rms']['1665'] = findrms(Texp_spectra[1])
			data['Texp_rms']['1667'] = findrms(Texp_spectra[2])
			data['Texp_rms']['1720'] = findrms(Texp_spectra[3])
			
			data['Tbg']['1612']				= Tbg[0]
			data['Tbg']['1665']				= Tbg[1]
			data['Tbg']['1667']				= Tbg[2]
			data['Tbg']['1720']				= Tbg[3]

			data['Texp_spectrum']['1612']	= Texp_spectra[0]
			data['Texp_spectrum']['1665']	= Texp_spectra[1]
			data['Texp_spectrum']['1667']	= Texp_spectra[2]
			data['Texp_spectrum']['1720']	= Texp_spectra[3]


	###############################################
	#                                             #
	#         Identify significant ranges         #
	#                                             #
	###############################################

	findranges(data, sigma_tolerance)

	###############################################
	#                                             #
	#                Fit gaussians                #
	#                                             #
	###############################################
	
	final_p = placegaussians(data, Bayes_threshold = Bayes_threshold, 
		quiet = quiet)

	return final_p








