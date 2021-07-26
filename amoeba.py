import copy
import emcee
import itertools
import numpy as np
import pickle
from scipy.interpolate import interpn
from scipy.special import logsumexp
import matplotlib.pyplot as plt


'''
                                   ......                                     
                                 ..     ...                                   
                                ..        ..                                  
                               ..          ..                         .....   
                               ..           ..                    ...     ... 
                              ..             ..                ...          ..
                              ..              ..            ...             ..
                             ..                ....    .....               .. 
                            ..                   .......                  ..  
                            ..                                           ..   
                          ..                                            ..    
                        ...                                            ..     
      ...................                        b                    ..      
  ...                                            b                   ..       
 ..  aaaaa a   m mmm mmmm     ooooo     eeeee    b bbbb     aaaaa a  ..       
..  a     aa   m    m    m   o     o   e     e   b     b   a     aa  .        
..  a      a   m    m    m   o     o   e eee     b     b   a      a  .        
 ..  aaaaa a   m    m    m    ooooo     eeeee    b bbbb     aaaaa a  .        
  ...                                                                ..       
      ..........                                                      ..      
               .......                                                 ..     
                     ...                                                ..    
                       ...                       ........                ..   
                        ...                   ....      ...              ..   
                         ..                 ...             ....          ..  
                          ..              ...                   ..       ..   
                          ..            ...                        .......    
                          ..           ..                                     
                           ..        ..                                       
                            ....  ....                                        
                              .....                                           
'''

'''

The four hyperfine levels of the OH ground rotational state:

	1612 1665 1667 1720
	_____________________ 4
	___________|____|____ 3
	_|____|____|____|____ 2
	______|_________|____ 1

'''

# Some key parameters:
# number of burn iterations for emcee (100 is low, 10000 is high)
burn_iter=500
# number of final iterations for emcee (50 is low, 1000 is high)
final_iter=100
# number of walkers per gaussian (must be even and no less than 12, 60 is better)
nwalkers_mult=60



def main(source_name, vel_axes = None, tau_spectra = None, 
	Texp_spectra = [], tau_rms = [], 
	Texp_rms = [], Tbg = None, quiet = True, 
	best_result = 'median_actual', Bayes_threshold = 10., 
	sigma_tolerance = 1.4, num_chan = 3, 
	seed=None, extra_gaussians=3,sig_vels=[],lfwhm_mean=None,lfwhm_sig=None):
	'''
	Performs fully automated gaussian decomposition. This code was written 
	specifically to decompose spectra of the four ground rotational state 
	transitions of the hydroxyl molecule, but can also be used to decompose 
	any set of simultaneous spectra.

	Parameters:
		source_name (str): name of source to be included with results reports
	Keyword Arguments:
		vel_axes (array-like): set of 4 1D arrays corresponding to the 4 
			spectra of the ground state transitions of OH.
		tau_spectra (array-like): set of 4 1D arrays of optical depth vs 
			velocity (tau spectra as opposed to exp(-tau), etc) in the 4 ground 
			rotational state transitions of OH.
		Texp_spectra (array-like): set of 4 1D arrays of continuum subtracted 
			expected brightness temperature in the 4 ground rotational state 
			transitions of OH: Texp=(Tex-Tbg)*(1-exp(-tau)).
		Tbg (array-like): set of 4 background brightness temperatures at 1612, 
			1665, 1667 and 1720 MHz (K). This is any background continuum in 
			the `off-source' position: synchrotron, CMB.
		quiet (boolean): whether or not to provide various outputs to terminal.
		best_result (str): Which result from the markov chain is to be reported
			as the best result 'median', 'median_actual', 'max' (note, all are 
			printed if quiet = False). 'median' returns the median position of 
			each parameter (may not correspond to an actual set of parameters 
			measured), 'median_actual' finds the point in the flattened chain 
			closest to the medians of each parameter and compares the lnprob 
			here to that of the actual medians, returns the one with better 
			lnprob, and 'max' returns the set of parameters with the highest 
			lnprob.
		Bayes_threshold (float): minimum Bayes' Factor (K) required for the 
			fitting routine to attempt to fit an additional gaussian feature.
		sigma_tolerance (float): factor used to identify 'significant' ranges 
			in the supplied spectra. Roughly related to sigma level but *NOT* 
			equivalent. Increase sigma_tolerance to reduce the range of spectra 
			identified.
	Returns:
		array-like: Full list of accepted parameters with upper and lower 
			bounds of credibile intervals. Format: [[1min, 1median, 1max], 
			[2min, 2median, 2max],...] for parameters 1, 2, etc.
	'''

	# Quick checks of input data:
	if len(Texp_spectra) != 0:
		for x in range(4):
			if len(vel_axes[x]) != len(Texp_spectra[x]):
				# print('Provided axes (' + str(x) + ') lengths do not match')
				# print('\tvel axis length: '+str(len(vel_axes[x]))+
				# 	' spectrum axis length: '+str(len(Texp_spectra[x])))
				return None
	for x in range(4):
		if len(vel_axes[x]) != len(tau_spectra[x]):
			# print('Provided axes (' + str(x) + ') lengths do not match')
			# print('\tvel axis length: '+str(len(vel_axes[x]))+
			# 	' spectrum axis length: '+str(len(tau_spectra[x])))
			return None

	# initialise data dictionary
	full_data = {'source_name': source_name, 
		'vel_axes':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'tau_spectrum':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'tau_rms':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'Texp_spectrum':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'Texp_rms':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'Tbg':{'1612':[],'1665':[],'1667':[],'1720':[]},
		'sig_vel_ranges':[[]]}

		###############################################
		#                                             #
		#   Load Data into 'data' dictionary object   #
		#                                             #	
		###############################################
	
	if len(tau_rms) == 0:
		full_data['tau_rms']['1612'] = findrms(tau_spectra[0])
		full_data['tau_rms']['1665'] = findrms(tau_spectra[1])
		full_data['tau_rms']['1667'] = findrms(tau_spectra[2])
		full_data['tau_rms']['1720'] = findrms(tau_spectra[3])
	else:
		full_data['tau_rms']['1612'] = tau_rms[0]
		full_data['tau_rms']['1665'] = tau_rms[1]
		full_data['tau_rms']['1667'] = tau_rms[2]
		full_data['tau_rms']['1720'] = tau_rms[3]

	full_data['vel_axes']['1612']		= vel_axes[0]
	full_data['tau_spectrum']['1612']	= tau_spectra[0]

	full_data['vel_axes']['1665']		= vel_axes[1]
	full_data['tau_spectrum']['1665']	= tau_spectra[1]

	full_data['vel_axes']['1667']		= vel_axes[2]
	full_data['tau_spectrum']['1667']	= tau_spectra[2]

	full_data['vel_axes']['1720']		= vel_axes[3]
	full_data['tau_spectrum']['1720']	= tau_spectra[3]

	if len(Texp_spectra) != 0: #absorption and emission spectra are available 
			# (i.e. on-off observations)
		if len(Texp_rms) == 0:
			full_data['Texp_rms']['1612'] = findrms(Texp_spectra[0])
			full_data['Texp_rms']['1665'] = findrms(Texp_spectra[1])
			full_data['Texp_rms']['1667'] = findrms(Texp_spectra[2])
			full_data['Texp_rms']['1720'] = findrms(Texp_spectra[3])
		else:
			full_data['Texp_rms']['1612'] = Texp_rms[0]
			full_data['Texp_rms']['1665'] = Texp_rms[1]
			full_data['Texp_rms']['1667'] = Texp_rms[2]
			full_data['Texp_rms']['1720'] = Texp_rms[3]
		
		full_data['Tbg']['1612']				= Tbg[0]
		full_data['Tbg']['1665']				= Tbg[1]
		full_data['Tbg']['1667']				= Tbg[2]
		full_data['Tbg']['1720']				= Tbg[3]

		full_data['Texp_spectrum']['1612']	= Texp_spectra[0]
		full_data['Texp_spectrum']['1665']	= Texp_spectra[1]
		full_data['Texp_spectrum']['1667']	= Texp_spectra[2]
		full_data['Texp_spectrum']['1720']	= Texp_spectra[3]


	###############################################
	#                                             #
	#         Identify significant ranges         #
	#                                             #
	###############################################
	if len(sig_vels)==0:
		findranges(full_data, sigma_tolerance, num_chan)
		print([source_name,full_data['sig_vel_ranges']])
	else:
		full_data['sig_vel_ranges']=sig_vels
	
	###############################################
	#                                             #
	#                Fit gaussians                #
	#                                             #
	###############################################
	print('Placing Gaussians')
	final_p=placegaussians(full_data,Bayes_threshold,quiet,seed,
		extra_gaussians=extra_gaussians,lfwhm_mean=lfwhm_mean,
		lfwhm_sig=lfwhm_sig)

	return final_p

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

def findranges(data, sigma_tolerance, num_chan): 
	'''
	Identifies velocity ranges likely to contain features. Adds 
	'sig_vel_ranges' (velocity ranges that satisfy a significance test) to the 
	'data' dictionary.	

	Parameters:
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		sigma_tolerance (float): signal to noise ratio threshold for a channel 
			to be identified as 'significant'.
		num_chan (int): number of consecutive channels that must be 
			'significant' to be reported order to check for significance
	Returns:
		dict: 'data' dictionary with updated 'sig_vel_ranges' keyword
	'''
	
	num_spec=4
	vel_axes = [data['vel_axes']['1612'], data['vel_axes']['1665'], 
				data['vel_axes']['1667'], data['vel_axes']['1720']]
	spectra = [	data['tau_spectrum']['1612'], data['tau_spectrum']['1665'], 
				data['tau_spectrum']['1667'], data['tau_spectrum']['1720']]	
	spec_rmss = [data['tau_rms']['1612'], data['tau_rms']['1665'], 
				data['tau_rms']['1667'], data['tau_rms']['1720']]	
	if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		num_spec+=4
		vel_axes += [data['vel_axes']['1612'], data['vel_axes']['1665'], 
					data['vel_axes']['1667'], data['vel_axes']['1720']]
		spectra += [data['Texp_spectrum']['1612'], 
					data['Texp_spectrum']['1665'], 
					data['Texp_spectrum']['1667'], 
					data['Texp_spectrum']['1720']]
		spec_rmss += [data['Texp_rms']['1612'], data['Texp_rms']['1665'], 
					data['Texp_rms']['1667'], data['Texp_rms']['1720']]
	sig_vel_list=[]
	for spec in range(num_spec):
		vel_axis=vel_axes[spec]
		spectrum=spectra[spec]
		spec_rms=spec_rmss[spec]
		test_spec=[1 if x>=sigma_tolerance*spec_rms else -1 if 
			x<=-sigma_tolerance*spec_rms else 0 for x in spectrum]
		for c in range(int(len(vel_axis)-num_chan)):
			if (np.sum(test_spec[c:c+num_chan]) >= num_chan) or (np.sum(
				test_spec[c:c+num_chan]) <= -num_chan):
				for x in range(num_chan):
					if sig_vel_list == []:
						sig_vel_list=[vel_axis[int(c+x)]]
					else:
						sig_vel_list+=[vel_axis[int(c+x)]]
	
	[av_chan,minv,maxv]=[np.mean([np.abs(v[1]-v[0]) for v in vel_axes]),
		np.max([np.min(v) for v in vel_axes]),
		np.min([np.max(v) for v in vel_axes])]
	
	sig_vel_ranges = reducelist(sig_vel_list,2*num_chan*av_chan,minv,maxv)
	# print(str([data['source_name'],sig_vel_ranges]))
	data['sig_vel_ranges'] = sig_vel_ranges
	return data
def findrms(spectrum): # returns rms
	'''
	Calculates the root mean square of 'spectrum'. This is intended as a 
	measure of noise only, so rms is calculated for several ranges, and the 
	median is returned.

	Parameters:
		spectrum (array-like): 1d array of values for which to compute rms
	Returns:
		float: Median rms of those calculated 
	'''
	# print(spectrum)
	a=len(spectrum)/20
	a=int(a-a%1)
	rms_list=[np.std(spectrum[int(20*(x-1)):int(20*(x-1)+30)]) 
		for x in range(1,a)]
	return np.median(rms_list)
def reducelist(vlist, spacing,minv,maxv):
	'''
	Takes a list of velocities with 'significant' signal (as determined by 
	'findranges') and identifies places where the spectra can be safely 
	segmented in order to minimise the length of spectra fit by 
	'placegaussians'.

	Parameters:
		vlist (array-like): unsorted list of velocity channels at which 
			'significant' signal exists in at least one spectrum.
		spacing (float): extent in velocity (km/s) required to be 
			'insignificant' in order to allow spectra to be 'cut'. This should 
			equate to some small multiple (i.e. 2) of the 'num_chan' parameter 
			in 'findranges'.
	Returns:
		vlist_output (2D list): list of min and max velocity of each spectrum 
			segment, in the form [[v1min,v1max],...,[vnmin,vnmax]].
	'''
	vlist=sorted(vlist)

	cut_list=[(vlist[x]+vlist[x+1])/2 for x in range(int(len(vlist)-1)) if 
		(vlist[x+1]-vlist[x])>=spacing]
	if len(cut_list) == 0:
		cut_list=[minv,maxv]
	else:
		if cut_list[0] >= minv:
			cut_list = [minv] + cut_list
		if cut_list[-1] <= maxv:
			cut_list = cut_list + [maxv]
	vlist_output=[[cut_list[x],cut_list[x+1]] for x in 
		range(int(len(cut_list)-1))]
	return vlist_output

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

def placegaussians(full_data, Bayes_threshold, quiet,  
	seed, extra_gaussians=3,lfwhm_mean=None,lfwhm_sig=None,logN1_mean=None,logN1_sigma=None): 
	'''
	Manages the process of placing and evaluating gaussian features. 

	Parameters:
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		Bayes_threshold (float): Bayes Factor required for a model to be 
			preferred
		quiet (boolean): Whether or not to print outputs to terminal
		best_result (str): Which result from the markov chain is to be reported
			as the best result 'median', 'median_actual', 'max' (note, all are 
			printed if quiet = False). 'median' returns the median position of 
			each parameter (may not correspond to an actual set of parameters 
			measured), 'median_actual' finds the point in the flattened chain 
			closest to the medians of each parameter and compares the lnprob 
			here to that of the actual medians, returns the one with better 
			lnprob, and 'max' returns the set of parameters with the highest 
			lnprob.
	Returns:
		array: Full list of accepted parameters with upper and lower bounds of
			credibile intervals. Format: [[1min, 1best, 1max], [2min, 2best, 
			2max],...] for parameters 1, 2, etc.
	'''
	
	accepted_full = []
	for vel_range in full_data['sig_vel_ranges']:
		print('current vel range = '+str(vel_range))
		last_accepted_full = []
		[min_vel, max_vel] = vel_range
		data = trimdata(full_data, min_vel, max_vel)
		num_gauss = 1
		keep_going = True
		extra = 0
		null_evidence = nullevidence(data)
		print(str([data['source_name'],vel_range,null_evidence]))
		prev_evidence = null_evidence
		evidences = [prev_evidence]
		while keep_going == True:
			nwalkers = nwalkers_mult * num_gauss
			# generate p0
			print('Generating initial positions')
			p0 = p0gen(vel_range,num_gauss,data,nwalkers,
					seed)
			if p0 != []: # assuming p0 didn't fail:
				print('Sampling posterior for '+str(num_gauss)+' Gaussians')
				(chain, lnprob_) = sampleposterior(data,num_gauss,p0,vel_range,
					nwalkers,seed,accepted_full)
				if len(chain) != 0: # assuming sampleposterior didn't fail:
					(current_full, current_evidence) = bestparams(chain,
						lnprob_,data,vel_range,num_gauss,quiet,
						lfwhm_mean,lfwhm_sig)
					evidences += [current_evidence]
					# if current is better than previous
					if current_evidence - prev_evidence > np.log(Bayes_threshold):
						extra = 0
						last_accepted_full = current_full
						prev_evidence = current_evidence
						num_gauss += 1
					elif extra < extra_gaussians: # if current isn't better, try some more
						print('That one wasn\'t good, trying another.')
						extra += 1
						num_gauss += 1
					else:
						keep_going = False
				else: 
					print('sampleposterior failed (but p0 is ok)')
					keep_going = False
			else: # try one more time
				p0 = p0gen(vel_range, num_gauss, data, nwalkers,
					seed)
				if p0 != []:
					(chain, lnprob_) = sampleposterior(data,num_gauss,p0,
						vel_range,nwalkers,seed,accepted_full)
					if len(chain) == 0: # sampleposterior failed
						print('sampleposterior failed (and p0 not ok)')
						keep_going = False
				else: # p0 failed a second time
					print('p0gen failed again')
					keep_going = False
		accepted_full = list(itertools.chain(accepted_full, 
			last_accepted_full))
	return accepted_full
def trimdata(data, min_vel, max_vel):
	'''
	Trims the spectra within the 'data' dictionary to the velocity ranges 
	given by min_vel and max_vel.

	Parameters:
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		min_vel (float): minimum spectrum x-axis value (eg. velocity, freq)
		max_vel (float): maximum spectrum x-axis value (eg. velocity, freq)
	Returns:
		dict: Dictionary containing trimmed spectra
	'''
	data_temp = copy.deepcopy(data)

	for f in ['1612','1665','1667','1720']:
		vel = np.array(data_temp['vel_axes'][f])
		tau = np.array(data_temp['tau_spectrum'][f])
		
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			Texp = np.array(data_temp['Texp_spectrum'][f])

		mini = np.amin([np.argmin(np.abs(vel - min_vel)), 
			np.argmin(np.abs(vel - max_vel))])
		maxi = np.amax([np.argmin(np.abs(vel - min_vel)), 
			np.argmin(np.abs(vel - max_vel))])

		data_temp['vel_axes'][f] = vel[mini:maxi + 1]
		data_temp['tau_spectrum'][f] = tau[mini:maxi + 1]

		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			data_temp['Texp_spectrum'][f] = Texp[mini:maxi + 1]
	data_temp['tau_ranges'] = [
		[-5*np.abs(np.amin(data_temp['tau_spectrum']['1612'])), 
		5*np.abs(np.amax(data_temp['tau_spectrum']['1612']))], 
		[-5*np.abs(np.amin(data_temp['tau_spectrum']['1665'])), 
		5*np.abs(np.amax(data_temp['tau_spectrum']['1665']))], 
		[-5*np.abs(np.amin(data_temp['tau_spectrum']['1667'])), 
		5*np.abs(np.amax(data_temp['tau_spectrum']['1667']))], 
		[-5*np.abs(np.amin(data_temp['tau_spectrum']['1720'])), 
		5*np.abs(np.amax(data_temp['tau_spectrum']['1720']))]]
	return data_temp
def nullevidence(data): 
	'''
	Calculates the evidence of the null model: a flat spectrum.
	
	Parameters:
		data (dict): dictionary of trimmed spectra
	Returns:
		float: Natural log of the evidence of the null model.
	'''
	lnllh = 0
	for f in ['1612','1665','1667','1720']:
		model = np.zeros(len(data['tau_spectrum'][f]))
		lnllh_tau = lnlikelihood(model, data['tau_spectrum'][f], 
			data['tau_rms'][f])
		lnllh += lnllh_tau
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			lnllh_Texp = lnlikelihood(model, data['Texp_spectrum'][f], 
				data['Texp_rms'][f])
			lnllh += lnllh_Texp
	# print('null evidence = '+str(lnllh))
	return lnllh	
# @jit
def lnlikelihood(model, spectrum, sigma):
	'''
	Calculates the likelihood of a spectrum given a model and spectrum noise.

	Parameters:
		model (array-like): Spectrum generalted by the model
		spectrum (array-like): Observed spectrum
		sigma (float): rms noise level of observed spectrum
	Returns:
		float: Natural log of the likelihood
	''' 
	N = len(spectrum)
	sse = np.sum((np.array(model) - np.array(spectrum))**2.)
	return -N*np.log(sigma*np.sqrt(2.*np.pi))-(sse/(2.*(sigma**2.)))
def bestparams(chain, lnprob_, data, vel_range, num_gauss, quiet, best_result,
	N_range,lfwhm_mean=None,lfwhm_sig=None): # 
	'''
	Finds the 'best' of a Markov chain, and the inner 68th quantile range of 
	each parameter (~+/- 1 sigma range).
	Parameters:
		chain (array-like): flattened markov chain
		lnprob (array-like): flattened lnprob
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		quiet (boolean): prevents printing of evidence and results to terminal
		best_result (str): Which result from the markov chain is to be reported
			as the best result 'median', 'median_actual', 'max' (note, all are 
			printed if quiet = False). 'median' returns the median position of 
			each parameter (may not correspond to an actual set of parameters 
			measured), 'median_actual' finds the point in the flattened chain 
			closest to the medians of each parameter and compares the lnprob 
			here to that of the actual medians, returns the one with better 
			lnprob, and 'max' returns the set of parameters with the highest 
			lnprob.
		N_range (tuple): range of log column density expected. Values in this 
			range will have a flat prior, the prior for values outside it will 
			drop off exponentially.
	Returns:
		tuple: Array of three values for each parameter: lower sigma bound, 
		median, upper sigma bound, and natural log of the evidence.
	'''
	
	if (best_result != 'median' and best_result != 'median_actual' and 
		best_result != 'max'):
		# print('Warning! \'best_result\' incorrectly set.'+
		# 	'\nSetting best_result = \'median\'')
		best_result = 'median'

	num_steps = len(chain)
	num_param = len(chain[0])

	final_darray = [list(reversed(sorted(lnprob_)))]

	for param in range(num_param):
		param_chain = [chain[x][param] for x in range(num_steps)]
		zipped = sorted(zip(param_chain, lnprob_))
		sorted_param_chain, sorted_lnprob = zip(*zipped)
		dparam_chain = [0] + [sorted_param_chain[x] - sorted_param_chain[x-1] 
			for x in range(1, len(sorted_param_chain))]
		sorted_dparam_chain = [[x for _,x in 
			list(reversed(sorted(zip(sorted_lnprob, dparam_chain))))]]
		final_darray = np.concatenate((final_darray, sorted_dparam_chain), 
			axis = 0) 

	total_evidence = -np.inf
	for step in range(num_steps):
		# multiply all dparam values
		param_volume = 1
		for param in range(1, len(final_darray)):
			param_volume *= final_darray[param][step]
		if param_volume != 0:
			contribution_to_lnevidence = (np.log(param_volume) + 
				final_darray[0][step])
			total_evidence = np.logaddexp(
				total_evidence, contribution_to_lnevidence)

	lower = np.percentile(chain,16,axis=0)
	upper = np.percentile(chain,84,axis=0)
	median = np.percentile(chain,50,axis=0)
		
	median_lnprob = lnprob(median, data, vel_range, num_gauss,[],N_range,
		lfwhm_mean,lfwhm_sig,False)
	
	dmedian = [sum([(x[y]-median[y])**2. for y in range(num_param)]) 
		for x in chain]
	median_actual = chain[np.argmin(dmedian)]
	median_actual_lnprob = lnprob_[np.argmin(dmedian)]

	[max_lnprob, max_] = [x[-1] for x in zip(*sorted(list(zip(lnprob_,chain)), 
		key=lambda tup: tup[0]))]

	final_results_median = [[lower[x], median[x], upper[x]] 
		for x in range(num_param)]
	if median_actual_lnprob < median_lnprob:
		final_results_median_actual = [[lower[x], median[x], upper[x]] 
			for x in range(num_param)]		
	else:		
		final_results_median_actual = [[lower[x], median_actual[x], upper[x]] 
			for x in range(num_param)]
	final_results_max = [[lower[x], max_[x], upper[x]] 
		for x in range(num_param)]

	
	spectra = [	data['tau_spectrum']['1612'], 
				data['tau_spectrum']['1665'], 
				data['tau_spectrum']['1667'], 
				data['tau_spectrum']['1720'], 
				data['Texp_spectrum']['1612'], 
				data['Texp_spectrum']['1665'], 
				data['Texp_spectrum']['1667'], 
				data['Texp_spectrum']['1720']]
	models=makemodel([x[1] for x in final_results_median], data, num_gauss)
	colors=['blue','green','red','cyan']
	label=[r'$1612\ \mathrm{MHz}$',r'$1665\ \mathrm{MHz}$',r'$1667\ \mathrm{MHz}$',r'$1720\ \mathrm{MHz}$']
	residuals=[[spectra[x][y]-models[x][y] for y in range(len(models[x]))] 
		for x in range(len(models))]
	vels=[data['vel_axes']['1612'],data['vel_axes']['1665'],
		data['vel_axes']['1667'],data['vel_axes']['1720']]
	# fig, axes = plt.subplots(nrows = 5, ncols = 2, sharex = True, figsize=[10,10])
	# for row in range(4):
	# 	for col in range(2):
	# 		# data
	# 		axes[row,col].plot(vels[row],spectra[int(row+4*col)],color='black',
	# 			linewidth=1,zorder=0)
	# 		# models
	# 		axes[row,col].plot(vels[row],models[int(row+4*col)],linewidth=1)
	# 		# residuals
	# 		axes[4,col].plot(vels[row],spectra[int(row+4*col)]-models[int(row+4*col)],
	# 			color=colors[row],linewidth=1)
	# plt.show()
	# plt.close()

	if best_result == 'median':
		print('source: '+str(data['source_name'])+' vel range: '+str(vel_range)
			+' num gauss: '+str(num_gauss)+' evidence: '+str(total_evidence)+
			' params: '+str(final_results_median))
		return (final_results_median,total_evidence)
	elif best_result == 'median_actual':
		print('source: '+str(data['source_name'])+' vel range: '+str(vel_range)
			+' num gauss: '+str(num_gauss)+' evidence: '+str(total_evidence)+
			' params: '+str(final_results_median_actual))
		return (final_results_median_actual,total_evidence)
	elif best_result == 'max':
		print('source: '+str(data['source_name'])+' vel range: '+str(vel_range)
			+' num gauss: '+str(num_gauss)+' evidence: '+str(total_evidence)+
			' params: '+str(final_results_max))
		return (final_results_max,total_evidence)
def p0gen(vel_range, num_gauss, data, nwalkers, seed=None,lfwhm_mean=None,
	lfwhm_sig=None,logN1_mean=None,logN1_sigma=None):
	'''
	Generates initial positions of walkers for the MCMC simulation.

	Parameters:
		vel_range (array-like): allowed velocity range
		num_gauss (int): number of gaussian components in model
		data (dict): dictionary of trimmed spectra
		nwalkers (int): number of walkers
	Returns:
		array: Initial positions of all walkers.
	'''
	trial=0
	while trial<=10:
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665']!=[]:
			print('Beginning to identify initial positions')
			p0_to_return=Texpp0(vel_range, num_gauss, nwalkers, seed)
		else:
			print('Beginning to identify initial positions')
			p0_to_return=taup0(vel_range, num_gauss, data, nwalkers,seed)
		
		p0_to_return=checkp0(p0_to_return,num_gauss,data,vel_range,seed,
			lfwhm_mean,lfwhm_sig,logN1_mean,logN1_sigma)
		if p0_to_return != []:
			return p0_to_return
		trial+=1
	# print('p0gen failed')
	return []
def Texpp0(vel_range, num_gauss, nwalkers, seed):
	# parameters will be v, fwhm, N1, 1/Tex_1612, 1/Tex_1665, 1/Tex_1667
	p0_2 = 0
	vel_guesses = [sorted([np.random.uniform(vel_range[0],vel_range[1])
				for x in range(num_gauss)]) for y in range(nwalkers)]
	for walker in range(nwalkers):
		p0_1 = 0
		for comp in range(num_gauss):
			# vel, FWHM, N
			p0_0 = [vel_guesses[walker][comp],np.random.uniform(-1,1),
				np.random.normal(12.8,0.75)] + list(
					0.1*np.random.randn(3))
			if p0_1 == 0:
				p0_1 = p0_0
			else:
				p0_1 += p0_0
		if p0_2 == 0:
			p0_2 = [p0_1]
		else:
			p0_2 += [p0_1]
	return p0_2
def taup0(vel_range, num_gauss, data, nwalkers,seed):
	[t12_range,t65_range,t67_range,t20_range]=data['tau_ranges']
	
	n=num_gauss*nwalkers

	vels=np.array([np.sort(np.random.uniform(*vel_range,n))])
	lfws=np.array([np.random.normal(-0.05,0.2,n)])

	t12s=np.random.uniform(*t12_range,n)
	t65s=np.random.uniform(*t65_range,n)
	t67s=np.random.uniform(*t67_range,n)
	t20s=np.random.uniform(*t20_range,n)
	taus=np.stack((t12s,t65s,t67s,t20s)).transpose()

	p1=np.concatenate((vels.T,lfws.T,taus),axis=1)
	# change into p0:
	if num_gauss!=1:
		p0=p1[0:nwalkers]
		for x in range(1,num_gauss):
			p0=np.concatenate((p0,p1[int(x*nwalkers):int((x+1)*nwalkers)]),axis=1)
		return p0
	return p1
def checkp0(p0,num_gauss,data,vel_range,seed=None,lfwhm_mean=None,lfwhm_sig=None,
	logN1_mean=None,logN1_sigma=None):
	nwalkers=len(p0)
	probs=[lnprob(x,data,vel_range,num_gauss,[],lfwhm_mean,lfwhm_sig,
		logN1_mean,logN1_sigma,False) for x in p0]
	if probs.count(-np.inf) == nwalkers:
		# print('all p0 values return -inf :(')
		return []
	elif probs.count(-np.inf) == 0:
		return p0

	p0=[p0[x] for x in range(nwalkers) if probs[x]!=-np.inf]
	iterations=0

	while len(p0)<nwalkers and iterations < 1e4:
		iterations+=1
		needed = nwalkers-len(p0)
		if needed >len(p0):
			new_ind=np.random.choice(range(len(p0)), needed, replace = True)
		else:
			new_ind=np.random.choice(range(len(p0)), needed, replace = False)
		new=[p0[x] for x in new_ind]
		new=[[x*(0.01*(np.random.randn()) + 1) for x in y] for y in new]
		new_probs=[lnprob(x,data,vel_range,num_gauss,[],lfwhm_mean,lfwhm_sig,
			logN1_mean,logN1_sigma,False) for x in new]
		new=[new[x] for x in range(len(new)) if new_probs[x]!=-np.inf]
		if len(new)!=0:
			p0+=new
		# else:
			# print('new p0 values all failed :(')
			
	if iterations >= 1e4:
		# print('Error! checkp0 stopped because of too many iterations. Check p0gen.')
		return []
	# else:
	# 	print('p0 generated successfully!')
	return p0
# @jit
def texp(tau,Tbg,Tex): 
	'''
	Calculates the expected brightness temperature given the excitation 
	temperature, background brightness temperature and optical depth.

	Parameters:
		tau (float): optical depth
		Tbg (float): background brightness temperature (i.e. synchrotron + 
			CMB) (kelvin)
		Tex (float): excitation temperature (kelvin)
	Returns:
		float: Expected brightness temperature in kelvin
	'''
	return (Tex-Tbg)*(1-np.exp(-tau))
# @jit
def tau3(tau_1612=None, tau_1665=None, tau_1667=None, tau_1720=None): 
	'''
	Applies the OH optical depth sum rule:
		tau_1612 + tau_1720 = tau_1665/5 + tau_1667/9
	to either:
		a set of 3 optical depths to find the fourth, or
		a set of 4 optical depths to evaluate adherance to the sum rule

	Parameters:
		tau_[freq] (float): Optical depth of OH ground 
			rotational state transition
	Returns:
		list: List of all four optical depths, or prints residual of sum rule
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
		# print('4 values of tau provided, confirming adherence to sum rule.')
		sum_res = np.abs(tau_1665/5 + tau_1667/9 - tau_1612 - tau_1720)
		# print('Residual of sum rule = ' + str(sum_res))
		# print('Percentages of supplied tau values: ' + 
		# 	str(np.abs(sum_res/tau_list)))
	else: # can't do anything
		# print('Error, at least 3 values of tau needed to apply the sum rule.')
		return None
def Tex3(Tex_1612=None, Tex_1665=None, Tex_1667=None, Tex_1720=None): 
	'''
	Applies the OH excitation temperature sum rule:
		nu_1612/Tex_1612+nu_1720/Tex_1720 = nu_1665/Tex_1665+nu_1667/Tex_1667
	to a set of 3 excitation temperatures to find the fourth.

	Parameters:
		Tex_[freq] (float): Excitation temperature of OH ground 
			rotational state transition
	Returns:
		list: List of all four excitation temperatures
	'''
	[nu_1612,nu_1665,nu_1667,nu_1720]=[1.612231e9,
		1.665402e9,1.667359e9,1.720530e9]
	Tex_list = np.array([Tex_1612, Tex_1665, Tex_1667, Tex_1720])
	if (Tex_list == None).sum() == 1: # only one is left blank, can proceed
		if Tex_1612 == None:
			Tex_1612=nu_1612/((nu_1665/Tex_1665)+(nu_1667/Tex_1667)-
				(nu_1720/Tex_1720))
		elif Tex_1665 == None:
			Tex_1665=nu_1665/((nu_1612/Tex_1612)+(nu_1720/Tex_1720)-
				(nu_1667/Tex_1667))
		elif Tex_1667 == None:
			Tex_1667=nu_1667/((nu_1612/Tex_1612)+(nu_1720/Tex_1720)-
				(nu_1665/Tex_1665))
		elif Tex_1720 == None:
			Tex_1720=nu_1720/((nu_1665/Tex_1665)+(nu_1667/Tex_1667)-
				(nu_1612/Tex_1612))

		return np.array([Tex_1612, Tex_1665, Tex_1667, Tex_1720])
	else: # can't do anything
		# print('Error, at least 3 values of Tex needed to apply the sum rule.')
		return None
# @jit
def tauTexN(logN1,invTex_1612,invTex_1665,invTex_1667,logfwhm): 
	'''
	Calculates optical depths and excitation temperatures from the column 
	densities of the four ground rotaitonal state levels of OH.

	Parameters:
		logN1 (float): log (base 10) of the column density of the 
			lowest ground rotational state level of OH.
		invTex_[1612,1665,1667] (float): 1/Tex for the 1612, 1665 and 1667 MHz 
			lines of OH.
		fwhm (float): Full width at half-maximum of the gaussian feature 
			(km/sec)
	Returns:
		list: [tau_1612, tau_1665, tau_1667, tau_1720, Tex_1612, Tex_1665, 
		Tex_1667, Tex_1720]
	'''
	fwhm=10**logfwhm
	c1 = [0.6,1,1,5./3.]
	c2 = [-0.0773748694136189,-0.0799266744475076,-0.0800205956160266,-0.0825724006499154]
	c3 = [4130292895.686540,799525863.354490,739481982.512962,6449426425.933550]

	logN3=np.log10(c1[1]*np.exp(c2[1]*invTex_1665))+logN1
	logN2=logN3-np.log10(c1[0]*np.exp(c2[0]*invTex_1612))
	logN4=np.log10(c1[2]*np.exp(c2[2]*invTex_1667))+logN2

	Tex_1612 = 1/invTex_1612
	Tex_1665 = 1/invTex_1665
	Tex_1667 = 1/invTex_1667
	Tex_1720 = c2[3]/(np.log((10**logN4)/(c1[3]*(10**logN1))))

	tau_1612 = (c1[0]/c3[0])*(10**logN2)/(Tex_1612*fwhm*1e5)#convert to cm/s
	tau_1665 = (c1[1]/c3[1])*(10**logN1)/(Tex_1665*fwhm*1e5)#convert to cm/s
	tau_1667 = (c1[2]/c3[2])*(10**logN2)/(Tex_1667*fwhm*1e5)#convert to cm/s
	tau_1720 = (c1[3]/c3[3])*(10**logN1)/(Tex_1720*fwhm*1e5)#convert to cm/s

	return [tau_1612, tau_1665, tau_1667, tau_1720, 
			Tex_1612, Tex_1665, Tex_1667, Tex_1720]
# makes/plots model from params
def makemodel(params, data, num_gauss, accepted_params = []): 
	'''
	Generates model spectra from input parameters.	

	Parameters:
		params (array-like): parameters for the models. If 'data' contains tau 
			and Texp spectra, these will be [mean vel(1), fwhm(1), logN1(1), 
			1/Tex_1612(1), 1/Tex_1665(1), 1/Tex_1667(1), ...(num_gauss)] for 
			'num_gauss' gaussian components. If 'data'contains only tau 
			spectra, these will be [mean vel(1), fwhm(1), tau_1(1), tau_2(1), 
			tau_3(1), tau_4(1), ...(num_gauss)].
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		num_gauss (int): number of gaussian components.	
	Keyword Arguments:
		accepted_params (array-like): parameters of any previously accepted 
			components i.e. from another velocity range. The same form as 
			'params'. These are included to account for the tail of a 
			neighbouring gaussian that may influence the current velocity 
			range.	
	Returns:
		tuple: Tuple of model spectra. If 'data' contains tau and Texp 
			spectra, these will be (tau_1612, tau_1665, tau_1667, tau_1720, 
			tau_1612, tau_1665, tau_1667, tau_1720). If 'data' 
			contains only tau spectra, these will be [mean vel(1), fwhm(1), 
			tau_1(1), tau_2(1), tau_3(1), tau_4(1), ...(num_gauss)]. 
	'''
	# initialise models
	if accepted_params != []:
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			(tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, 
				Texp_m_1612, Texp_m_1665, 
				Texp_m_1667, Texp_m_1720) = makemodel(accepted_params, data, 
					int(len(accepted_params) / 6))
		else:
			(tau_m_1612, tau_m_1665, 
				tau_m_1667, tau_m_1720) = makemodel(accepted_params, data, 
					int(len(accepted_params) / 6))
	else:
		vel_1612 = data['vel_axes']['1612']
		vel_1665 = data['vel_axes']['1665']
		vel_1667 = data['vel_axes']['1667']
		vel_1720 = data['vel_axes']['1720']

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
	if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		for comp in range(int(num_gauss)): 
			[vel,logFWHM,logN1,invTex_1612,invTex_1665,
				invTex_1667]=params[comp*num_params:(comp+1)*num_params]
			[tau_1612, tau_1665, tau_1667, tau_1720, 
				Tex_1612,Tex_1665,Tex_1667,Tex_1720]=tauTexN(logN1,
					invTex_1612,invTex_1665,invTex_1667,logFWHM)
			[Texp_1612, Texp_1665, Texp_1667, Texp_1720] = [
				texp(tau_1612,data['Tbg']['1612'],Tex_1612), 
				texp(tau_1665,data['Tbg']['1665'],Tex_1665), 
				texp(tau_1667,data['Tbg']['1667'],Tex_1667), 
				texp(tau_1720,data['Tbg']['1720'],Tex_1720)]
			tau_m_1612 += gaussian(vel, 10**logFWHM, tau_1612)(np.array(vel_1612))
			tau_m_1665 += gaussian(vel, 10**logFWHM, tau_1665)(np.array(vel_1665))
			tau_m_1667 += gaussian(vel, 10**logFWHM, tau_1667)(np.array(vel_1667))
			tau_m_1720 += gaussian(vel, 10**logFWHM, tau_1720)(np.array(vel_1720))
			Texp_m_1612 += gaussian(vel, 10**logFWHM, Texp_1612)(np.array(vel_1612))
			Texp_m_1665 += gaussian(vel, 10**logFWHM, Texp_1665)(np.array(vel_1665))
			Texp_m_1667 += gaussian(vel, 10**logFWHM, Texp_1667)(np.array(vel_1667))
			Texp_m_1720 += gaussian(vel, 10**logFWHM, Texp_1720)(np.array(vel_1720))
	else:
		for comp in range(int(num_gauss)): 
			[vel, logFWHM, tau_1612, tau_1665, tau_1667, tau_1720] = params[comp * 
				num_params:(comp + 1) * num_params]
			tau_m_1612 += gaussian(vel, 10**logFWHM, tau_1612)(np.array(vel_1612))
			tau_m_1665 += gaussian(vel, 10**logFWHM, tau_1665)(np.array(vel_1665))
			tau_m_1667 += gaussian(vel, 10**logFWHM, tau_1667)(np.array(vel_1667))
			tau_m_1720 += gaussian(vel, 10**logFWHM, tau_1720)(np.array(vel_1720))

	# return models
	if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720, 
			Texp_m_1612, Texp_m_1665, Texp_m_1667, Texp_m_1720)	
	else:
		return (tau_m_1612, tau_m_1665, tau_m_1667, tau_m_1720)
def gaussian(mean, FWHM, height): 
	'''
	Generates a gaussian profile with the given parameters. Mean must be 
		provided, as well as FWHM or sigma, and height or amp. Must be called 
		with a 'x' axis array (i.e. velocity axis)

	Parameters:
		mean (float): mean of gaussian (line centre)
	Keyword Arguments:
		FWHM (float): full width at half maximum
		height (float): maximum 'y' value
		sigma (float): standard deviation
		amp (float): integral of full gaussian profile (= height * sigma * 
			sqrt(2*pi))
	Returns:
		lambda: Value of gaussian function at 'x'
	'''
	return lambda x: height * np.exp(-((x - mean)**2.) / ((FWHM**2.) / (4. * np.log(2.))))
# sample posterior using emcee
def sampleposterior(data, num_gauss, p0, vel_range, nwalkers, seed,accepted): 
	'''
	Samples the posterior probability distribution.

	Parameters:
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		num_gauss (int): number of gaussian components.
		p0 - (numpy array) initial positions of all walkers.
		vel_range (array-like): allowed velocity range
		nwalkers (int) the number of walkers. Must be even.
		accepted_params (array-like): parameters of any previously accepted 
			components (i.e. from another velocity range). If 'data' contains 
			tau and Texp spectra, these will be [mean vel(1), fwhm(1), N1(1), 
			N2(1), N3(1), N4(1), ...(num_gauss)] for 'num_gauss' gaussian 
			components. If 'data'contains only tau spectra, these will be 
			[mean vel(1), fwhm(1), tau_1(1), tau_2(1), tau_3(1), tau_4(1), 
			...(num_gauss)]. These are included to account for the tail of a 
			neighbouring gaussian that may influence the current velocity range.
	Returns:
		tuple: Two array-like: (2D array, 1D array). First is the flattened 
			Markov chain: the positions in parameter space visited by all walkers 
			for all iterations. Second is the flattened ln probability chain: the 
			value of the posterior probability distribution at all the points in 
			parameter space indicated by the flattened Markov chain.
	'''
	print('Starting sampleposterior')
	ndim = 6 * num_gauss
	
	if accepted != []:
		accepted_params = []
	else:
		accepted_params = [x[1] for x in accepted]

	burn_iterations = burn_iter
	final_iterations = final_iter
	
	args = [data,vel_range,num_gauss,accepted_params,False]
	a=3
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = args, 
			a=a)
	print('Starting burn phase')
	# burn
	[burn_run, test_result] = [0, False]
	while burn_run <= 3 and not test_result:
		try:
			# sampler.reset()
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations,rstate0=seed)
		except ValueError: # sometimes there is an error within emcee
			# print('emcee is throwing a value error for p0. Running again.')
			pos, prob, state = sampler.run_mcmc(p0, burn_iterations,rstate0=seed)
		# test convergence
		(test_result, p0) = convergencetest(sampler_chain=sampler.chain,
			num_gauss=num_gauss,seed=seed,data=data,vel_range=vel_range)
		# print('Converged? '+str(test_result))
		burn_run += 1
	# plot burned chains
	# for p in range(ndim):
	# 	plt.figure()
	# 	plt.title('Burning param num '+str(p))
	# 	for w in range(nwalkers):
	# 		plt.plot(range(sampler.chain.shape[1]),sampler.chain[w,:,p])
	# 	plt.show()
	# 	plt.close()

	print('Starting final phase')
	sampler.reset()
	sampler.run_mcmc(pos, final_iterations,rstate0=seed)
	flatchain = sampler.flatchain
	flatlnprob = sampler.flatlnprobability
	# pickle.dump(flatchain,open(data['source_name']+'_'+str(vel_range[0])+'_'
	# 	+str(num_gauss)+'.pickle','wb'))
	chain = [flatchain[x] for x in range(len(flatchain)) 
		if flatlnprob[x] != -np.inf]
	# plot final chains
	# for p in range(ndim):
	# 	plt.figure()
	# 	plt.title('Final param num '+str(p))
	# 	for w in range(nwalkers):
	# 		plt.plot(range(sampler.chain.shape[1]),sampler.chain[w,:,p])
	# 	plt.show()
	# 	plt.close()

	if np.array(chain).shape[0] == 0:
		# print('No finite members of posterior!')
		# pickle.dump([data,vel_range,sampler.chain],open('No_finite_post.pickle','wb'))
		return (np.array([]), np.array([]))
	else:
		lnprob_ = [x for x in flatlnprob if x != -np.inf]
		return (np.array(chain), np.array(lnprob_))
def lnprob(lnprobx,data,vel_range,num_gauss,accepted_params,
	lfwhm_mean=None,lfwhm_sig=None,logN1_mean=None,logN1_sigma=None,final=True):
	'''
	Returns the value of the posterior probability at the point in parameter 
	space given by 'lnprobx'.

	Parameters:
		lnprobx (array-like): 1D position vector in parameter space assigned by 
			emcee
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		vel_range (array-like): allowed velocity range
		num_gauss (int): number of gaussian components
		accepted_params (array-like): parameters of any previously accepted 
			components (i.e. from another velocity range). If 'data' contains 
			tau and Texp spectra, these will be [mean vel(1), fwhm(1), N1(1), 
			N2(1), N3(1), N4(1), ...(num_gauss)] for 'num_gauss' gaussian 
			components. If 'data'contains only tau spectra, these will be 
			[mean vel(1), fwhm(1), tau_1(1), tau_2(1), tau_3(1), tau_4(1), 
			...(num_gauss)]. These are included to account for the tail of a 
			neighbouring gaussian that may influence the current velocity range.
	Returns:
		float: natural log of the posterior probability distribution at the 
		sampled point in parameter space.
	'''
	
	lprob = lnprprior(data,lnprobx,vel_range,num_gauss,lfwhm_mean,lfwhm_sig,
		logN1_mean,logN1_sigma)
	if np.isnan(lprob):
		return -np.inf
	params = lnprobx
	models = makemodel(params, data, num_gauss, accepted_params)
	if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
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
	for a in range(len(spectra)):
		llh = lnlikelihood(models[a], spectra[a], rms[a])
		if np.isnan(llh):
			return -np.inf
		lprob += llh
	return lprob	
def convergencetest(sampler_chain, num_gauss,seed,data,vel_range): 
	'''
	Tests the convergence of a Markov chain. If the chain has not converged, 
	will replace the last position of the walkers with a 'better' position 
	in parameter space.

	Parameters:
		sampler_chain (array-like): 3D array (i.e. not flattened) of Markov 
			chain. Dimensions: [number of walkers, number of iterations, 
			number of parameters].
		num_gauss (int): number of gaussian components
		pos (array-like): final position in parameter space
	Returns:
		tuple: Tuple with two elements: first is a boolean, second is the 
		position in parameter space from which each walker will continue.
	'''

	nwal = sampler_chain.shape[0]
	nitr = sampler_chain.shape[1]
	ndim = sampler_chain.shape[2]
	if nitr>100:
		sampler_chain=sampler_chain[:,-100:,:]
		nitr=100

	pass_fail = True

	flat_chain=sampler_chain.reshape((int(nitr*nwal),ndim))
	bin_ranges=[]
	for parameter in range(ndim):
		passing_walkers=[]
		pos=[x[parameter] for x in flat_chain]
		(density,param_value)=np.histogram(pos,bins=500,density=True)
		dparam=param_value[1]-param_value[0]
		max_density_ind=np.argmax(density)
		center_param_value=param_value[max_density_ind]+dparam/2
		param_range=[center_param_value-abs(dparam/2),
			center_param_value+abs(dparam/2)]
		if bin_ranges == []:
			bin_ranges=[param_range]
		else:
			bin_ranges+=[param_range]

		if pass_fail:
			# Check variances
			pass_fail=checkvariances(sampler_chain,0.3)
			if pass_fail:
				for walker in range(nwal):
					walker_path=sampler_chain[walker,:,parameter]
					above=int((walker_path>=param_range[0]).sum())
					below=int((walker_path<=param_range[1]).sum())
					if above != 0 and below != 0:
						if len(passing_walkers)==0:
							passing_walkers=[walker]
						else:
							passing_walkers+=[walker]
				if len(passing_walkers)/nwal < 0.8:
					pass_fail = False
	if pass_fail:
		return (True, sampler_chain[:,-1,:])
	else:
		# change order of bin_ranges so they're in ascending order of v
		npar=int(ndim/num_gauss)
		vel_ranges=bin_ranges[::npar]
		vel_mins=[x[0] for x in vel_ranges]
		bin_ranges_old=bin_ranges
		a=0
		for g in np.argsort(vel_mins):
			bin_ranges[int(a):int(a+npar)]=bin_ranges_old[int(g*npar):int(
				g*npar+npar)]
			a+=npar
		new_p0=np.array([np.random.uniform(bin_ranges[x][0],bin_ranges[x][1],
			nwal) for x in range(ndim)])
		new_p0=new_p0.transpose()
		if len(checkp0(new_p0,num_gauss,data,vel_range))==0:
			return (False, sampler_chain[:,-1,:])
		return (False, new_p0)
def checkvariances(chain, tolerance):
	if tolerance<=0 or tolerance>1:
		# print('Checkvariances requires a tolerance in the range (0,1].')
		return None
	nwal = chain.shape[0]
	nitr = chain.shape[1]
	ndim = chain.shape[2]
	if nitr>100:
		chain=chain[:,-100:,:]
		nitr=100
	for dim in range(ndim):	
		var_by_step=np.zeros(nitr)
		var_by_walk=np.zeros(nwal)
		dim_ar=chain[:,:,dim]
		for walk in range(nwal):
			var_by_walk[walk]=np.var(dim_ar[walk])
		for step in range(nitr):
			var_by_step[step]=np.var(dim_ar[:,step])
		av_step=np.mean(var_by_step)
		av_walk=np.mean(var_by_walk)
		if (np.min([av_step,av_walk])/np.max([av_step,av_walk]))<tolerance:
			return False
	return True
# priors
def lnprprior(data,params,vel_range,num_gauss,N_range=[11,18],lfwhm_mean=None,
	lfwhm_sig=None,logN1_mean=12.8,logN1_sigma=0.7):
	'''
	Returns the value of the pprior probability at the point in parameter 
	space given by 'params'.

	Parameters:
		data (dict): observed spectra and other observed quantities (see 
			Main function)
		params (array-like): 1D position vector in parameter space
		vel_range (array-like): allowed velocity range
		num_gauss (int): number of gaussian components
		N_range (tuple): range of log column density expected. Values in this 
			range will have a flat prior, the prior for values outside it will 
			drop off exponentially.
	Returns:
		float: natural log of the prior probability distribution at the 
		sampled point in parameter space.
	'''
	# velocity prior
	vr=(vel_range[1]-vel_range[0])
	lnprprior = np.log(np.math.factorial(num_gauss)/((vr)**num_gauss))
	
	vel_prev = vel_range[0]
	for gauss in range(num_gauss):
		# define params
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			[vel,logFWHM,logN1,invTex_1612,invTex_1665,
				invTex_1667] = params[int(gauss * 6):int((gauss + 1) * 6)]
			[tau_1612, tau_1665, tau_1667, tau_1720, Tex_1612, Tex_1665, 
				Tex_1667, Tex_1720] = tauTexN(logN1,invTex_1612,invTex_1665,
				invTex_1667,logFWHM)
		else:
			[vel, logFWHM, tau_1612, tau_1665, tau_1667, 
				tau_1720] = params[int(gauss * 6):int((gauss + 1) * 6)]
		# calculate priors
		if (vel > vel_range[1] or vel < vel_prev):
			return -np.inf
		vel_prev=vel		
		lnprprior+=logfwhmlnprior(logFWHM,lfwhm_mean,lfwhm_sig)
		if 'Texp_spectrum' in data and data['Texp_spectrum']['1665'] != []:
			# N prior
			lnprprior+=lnNprior(logN1)
			# inv Tex priors
			lnprprior+=lninvTexprior([invTex_1612,invTex_1665,invTex_1667])
		else:
			lnprprior+=lntauprior([tau_1612,tau_1665,tau_1667,tau_1720],
				data['tau_ranges'])
	return lnprprior
def lnNprior(logN1,logN1_mean=12.5,logN1_sigma=0.75):
	'''
	This distribution is based on data from Li et al. 2018 where mean 
	logN~12.8.
	'''
	return np.log(1/(logN1_sigma*np.sqrt(2*np.pi)))-((logN1-
		logN1_mean)**2)/(2*logN1_sigma**2)
def logfwhmlnprior(logfwhm,mean,sigma):
	'''
	Calculates natural log of prior for the log value of full width at 
	half-maximum using a Gaussian distribution. Default values approximate 
	the distribution of fwhm found in GNOMES and Millenium Survey.

	Parameters:
		logfwhm (float): value of log fwhm (km/s)
	Keyword Arguments:
		mean (float)
		sigma (float)
	Returns:
		float: natural log of prior probability distribution at fwhm
	'''
	if (10**logfwhm < 0.1)|(10**logfwhm>10):
		return -np.inf
	if mean==None:
		mean=-0.5
	if sigma==None:
		sigma=0.2
	return np.log(1/(sigma*np.sqrt(2*np.pi)))-((logfwhm-mean)**2)/(2*sigma**2)
def lntauprior(taus,tau_ranges,m=[0.0075,0.0005],s=[0.012,0.0015],w=[0.7,0.3]):
	# Add some 'pressure' to adhere to the sum rule
	# multiply by a 'gaussian' that drops off based on sum rule residual - 
	# should be able to find integral of that easily enough.

	[t12,t65,t67,t20]=taus
	[t12_r,t65_r,t67_r,t20_r]=tau_ranges
	if ((t12<t12_r[0])|(t12>t12_r[1])|
		(t65<t65_r[0])|(t65>t65_r[1])|
		(t67<t67_r[0])|(t67>t67_r[1])|
		(t20<t20_r[0])|(t20>t20_r[1])):
		return -np.inf
	sumrule_resid=t12+t20-t65/5-t67/9
	sumrule_sigma=0.5 
	return (np.sum([logsumexp([np.log(w[x]/(s[x]*np.sqrt(2*np.pi)))-
		((taus[y]-m[x])**2)/(2*s[x]**2) for x in range(len(m))]) for y in 
		[0,2,3]])+np.log(1/(sumrule_sigma*np.sqrt(2*np.pi)))-
		((sumrule_resid)**2)/(2*sumrule_sigma**2))
def lninvTexprior(invTexs,m=[0.35],s=[0.4],w=[1]):
	components=np.array([[w[x]/(s[x]*np.sqrt(2*np.pi))-((invTexs[y]-m[x])**2)/
		(2*s[x]**2) for x in range(len(m))] for y in range(3)]).flatten()
	return logsumexp(components)






















































