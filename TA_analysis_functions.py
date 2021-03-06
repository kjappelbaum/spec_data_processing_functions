# Libraries

print("Loading useful functions for processing of TA data")

#Pandas, for easy import of data
import pandas as pd 
#print('Pandas version:', pandas.__version__)

#Torch, for GVD correction
import torch
print('Torch version:', torch.__version__)
from torch.nn import functional
from torch.autograd import Variable

#Numpy
import numpy as np
print('Numpy version:', np.__version__)


#Scipy
from scipy.interpolate import interp1d
import scipy.optimize as optimize
from scipy.ndimage.interpolation import shift
#print('Scipy version:', scipy.__version__)

#Sklearn
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from skimage import exposure


#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


import plotly
import plotly.offline as py
from plotly import tools
from plotly import graph_objs as go
py.init_notebook_mode(connected=True)
print('Plotly version:', plotly.__version__)
import colorlover as cl

#System tools
import glob
import os
import re # for regular expression matching

### Constants
numberpixel = 512
c = 299792458
conversion =  0.14997 #delay to ps 

usage = """ This is a collection of functions that might be helpful 
			in your data processing. Suggested usage:
			(1) import_OD() and import_delay()
			(2) run import checks
			(3) reshape into maps
			(4) plot a map, select area for baseline substraction and substract base line there
			(5) do SVD noise correction 
			(6) do GVD correction (check the units of the delays in your files and 
				use a appropiate conversion, e.g. via length and speed of light)
			(7) plot corrected map, timetraces and spectral cuts for first impressions 
			(8) do global analysis (to be implemented) or export into Optimus or PyLDM for 
				global and target analysis or LDA. 

			Extension into analysis classes is planned."""

print(usage)


'''
AUXILLARY, GENERAL FUNCTIONS
'''

def gaussian(x, amp, cen, wid):
	return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2) )
	
def FWHM(sigma):
	return 2.55*sigma

def find_closest_index(your_list, your_value):
	'''
	Find the index closest to the value.

	Input: 
		(1) List
		(2) Value

	Output: 
		(1) Index
	'''
	idx = (np.abs(your_list - your_value)).argmin()
	return idx

def angle_to_wl(angle):
	'''
	Change according to your fitting function. Converts BBO angle into 
	pump wavelength.
	'''
	return 0.0667*angle**2 -11.9*angle + 778.8


'''
IMPORT FUNCTIONS
'''

# Import OD data 
def import_OD(path, pattern, numberpixel=512, debug=False):
	'''
	Function that imports OD data at path, which is identified by
	the regular expression specified in pattern.
	Keeping track of the filenames is recommended, ordered import 
	pivotal for correct processing of anisotropy measurments

	Input:
		(1) Path, e.g. path = "Mypath/"
		(2) Regular expression to find the correct files, e.g.
		pattern = r'\w+\d+aa-\d+\.dat'
		(3) Number of pixels of your detector, default is 512.
		(4) Debugging options, if one éncounters problems 
		(e.g. with the regular expression) one can use this 
		to print the found file names 

	Output: 
		Call the function appropiatly, i.e. 
		filename_list, save_list, save_wl = import_OD(arguments...)

		(1) Ordered list of filenames
		(2) Ordered list with the scans found in the directory
		(3) Wavelengths (probe) found in the scan files. 
		Assummes currently that this is the same for all the imported 
		scans. 
	'''
	filename_list = []
	assert  (numberpixel > 0), "Where are your pixels?"
	save_temp = []
	print("Start import OD data")
	for i in os.listdir(path):
		if re.search(pattern,i):
			if debug == True:
				print(i)
			tmp = pd.read_csv(path+i, sep='\t')
			tmp.dropna() # skip empty lines
			save_temp.append(tmp)
			filename_list.append(i)
	save_wl = save_temp[1]['Wavelengths'][:numberpixel].values 
	
	print("Sorting your lists")
	filename_list, save_list = (list(t) for t in zip(*sorted(zip(filename_list, save_temp))))
	
	print("OD import completed")

	return filename_list, save_list, save_wl
	
def import_delay(path, pattern, debug=False):
	'''
	Function that imports thr delay data at path, which is identified by
	the regular expression specified in pattern.
	
	Keeping track of the filenames is recommended. 
	'''
	
	filename_list = []
	delay_list = []

	for i in os.listdir(path):
		if re.search(pattern,i):
			if debug == True:
				print(i)
			delay_list.append(pd.read_csv(path+i, sep='\t'))
			filename_list.append(i)
	
	print("Sorting your lists")
	filename_list, save_list = (list(t) for t in zip(*sorted(zip(filename_list, delay_list))))
	
	print("Delay import completed")
		
	return filename_list, save_list


def split_anisotropy(input_list):
	'''
	Takes a list of scans which were performed with alternating 
	polarizations and returns two lists, each for a different polarizations.
	Useful to check if there is some photoselection effect. 

	Might be applied after preprocessing and creation of averaged maps.  
	'''
	print("Check it also on the filenames to make sure that the order of import was correct")
	pol_a = []
	pol_b = []
	i     = 0 
	assert  ((len(input_list) % 2) == 0), "In order to be able to split the list into two parts it should contain an even number of elements."
	for scan in input_list: 
		if i % 2 == 0: 
			pol_a.append(scan)
		else:
			pol_b.append(scan)
		i += 1 
	return pol_a, pol_b 


def create_maps(input_list_OD, input_list_delays, numberpixel=512):
	'''
	Input data:
		(1) Delta OD scan list
		(2) Delay scan list
		(3) Number of pixels in detector
	
	Does:
		Reshaping of (1) in Fortran ordering
	
	Returns:
		Reshaped version of (1)
	'''
	assert  (numberpixel > 0), "Where are your pixels?"
	timesteps = []
	for i in range(0,len(input_list_delays)):
		timesteps.append(input_list_delays[i].Delay)

	meas = []
	for i in range(0,len(input_list_OD)):
		meas.append(input_list_OD[i]["Delta OD"].values.reshape(numberpixel,len(timesteps[i]),  order='F'))
		
	print("Maps created!")
	return meas


def cut_edges(maps, cut_top_wl, cut_bottom_wl, wavelength_list): 
	'''
	Input:
		(1)  List of maps
		(2)  Wavelength of cut at top
		(3)  Wavelength of cut at bootm
		(4)  List of wavelengths
	Returns:
		new maps, where the noise on the edges is cut away and a new
		list of wavelengths, which only contains the ones which are also in 
		the maps 
	'''

	cut_top_index    = find_closest_index(wavelength_list, cut_top_wl)
	cut_bottom_index = find_closest_index(wavelength_list, cut_bottom_wl)
	
	if cut_top_index > 	cut_bottom_index:
		meas_no_edge     = []
		for i in range(0,len(maps)):
			meas_no_edge.append(maps[i][cut_bottom_index:cut_top_index,:])
		
		wavelengths_cut = wavelength_list[cut_bottom_index:cut_top_index]
	
	else:
		meas_no_edge     = []
		for i in range(0,len(maps)):
			meas_no_edge.append(maps[i][cut_top_index:cut_bottom_index,:])
		
		wavelengths_cut = wavelength_list[cut_top_index:cut_bottom_index]


	return meas_no_edge, wavelengths_cut


'''
CHECK FUNCTIONS
'''

def boxcar_regimecheck(check_list, angle=107):
	'''
	Checks if some BBO angles in the delay list, provided as the
	check list argument are in the non-linear boxcar regime 
	'''
	note = []
	for i in range(0,len(check_list)):
		for j in range(0,len(check_list[i])):
			if any(df_bbo_delay[i].BBO < angle) == True:
				note.append(1)

	if len(note) !=0:
		print('You are in the non-linear boxcar regime. Be afraid!')



def check_import(OD_List, delay_list, numberpixel=512):
	'''
	Compares the length of the lists and of each scan
	'''
	assert  (numberpixel > 0), "Where are your pixels?"
	if (len(OD_List) == len(delay_list)):
		for i in range(0,len(OD_List)):
			if len(OD_List[i]) == numberpixel*len(delay_list[i]):
				note = 0
			else: 
				note = 1 
		if note == 0:
			print('Import correct')
	else:
		print('Recheck patterns and files, the OD files and delay files are not consistent')


'''
PLOTTING
'''

def plot_map(scan, x_axis=None, y_axis=None, title_map="Map", xlabel="delay", ylabel = "wavelength", z_min=-0.01, z_max=0.01, plottype="linear"):
	trace = go.Heatmap(
				z    = scan, 
				x    = x_axis,
				y    = y_axis,
				zmin = z_min,
				zmax = z_max, 
			)
	
	layout = go.Layout(
				title = title_map,
				xaxis=dict(
					title=xlabel,
					type = plottype,
				),
				yaxis=dict(
					title=ylabel,
				)
			)
	
	fig = go.Figure(data = [trace], layout=layout)
	py.iplot(fig)



def plot_map_compare(scan1, scan2, x_axis1=None, y_axis1=None, x_axis2=None, y_axis2=None, title_map="Map", xlabel="delay / ps", ylabel = "wavelength / nm"):
	trace1 = go.Heatmap(
				z    = scan, 
				x    = x_axis1,
				y    = y_axis1,
			)
	
	trace1 = go.Heatmap(
				z    = scan, 
				x    = x_axis2,
				y    = y_axis2,
			)

	layout = go.Layout(
				title = title_map,
				xaxis=dict(
					title=xlabel,
				),
				yaxis=dict(
					title=ylabel,
				)
			)
	
	fig = tools.make_subplots(rows=1, cols=2)

	fig.append_trace(trace1, 1, 1)
	fig.append_trace(trace2, 1, 2)

	py.iplot(fig)


def plot_timetrace(wl, wl_list, data, normalize=False, plottype="linear"):
	'''
	Input:
		(1) Wavelength for the timetrace
		(2) tree column array, 0 column contains map, 1 column contains time
			list, 2 column contains names 
	Output: 
		Plot of timetrace
	'''

	if normalize == False:
		if len(data) == 1:
			ryb = [(0,0,255)]
		else:
			ryb = sns.color_palette('RdBu_r', len(data))
		data_list = []
		for i in range(0,len(data)):
			c = "rgb"+str(ryb[i])
			trace = go.Scatter(
						x = data[i][1],
						y = data[i][0][find_closest_index(wl_list, wl),:],
						mode = 'lines',
						name = str(data[i][2]),
						line = dict(
								color =  (c),
								)
					)
			data_list.append(trace)
				
		layout = go.Layout(
					showlegend=True,
					title = "Timetrace at "+ str(wl) + " nm",
					xaxis=dict(
						title='time delay / ps',
						type = plottype,
					),
					yaxis=dict(
						title='Δ OD',
					)
			   )
		fig = go.Figure(data=data_list, layout=layout)
		py.iplot(fig)


	else:
		if len(data) == 1:
			ryb = [(0,0,255)]
		else:
			ryb = sns.color_palette('RdBu_r', len(data))
		data_list = []
		for i in range(0,len(data)):
			c = "rgb"+str(ryb[i])
			trace = go.Scatter(
						x = data[i][1],
						y = data[i][0][find_closest_index(wl_list,wl),:]/data[i,0][find_closest_index(wl_list,wl),find_closest_index(data[i][1],0)]:
							pass)],
						mode = 'lines',
						name = str(data[i][3]),
						line = dict(
								color =  (c),
								)
					)
			data_list.append(trace)
				
		layout = go.Layout(
					showlegend=True,
					title = "Timetrace at "+ str(wl) + " nm",
					xaxis=dict(
						title='time delay / ps',
						type = plottype,
					),
					yaxis=dict(
						title='Δ OD',
					)
			   )
		fig = go.Figure(data=data_list, layout=layout)
		py.iplot(fig)


def plot_spectral_cut(data, time, wavelengths, normalize=False):
	'''
	Input:
		(1) Time for the spectral cut
		(2) tree column array, 0 column contains map, 1 column contains time
			list, 2 column contains names 
	Output: 
		Plot of the spectral cut
	'''

	if normalize == False:
		if len(data) == 1:
			ryb = [(0,0,255)]
		else:
			ryb = sns.color_palette('RdBu_r', len(data))
		data_list = []
		for i in range(0,len(data)):
			c = "rgb"+str(ryb[i])
			trace = go.Scatter(
						x = wavelengths,
						y = data[i][0][:,find_closest_index(data[i][1], time)],
						mode = 'lines',
						name = str(data[i][2]),
						line = dict(
								color =  (c),
								)
					)
			data_list.append(trace)
				
		layout = go.Layout(
					showlegend=True,
					title = "Spectral cut at "+ str(time) + "ps",
					xaxis=dict(
						title='wavelength / nm',
					),
					yaxis=dict(
						title='Δ OD',
					)
			   )
		fig = go.Figure(data=data_list, layout=layout)
		py.iplot(fig)

	else:
		if len(data) == 1:
			ryb = [(0,0,255)]
		else:
			ryb = sns.color_palette('RdBu_r', len(data))
		data_list = []
		for i in range(0,len(data)):
			c = "rgb"+str(ryb[i])
			trace = go.Scatter(
						x = wavelengths,
						y = data[i][0][:,find_closest_index(data[i][1], time)]/np.max(data[i][0][:, find_closest_index(data[i][1], time)] ),
						mode = 'lines',
						name = str(data[i][2]),
						line = dict(
								color =  (c),
								)
					)
			data_list.append(trace)
				
		layout = go.Layout(
					showlegend=True,
					title = "Spectral cut at "+ str(time) + "ps",
					xaxis=dict(
						title='wavelength / nm',
					),
					yaxis=dict(
						title='normalized Δ OD',
					)
			   )
		fig = go.Figure(data=data_list, layout=layout)
		py.iplot(fig)

	
			


'''
PROCESSING
'''

def svd_noise_corr(input_maps, threshold=15):
	'''
	Input:
		(1) List of input maps on which the SVD noise correction should be performed
		(2) Threshold, up to which eigenvalues the SVD results should be kept
		
	Does: 
		Singular value decomposition of the input matrix. 
		
	Output: 
		(1) Map with hopefully less noise 

	Reference:
		1. van Stokkum, I. H. M., Larsen, D. S. & van Grondelle, 
		R. Global and target analysis of time-resolved spectra. 
		Biochim. Biophys. Acta - Bioenerg. 1657, 82–104 (2004).
		2. Ruckebusch, C., Sliwa, M., Pernot, P., de Juan, A. & Tauler, R. 
		Comprehensive data analysis of femtosecond transient absorption spectra: A review. 
		J. Photochem. Photobiol. C Photochem. Rev. 13, 1–27 (2012).
	'''
	
	noise_anal_array= []
	for i in range(0,len(input_maps)): 
		noise_anal_array.append(input_maps[i].reshape((input_maps[i].size,1), order='F' ))
	
	
	svd_array = np.concatenate( noise_anal_array, axis=1 )
	
	
	U, S, V = np.linalg.svd(svd_array, full_matrices=False) # not exact but fast
	
	
	assert np.allclose(svd_array, np.dot(U * S, V)) == True, "Reconstruction failed"
	
	U_filter = U[:,:threshold+1]
	S_filter = S[:threshold+1]
	V_filter = V[:threshold+1,:]
	
	reconstructed = np.dot(U_filter * S_filter, V_filter)
	
	reconstructed_array = []
	for i in range(0,reconstructed.shape[1]):
		reconstructed_array.append(reconstructed[:,i])
	
	for i in range(0,len(reconstructed_array)): 
		reconstructed_array[i] = reconstructed_array[i].reshape(input_maps[i].shape[0],input_maps[i].shape[1], order='F')
	
	return reconstructed_array


def average_scan(input_maps):
	'''
	Input: List of scans that should be averaged
	Ouput: Averaged map
	'''
	return np.mean(input_maps,axis=0)

def baseline_corr(scan_map, wavelength_list, x_min, x_max, wl_min, wl_max):
	'''
	Plot a map and look for a region before time zero. Remember the 
	coordinates of this region. This function averages the map in this region
	and substracts it from the map and in this way performs the simplest
	possible basline correction. 

	The order x_min/x_max and wl_min/wl_max does not matter. The function 
	automatically finds which is the correct upper/lower bound. 

	Input: 
		(1) Map 
		(2) List of wavelengths
		(3) Minimum in time 
		(4) Maximum in time
		(5) Minimum in wavelength
		(6) Maximum in wavelength

	Output: 
		baseline corrected map
	'''
		y_min = find_closest_index(wavelength_list, wl_min)
		y_max = find_closest_index(wavelength_list, wl_max)
		
		if y_min < y_max:
			noise = np.mean(scan_map[x_min:x_max, y_min:y_max])
			noise_corr = scan_map - noise
		else:
			noise = np.mean(scan_map[x_min:x_max, y_max:y_min])
			noise_corr = scan_map - noise
			
		return noise_corr

def automatic_parity_corr(maps,x_window=0.01,z_percentage=0.5):
	'''
	(EXPERIMENTAL)
	slow algorithm that needs revision and aims to correct 
	parity flips of the detector by comparing the sign of the signal
	at one pixel with. This can also lead to smoothening.  
	
	Input:
		(1) Map on which parity correction should be performed
		(2) Optional: Ranges in which the comparison should be performed
	
	Output: 
		List with parity corrected maps
	'''
	meas_parity_corr = []
	
	for j in range(0,len(maps)):
		tmp = np.zeros(maps[j].shape)
		for k in range(0,maps[j].shape[1]):
			for i in range (0, maps[j].shape[0]):
				average = np.sum(maps[j][i,k-int(x_window/2):k+int(x_window/2)])/x_window
				if maps[j][i,k] < 0  and (average*(1-z_percentage)< maps[j][i,k]*(-1) < average*(1+z_percentage)):
					tmp[i][k] = maps[j][i,k]*(-1)
				elif maps[j][i,k] > 0  and (-average*(1-z_percentage)< maps[j][i,k] < -average*(1+z_percentage)):
					tmp[i][k] = maps[j][i,k]*(-1)
				else: 
					 tmp[i][k] = maps[j][i,k]
		meas_parity_corr.append(tmp)
		
	return meas_parity_corr


def manual_parity_corr(map, wl, wl_list, t_lim, sign=-1):
	'''
	This version of the parity correction assumes that you know (i.e. by looking at the map)
	the sign of the signal at a paritcular wavelength. If at some time of the scan the sign
	flips due to a parity flip of the detector, this function will correct for it up to a time which you specify. 

	Input: 
		(1) Map 
		(2) Wavelength at which parity correction should be performed 
		(3) Minimum time (you will rather get problems before time zero) given as a index
		(4) Sign of the signal at the particular wavelength, encodeding 1 = positive, -1 = negative. 
			In the default settings a negative bleach signal is assumed. 
		(5) List of wavelengths 

	Does:
		Scans at the particular timetrace the sign of the signal and reverses it for all wavelengths
		at that time if it is not equal to the sign provided in the function call

	Output: 
		Parity corrected map
	'''

	wl_idx = find_closest_index(wl_list, wl)
	parity_corr_map = np.zeros(map.shape)
	for i in range(0,t_lim):
		if ((map[wl_idx,i] > 0 & sign==-1) | (map[wl_idx,i] < 0 & sign==+1)):
			parity_corr_map[:,i] = map[:,i] * (-1)
		else: 
			parity_corr_map[:,i] = map[:,i]
			
	return  parity_corr_map


'''
Maybe look for some Sobel implementation
'''

def process_map(input_map, hws=4,wavelengths=512,timesteps=150):
	'''
	Average and find maximum contrast using a convolution. 
	'''
	map_tensor = torch.from_numpy(input_map).float()
	map_variable = Variable(map_tensor.view(1, 1, wavelengths, timesteps))
	kernel = [-1]*hws + [1]*hws
	kernel_pytorch = Variable(torch.Tensor(kernel).view(1, 1, 1, -1))
	padded_before_conv = functional.pad(
		map_variable, pad=(0, len(kernel)-1, 0, 0), mode='replicate')
	return functional.conv2d(
		padded_before_conv, kernel_pytorch, stride=1)


	
def find_curve(input_map, hws=4, deg=3, wavelengths=512,timesteps=150):
	'''
	Find the GVD curve using Huber regression
	'''
	wavelength_max, max_ids = torch.max(input_map.view(wavelengths, timesteps).abs(), dim=1)
	max_ids += hws
	wavelengths = np.arange(wavelengths)
	wavelength_max = wavelength_max.data.numpy()
	max_ids = max_ids.data.numpy()
	pipe = Pipeline([
		('poly', PolynomialFeatures(degree=deg, include_bias=False)),
		('scaler', StandardScaler()),
		('huber', HuberRegressor())
	])
	pipe.fit(wavelengths[:, None], max_ids, huber__sample_weight=wavelength_max)
	return pipe.predict(wavelengths[:, None])

def plot_map_gvd(input_map, wavelengths,predictions=None):
	'''
	Plot map and Huber regression if available
	'''
	#plt.figure(figsize=(6, 5))
	#plt.imshow(input_map, cmap=plt.cm.coolwarm, aspect='auto', vmin = -0.0005, vmax = 0.0005)
	#plt.colorbar(label='intensity')
	#if predictions is not None: # Prediction line if any
	#	plt.plot(wavelengths)
	#	plt.plot(predictions)
	#plt.show()
	if predictions is None:
		plot_map(input_map, y_axis=wavelengths)

	else:
		trace = go.Heatmap(
				z    = input_map, 
				y    = wavelengths,
			)

		trace2 = go.Scatter(
						x = predictions,
						y = wavelengths,
						mode = 'lines',
					)
	
		layout = go.Layout(
				xaxis=dict(
					title="delay",
				),
				yaxis=dict(
					title="wavelengths",
				)
			)
	
		fig = go.Figure(data = [trace,trace2], layout=layout)
		py.iplot(fig)




def GVD_correction(scan_map,timesteps,wavelength_list,hws=4,order=6):
	'''
	Function that calls all other functions needed for GVD control. 

	Input: 
		(1) Map on which GVD correction should be performed
		(2) List of timesteps
		(4) Optional: Kernel size

	Output: 
		(1) GVD corrected map
		(2) New list of delays

	Reference: 
		For idea behind GVD correction, consult e.g. 
		1. Ruckebusch, C., Sliwa, M., Pernot, P., de Juan, A. & Tauler, R. 
		Comprehensive data analysis of femtosecond transient absorption spectra: A review. 
		J. Photochem. Photobiol. C Photochem. Rev. 13, 1–27 (2012).
	'''
	x_dim = len(timesteps)
	y_len = len(wavelength_list)

	t0_zero = np.zeros(scan_map.shape)
	plot_map(scan_map,y_axis=wavelengths)
	
	diffentiated2 = process_map(scan_map, wavelengths=y_len, timesteps=x_dim)
	#plot_map_gvd(diffentiated2.abs().data.view(scan_map.shape[1], scan_map.shape[0]).numpy(), wavelength_list)

	predictions = find_curve(diffentiated2, deg=6, wavelengths=y_len,timesteps=x_dim)
	plot_map_gvd(scan_map,wavelength_list, predictions)
	t0_wl = np.argmax(predictions)
	t0 = np.max(predictions)
	
	startval = int(round(t0))

	t0_delays =  timesteps - timesteps[startval] 
	t0_index = np.argmax(t0_delays == 0)
		
	print("the time zero index is {:.4f} and at wavelength index {:.1f}\n".format(t0, t0_wl) ) 
	
	print("starting now the shifting loop")
	for i in range(0,scan_map.shape[0]):
		shift_i = t0 -predictions[i] # difference in indices to maximum of GVD curve 
		#Shift now according to this difference in indices
		# Fill previously undefined values with 0. This is a slow scipy function, maybe 
		#implement on shifting function later https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
		t0_zero[i,:] = shift(scan_map[i,:], shift_i, cval = 0) 
		t0_delays_shifted = shift(t0_delays, shift_i, cval=0) # shift delays
		t0_delays_shifted = t0_delays_shifted - t0_delays_shifted[startval]
		f = interp1d(t0_delays_shifted,  t0_zero[i,:], bounds_error=False,fill_value=0.,kind='linear')
		t0_zero[i,:] = f(t0_delays)

	plot_map(t0_zero, y_axis=wavelength_list)
	print("shift completed \n")

	return t0_zero, t0_delays



'''
EXPORT Options
'''
def create_optimus_file(input_map,times,wavelength,outputname):
	'''
	For optimus, .ana files are needed. The information is provided after
	keyword in a space-separated form. 
	
	We noticed that it is important that the order of the times and wavelengths 
	is correct.
	
	Input:
		(1) Input map, the code here assumes that the map is in such a form that the x dimension 
			is the time, and the y dimension is the wavelength and that the ordering in both dimensions 
			corresponds to the one provided in the wavelength and time lists as the map will be flipped 
			assuming that this is true. 
		(2) Timelist
		(3) Wavelength
		
	Ouput: 
		.ana file for optimus

	Reference:
		1. Slavov, C., Hartmann, H. & Wachtveitl, J. Implementation and Evaluation of Data Analysis
		 Strategies for Time-Resolved Optical Spectroscopy. Anal. Chem. 87, 2328–2336 (2015).
	'''
	assert (len(wavelength) == input_map.shape[0]), "Somehow the shape of the map is not consistent with the number of wavelengths" 
	assert (len(times) == input_map.shape[1]), "Somehow the shape of the map is not consistent with the number of delays" 
	
	if wavelength[0] > wavelength[-1]:
		wavelength = wavelength[::-1]
		input_map  = np.flipud(input_map)
		
	if times[0] > times[-1]:
		times = times[::-1]
		input_map = np.fliplr(input_map)
	
	filename = str(outputname) +  ".ana"
	with open(filename, "w") as f_handle:
		f_handle.write("%FILENAME="+filename[:-4]+"\n")
		f_handle.write("%DATATYPE=TAVIS\n")
		f_handle.write("%TIMESCALE=ps\n")
		f_handle.write("%TIMELIST=")
		np.savetxt(f_handle, times)
		f_handle.write("%WAVELENGTHLIST=")
		np.savetxt(f_handle, wavelength)
		f_handle.write("%INTENSITYMATRIX=\n")
		np.savetxt(f_handle, input_map)
	f_handle.close()


def create_pyldm_file(input_map, times, wavelength, outputname):
	'''
	Creates maps in a form as they are needed for pyLDM. PyLDM wants one large 
	matrix in the format 

		WL 1, WL 2, ...
	t1,data1,data2, ...
	t2,data3,data3, ...
	
	Input:
		(1) Input map, the code here assumes that the map is in such a form that the x dimension 
			is the time, and the y dimension is the wavelength and that the ordering in both dimensions 
			corresponds to the one provided in the wavelength and time lists
		(2) Timelist
		(3) Wavelength
		
	Ouput: 
		PyLDM input file (.txt ending)

	Reference:
		1. Dorlhiac, G. F., Fare, C. & van Thor, J. J. PyLDM - 
		An open source package for lifetime density analysis of time-resolved spectroscopic data. 
		PLoS Comput. Biol. 13, 1–15 (2017).
	'''
	
	assert (len(wavelength) == input_map.shape[0]), "Somehow the shape of the map is not consistent with the number of wavelengths" 
	assert (len(times) == input_map.shape[1]), "Somehow the shape of the map is not consistent with the number of delays" 
	
	wavelengths_plydm = np.hstack([[0], wavelength])
	pyldm_array2 = np.vstack([times, input_map]) 
	pyldm_array2_2 = np.zeros((input_map.shape[0]+1, input_map.shape[1]+1))
	pyldm_array2_2[:,0] = wavelengths_plydm
	pyldm_array2_2[:,1:] = pyldm_array2
	filename = str(outputname) + ".txt"
	np.savetxt(filename, pyldm_array2_2.T, delimiter=",")



'''
SPECIAL CASES
'''

def fit_IRF(scan_map, delays, wavelengths, starting_values = [0.003, 1843, 0.55]):
	'''
	Assuming that you measure e.g. the two-photon absorption 
	in EtOH. Fits Gaussian at each wavelength to this signal and return the FWHMs. 

	This value should be used for convolution with the fitting function in 
	further analysis (LDA, global analysis ...) or for comparison with the value 
	that PyLDM or Optimus returns. 

	Input: 
		(1) Map of IRF measurement (preprocessed)
		(2) Delays
		(3) list of wavelengths
		(4) List of starting values for your Gaussian. In following order:
		pre-factor (amp), center (cen) (depends of course
		of the units of delay you choose), width (wid):
		(amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2) )

	Output:
		The function will create a plot of the IRF as a function of probe wavelength
		and return a list with the FWHMs. 
	'''

	widths = []
	pos = []
	for i in range(0,scan_map.shape[0]):
		popt, pcov = curve_fit(gaussian, delays, scan_map[i,:], p0 = [0.003, 1843, 0.55])
		widths.append(FWHM(popt[2]))
		pos.append(popt[1])

	trace= go.Scatter(
			x = wavelengths,
			y = widths, 
			)

	layout = go.Layout(
					showlegend=True,
					title = "IRF fit",
					xaxis=dict(
						title='wavelength / nm',
					),
					yaxis=dict(
						title='FWHM',
					)
			   )
	fig = go.Figure(data=[trace], layout=layout)
	py.iplot(fig)

	return widths

