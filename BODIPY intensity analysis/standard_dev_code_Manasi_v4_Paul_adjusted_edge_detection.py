import csv
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes
from skimage.filters import threshold_otsu, threshold_local
import os
import pandas as pd
from scipy.signal import medfilt
import matplotlib.cm as cm
import imageio
import tifffile as tiff
from skimage import measure
import cv2
from scipy import interpolate
from scipy.interpolate import interp1d
from skimage.measure import label, regionprops
from scipy.spatial.distance import euclidean
from scipy import stats
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt

# - Paul - :
# adjusted this code by adding a function that can create edge masks (taking the mask created by a channel (e.g. cellmask or DNA) and then diluting its edge so that a edge mask with defined width is created)
# the width of this masks goes "width" to the inside and "width % 2" to the outside, cause I saw that the edge is detected right outside the cell (so more of the insede part is actually desirable)
# furthermore the mean and total intensity of a chosen channel within the edgemask is then calculated and stored in a excel file (raw, normalized, and interpolated)


# this function finds all folders that contains the key string, for us that will be 'cell'
# all it needs is some dumb rule to follow by
def folder_finder(folder_addy=None, folder_key=None):
	if folder_addy is None:
		folder_addy = os.path.dirname(os.path.abspath(__file__))

	if folder_key == None:
		folder_key = 'cell'

	matching_folders = [
		os.path.join(folder_addy, item)
		for item in os.listdir(folder_addy)
		if os.path.isdir(os.path.join(folder_addy, item)) and folder_key in item.lower()
	]

	return matching_folders, folder_addy


# this function then finds the specific movie files that need to be processed
def file_list_process(matching_folders, custom_file_ext = None):
	
	if custom_file_ext == None:
		custom_file_ext = '.tif'
	
	temp_files = []
	for element in matching_folders:
		temp_file = os.path.basename(element)
		temp_file = temp_file + custom_file_ext
		temp_files.append(temp_file)
		temp_file = None
	return temp_files


# makes a mask given the array of an image
# very basic: normalizes the image based on max pixel intensity, blurs it, applys global threshold, applies dilation and erosin, fills holes!
def mask_maker(image1):

	# re-stating things
	image1 = image1
	# plt.imshow(image1, cmap='gray')  # cmap='gray' for grayscale, remove it for RGB
	# plt.axis('off')  # Hide axis
	# plt.show()

	# Optionally, rescale to a range like 0-1
	# normalized_image = image1 / np.max(image1)
	# normalizing with percentile because this makes it possible to get rid of those outliers
	#that make it hard to segment the roi when there is super high signal off to the side
	chosen_percentile = 95
	perc_val = np.percentile(image1.flatten(), chosen_percentile)
	normalized_image = image1 / perc_val
	normalized_image = np.clip(normalized_image, 0, 1)
	# plt.imshow(normalized_image, cmap='gray')  # cmap='gray' for grayscale, remove it for RGB
	# plt.axis('off')  # Hide axis
	# plt.show()

	#apply gaussian blur
	blurred_image1 = gaussian_filter(normalized_image, sigma=3)
	# plt.imshow(blurred_image1, cmap='gray')  # cmap='gray' for grayscale, remove it for RGB
	# plt.axis('off')  # Hide axis
	# plt.show()

	# Apply adaptive thresholding to create a binary mask
	# For global thresholding, you can use Otsu's method:
	global_thresh = threshold_otsu(blurred_image1)
	binary_mask_global = blurred_image1 > global_thresh
	# plt.imshow(binary_mask_global, cmap='gray')  # cmap='gray' for grayscale, remove it for RGB
	# plt.axis('off')  # Hide axis
	# plt.show()


	#apply dilation and close to connnect almost circles
	##Define a structuring element (3x3 square in this example) - this is the kernel!
	structuring_element = np.ones((1, 1), dtype=bool)

	##Apply dilation
	dilated_image = binary_dilation(binary_mask_global, structure=structuring_element)
	# plt.imshow(dilated_image, cmap='gray')  # cmap='gray' for grayscale, remove it for RGB
	# plt.axis('off')  # Hide axis
	# plt.show()


	##Apply erosion
	eroded_image = binary_erosion(dilated_image, structure=structuring_element)
	# plt.imshow(eroded_image, cmap='gray')  # cmap='gray' for grayscale, remove it for RGB
	# plt.axis('off')  # Hide axis
	# plt.show()


	# now filling all holes!
	filled_image = binary_fill_holes(eroded_image)#binary_fill_holes(eroded_image)
	# plt.imshow(filled_image, cmap='gray')  # cmap='gray' for grayscale, remove it for RGB
	# plt.axis('off')  # Hide axis
	# plt.show()


	# reassigning binary_global_mask to the filled_image
	binary_mask_global = filled_image
	# print(binary_mask_global)

	return binary_mask_global

# add on by Paul

def make_edge_mask(mask, edge_width=5):
    # """
    # Returns an edge band around the mask boundary.
    # - Inside width = edge_width
    # - Outside width = edge_width // 2
    # """
    mask_bool = mask.astype(bool)
    
    # Distance from inside mask to boundary
    dist_inside = distance_transform_edt(mask_bool)
    # Distance from outside mask to boundary
    dist_outside = distance_transform_edt(~mask_bool)
    
    inside_width = edge_width
    outside_width = edge_width // 2
    
    # Edge band: pixels within inside_width inside or outside_width outside
    edge_band = (dist_inside <= inside_width) & (dist_outside <= outside_width)
    
    return edge_band

# calculates the sharpness of the image where a higher score indicates more edges and more in focus probs: laplacian method (2nd deriv)
def calculate_sharpness_lapl(image1):

	# Apply Gaussian blur to reduce noise
	blurred = cv2.GaussianBlur(image1, (3, 3), 0)

	# Compute the Laplacian
	laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

	# Calculate the variance of the Laplacian as the sharpness score
	sharpness_score = np.var(laplacian)

	return sharpness_score


# calculates the sharpness of the image where a higher score indicates more edges and more in focus probs: sobel method (1st deriv)
def calculate_sharpness_sobel(image1):

	# Apply Gaussian blur to reduce noise
	blurred = cv2.GaussianBlur(image1, (3, 3), 0)

	sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
	sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
	gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
	sharpness_score = np.var(gradient_magnitude)

	return sharpness_score


# minwoos SDI calculation
def SDI_gini_coefficient(values,binary_mask=None):

	#if i am lazy and do not pass already flatten values and i pass the image with its mask
	if binary_mask is not None:
		values = values[binary_mask].flatten().tolist()
		values = np.array(values)
	#if i have not yet already 0-1 min max normalized all the values then i will do it here
	if max(values) > 1:
		values = (values - min(values)) / (max(values) - min(values))

    #"""Calculate the Gini coefficient of a numpy array."""
	sorted_values = np.sort(values)
	n = len(values)
	cumulative_values = np.cumsum(sorted_values)
	gini_index = (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * np.sum(sorted_values))) - (n + 1) / n
	SDI = 1 - gini_index
	return SDI


# taking SD but normalizing by media cuz i think its poisson distribution
def SD_medianNORM(values,binary_mask=None):

	#if i am lazy and do not pass already flatten values and i pass the image with its mask
	if binary_mask is not None:
		values = values[binary_mask].flatten().tolist()
		values = np.array(values)
	#normalize by the median here
	values = values/(np.median(values))

	return np.std(values,ddof=1)


# taking SD but normalizing by media cuz i think its poisson distribution
def SD_meanNORM(values,binary_mask=None):

	#if i am lazy and do not pass already flatten values and i pass the image with its mask
	if binary_mask is not None:
		values = values[binary_mask].flatten().tolist()
		values = np.array(values)
	#normalize by the median here
	values = values/(np.mean(values))

	return np.std(values,ddof=1)


# this function will determine when it is best to change Z planes based off our focus metric
# it will work by having a window and sliding it across time to see when the averages intersect and change
def ZFocus_slider(dictionary, settings, window, Z, z, c, d, stop_after_switch):

	focus_metric_comparer = []
	# for i in range(Z):
	# 	name = settings['channels_names'][c]+':'+'Z'+str(i)
	# 	focus_metric_comparer.append(dictionary[name])
	for i in range(Z):
		name = settings['channels_names'][settings['mask_channel'][c]] + ':' + settings['channels_names'][settings['extrapolate_channel'][c][d]] + ':Z' + str(i) #this is the key
		focus_metric_comparer.append(dictionary[name])
	# for i in range(Z):
	# 	for j in range(len(settings['extrapolate_channel'])):
	# 		for l in range(len(settings['extrapolate_channel'][j])):
	# 			name = settings['channels_names'][settings['mask_channel'][j]] + ':' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + ':Z' + str(i) #this is the key
	# 			focus_metric_comparer.append(dictionary[name])

	Z = len(focus_metric_comparer)
	T = len(focus_metric_comparer[0])
	# Initialize with the first plane as the best
	current_best = 0
	best_plane = [0] * T
	best_plane[:window] = [current_best] * window  # Assume the first choice for the initial window
	switch_made = False

	for i in range(T - window):
		if stop_after_switch and switch_made:
			# fill the rest with current best plane
			best_plane[i:] = [current_best]*(T-1)
			break

		comparing_list = [np.mean(focus_metric_comparer[j][i:i+window]) for j in range(Z)]

		# Find the plane with the highest mean
		new_best = np.argmax(comparing_list)

		# Check if we need to switch the best plane
		if new_best != current_best:
			# Confirm the current first time point in window supports a switch
			if focus_metric_comparer[new_best][i] > focus_metric_comparer[current_best][i]:
				current_best = new_best
				switch_made = True

		# Update the best plane indicator
		best_plane[i + window] = current_best

	# Convert to binary indicators per plane
	final_plane_choices = []
	for j in range(Z):
		final_plane_choices.append([1 if best_plane[t] == j else 0 for t in range(T)])

	return final_plane_choices[z]


# this function makes a best guess on which region is the true mask (if more than one) by basing it off what the previous area and location was
def select_best_region(mask, ref_area=None, ref_centroid=None, area_weight=0.5, dist_weight=0.5):
	labeled_mask = label(mask)
	props = regionprops(labeled_mask)

	if not props:
		return np.zeros_like(mask, dtype=bool), None, None  # No regions detected

	scores = []
	for prop in props:
		area = prop.area
		centroid = prop.centroid  # (y, x)

		# If no reference yet, just use area
		if ref_area is None or ref_centroid is None:
			score = area
		else:
			# Area similarity (closer to 1 is better)
			area_score = min(area, ref_area) / max(area, ref_area)

			# Distance similarity (closer to 0 is better, so we invert it)
			dist = euclidean(centroid, ref_centroid)
			dist_score = 1 / (1 + dist)  # avoids division by 0

			# Combined score
			score = area_weight * area_score + dist_weight * dist_score

		scores.append(score)

	# Pick region with max score
	best_idx = np.argmax(scores)
	best_label = props[best_idx].label
	best_mask = labeled_mask == best_label
	best_area = props[best_idx].area
	best_centroid = props[best_idx].centroid

	return best_mask, best_area, best_centroid


# acquiring the circularity, roundess, and area of the largest segemented region
# getting the largest region to ignore small spots that may have gotten masked in
def calculate_largest_region(mask):
    # Label connected components
    labeled_mask = measure.label(mask)

    # Extract properties of each region
    properties = measure.regionprops(labeled_mask)

    if not properties:
        return None, None, None  # Return None if no regions are detected

    # Find the largest region by area
    largest_region = max(properties, key=lambda x: x.area)

    # Calculate Circularity and Roundness
    perimeter = largest_region.perimeter if largest_region.perimeter > 0 else np.nan
    circularity = (4 * np.pi * largest_region.area) / (perimeter ** 2) if perimeter > 0 else np.nan
    roundness = (4 * largest_region.area) / (np.pi * (largest_region.major_axis_length ** 2)) if largest_region.major_axis_length > 0 else np.nan
    area = largest_region.area

    return circularity, roundness, area


# this function will read the info.txt file and label the movie accordingly
# for this func, state0 = whack, state1 = MV shed, state2 = NE rupture, state3 = PM rupt
def info_txt_reader(addyy_name,T):
	file_name = addyy_name + '//info.csv' #this is the file name
	df_info = pd.read_csv(file_name) #turning it into a dataframe yo
	timings = df_info['events'] - 1 #need to subtract at one because python indexes at 0 instead of 1 like imageJ

	NETosis_state = []
	state = 0
	for i in range(T): #sift thru time
		for j in range(len(timings)): #for each time point, check if we need to change the state!
			if i == timings[j]:
				state = state + 1
		NETosis_state.append(state) #appending the state to a time list
	# NETosis_state = [1 if x == 2 else x for x in NETosis_state] #this is combining MV shed and NE rupt cuz im curious
	return NETosis_state


# this function will read the info.txt file and label the movie accordingly
# for this func, state0 = MV shed, state1 = NE perm, state2 = NE rupture, state3 = PM rupt
def info_txt_reader_v2(addyy_name,T):
	file_name = os.path.join(addyy_name, 'info.csv') #file_name = addyy_name + '\\info.csv' #this is the file name
	df_info = pd.read_csv(file_name) #turning it into a dataframe yo
	timings = df_info['time']
	# this wont work cuz some spaces might have a NaN that is why i did it down below when i s # timings = df_info['events'] - 1 #need to subtract at one because python indexes at 0 instead of 1 like imageJ


	NETosis_state = []
	state = 0
	for i in range(T): #sift thru time
		for j in range(len(timings)): #for each time point, check if we need to change the state!
			if not pd.isna(timings[j]):
				if i == timings[j] - 1: #subtracted one because python indexes at 0 instead of 1 like imageJ
					state = state + 1
		NETosis_state.append(state) #appending the state to a time list
	# NETosis_state = [1 if x == 2 else x for x in NETosis_state] #this is combining MV shed and NE rupt cuz im curious
	return NETosis_state


# this is an intermediate to mess around with before trying to segment the image
def intermediate_mask_maker(image1, addyy_name, settings):

	T, Z, C, Y, X = image1.shape #just getting the shape so we can loop through each tiff file correctly

	temp_dict = {} #this dictionary will just temporarily hold the in focus metric - we are doing it here just cuz

	#going to quickly initialize the folders i will need to save everything!
	for i in range(Z):
		# for j in range(C):
		# 	folder_path = addyy_name + '\\' + settings['channels_names'][j] + '-Z' + str(i)
		# 	os.makedirs(folder_path, exist_ok=True)
		# 	# print(f"{folder_path} created successfully!")
		for j in range(len(settings['extrapolate_channel'])):
			for l in range(len(settings['extrapolate_channel'][j])):
				folder_path = os.path.join(addyy_name, settings['channels_names'][settings['mask_channel'][j]] + '-' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + '-Z' + str(i)) #folder_path = addyy_name + '\\' + settings['channels_names'][settings['mask_channel'][j]] + '-' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + '-Z' + str(i)
				os.makedirs(folder_path, exist_ok=True)
			# print(f"{folder_path} created successfully!")
				edge_folder_path = os.path.join(
                    addyy_name,
                    settings['channels_names'][settings['mask_channel'][j]] + '-EDGE-' +
                    settings['channels_names'][settings['extrapolate_channel'][j][l]] +
                    '-Z' + str(i)
                )
				os.makedirs(edge_folder_path, exist_ok=True)

	#going to go thru each Z and for each Z, going to look at a single channel, and then go thru time
	for i in range(Z):
		for j in range(len(settings['extrapolate_channel'])):
			in_focus_metric = [] #initializing this quickly so can save it to a temp dictionary
			ref_area = None #initializing
			ref_centroid = None #initializing
			for k in range(T):
				mask_temp = mask_maker(image1[k,i,settings['mask_channel'][j],:,:]) #make the mask!
				in_focus_metric.append(calculate_sharpness_lapl(image1[k,i,settings['mask_channel'][j],:,:])) #calculating how in focus the image is!
				#this function below has the purpose of selecting the correct region if mulitple things were chose in the mask
				mask_temp, ref_area, ref_centroid = select_best_region(mask_temp, ref_area, ref_centroid)
				# 🔹 NEW: Create edge mask
				edge_mask = make_edge_mask(mask_temp, edge_width=5)

				for l in range(len(settings['extrapolate_channel'][j])):
					folder_path = os.path.join(addyy_name, settings['channels_names'][settings['mask_channel'][j]] + '-' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + '-Z' + str(i)) #folder_path = addyy_name + '\\' + settings['channels_names'][settings['mask_channel'][j]] + '-' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + '-Z' + str(i) #quickly get the folder addy
					file = os.path.join(folder_path, 't' + str(k) + '.tif') #file = folder_path + '\\t' + str(k) + '.tif' #make the file name which goes in the folder addy we made above
					visualized_masked_image = np.where(mask_temp,image1[k,i,settings['extrapolate_channel'][j][l],:,:] , 0) #overlay mask and image
					tifffile.imwrite(file, visualized_masked_image.astype(image1[k,i,settings['extrapolate_channel'][j][l],:,:].dtype)) # need to save this shi
					# 🔹 NEW: Edge masked cholesterol
					edge_folder_path = os.path.join(addyy_name,settings['channels_names'][settings['mask_channel'][j]] + '-EDGE-' + settings['channels_names'][settings['extrapolate_channel'][j][l]] +'-Z' + str(i))
					edge_file = os.path.join(edge_folder_path, 't' + str(k) + '.tif')
					edge_masked_image = np.where(
                        edge_mask,
                        image1[k,i,settings['extrapolate_channel'][j][l],:,:],
                        0
                    )
					tifffile.imwrite(edge_file, edge_masked_image.astype(image1.dtype))
			
			#going to add the storing of the infocus metric here!
			for l in range(len(settings['extrapolate_channel'][j])):
				name = settings['channels_names'][settings['mask_channel'][j]] + ':' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + ':Z' + str(i) #this is the key 
				temp_dict[name] = in_focus_metric #adding this so it can be passed onto the next function so it can be appended correctly
	return temp_dict


#this is a function that will extract standard deviation, sdi, circularity/roundness, etc
#with this function, we can append many different things to try to understand a good metric for chromatin decompaction
def feature_extracter(image1,addyy_name, settings, focus_metric_dict):

	data_dictionary = {} #this holds info for each Z and channel!

	T, Z, C, Y, X = image1.shape #just getting the shape so we can loop through each tiff file correctly

	#instead of remaking masks and what not, we will use all the images we generated from making masks above
	for i in range(Z):
		for j in range(len(settings['extrapolate_channel'])):
			for l in range(len(settings['extrapolate_channel'][j])):
				temp_dict = {} #intializing a temporary dictionary to hold all our values for each channel and Z
				time = [] #initialize time domain
				std_dev = [] #initialize
				total_intensity = [] #initialize
				mean_intensity = [] #initialize
				SDI = [] #initialize
				std_dev_median = [] #initialize
				std_dev_mean = [] #initialize
				circularity = [] #initialize
				roundness = [] #initialize
				area = [] #initialize

				edge_total_intensity_list = [] #initialize
				edge_mean_intensity_list = [] #initialize

				for k in range(T):
					#getting the file addy of specific tif file
					file = os.path.join(addyy_name, settings['channels_names'][settings['mask_channel'][j]] + '-' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + '-Z' + str(i), 't' + str(k) + '.tif') #file = addyy_name + '\\' + settings['channels_names'][settings['mask_channel'][j]] + '-' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + '-Z' + str(i) + '\\t' + str(k) + '.tif'
					temp_image_array = tiff.imread(file) #reading the tif file (background is all 0's)
					temp_mask = temp_image_array > 0 #create a mask where the image is non-zero

					# Create edge mask (only border region, e.g. 3px wide)
					edge_mask = make_edge_mask(temp_mask, edge_width=5)

					# Edge intensity features
					edge_total_intensity_list.append(np.sum(temp_image_array[edge_mask]))
					edge_mean_intensity_list.append(np.mean(temp_image_array[edge_mask]))


					#extracting feature for each channel and Z here just cuz
					time.append(k) #time
					std_dev.append(np.std(temp_image_array[temp_mask],ddof=1)) #standard deviation, ddof=1 is for sample
					total_intensity.append(np.sum(temp_image_array[temp_mask])) #total intensity
					mean_intensity.append(np.mean(temp_image_array[temp_mask])) #mean intensity
					SDI.append(SDI_gini_coefficient(temp_image_array,temp_mask)) #minwoos SDI
					std_dev_median.append(SD_medianNORM(temp_image_array,temp_mask)) #SD normalized by median
					std_dev_mean.append(SD_meanNORM(temp_image_array,temp_mask)) #SD normalized by the mean
					circularity_ind, roundness_ind, area_ind = calculate_largest_region(temp_mask) #circularity, roundness, area
					circularity.append(circularity_ind) #circularity
					roundness.append(roundness_ind) #roundness which according to minwoo is better :)
					area.append(area_ind) #area just cuz im curious

				#now putting all this info in a dictionary
				temp_dict['Time[min]'] = time 
				temp_dict['SD'] = std_dev
				temp_dict['TotInt'] = total_intensity
				temp_dict['MeanInt'] = mean_intensity
				temp_dict['SDI'] = SDI 
				temp_dict['SD_Med'] = std_dev_median
				temp_dict['SD_Mean'] = std_dev_mean
				temp_dict['Circularity'] = circularity
				temp_dict['Roundness'] = roundness
				temp_dict['Area'] = area
				temp_dict['Edge_TotInt'] = edge_total_intensity_list
				temp_dict['Edge_MeanInt'] = edge_mean_intensity_list


				name = settings['channels_names'][settings['mask_channel'][j]] + ':' + settings['channels_names'][settings['extrapolate_channel'][j][l]] + ':Z' + str(i) #this is the key
				temp_dict['FocusMetric'] = focus_metric_dict[name] #quickly adding that focus metric we calculated from tha mask making function so its in the big boi dataframe to be saved
				Zchange_list = ZFocus_slider(focus_metric_dict, settings, 5, Z, i, j, l, stop_after_switch=True) #going to return 0's and 1's about when it is good to change Z plane
				temp_dict['ChangeZPlane'] = Zchange_list #adding where the best Z plane occurs! 
				#try:
				NETosis_state = info_txt_reader_v2(addyy_name,T) #we are adding the NETosis state here
				temp_dict['NETosisState'] = NETosis_state
				#except:
				#	label = 'no NETosis state txt file for '+addyy_name
				#	print(label)
				data_dictionary[name] = temp_dict #updating all this to a massive dictionary which i will save!

	# Convert each dict to DataFrame
	data_dictionary = {k: pd.DataFrame.from_dict(v) if isinstance(v, dict) else v 
						for k, v in data_dictionary.items()}
	# Now perform concatenation
	combined_df = pd.concat(
		[df.assign(Source=key) for key, df in data_dictionary.items()],
		ignore_index=True
	)
	file_name = os.path.join(addyy_name, "data_dictionary_output.csv") #file_name = addyy_name + "\\data_dictionary_output.csv"
	combined_df.to_csv(file_name, index=False)
	print("Saved data_dictionary_output.csv")


# this function is a glorified looper for each cell data file
# if frap file is labeled incorrectly or doesn't exist, then this will skip the file
# it will also state which file was skipped
def files_looper(settings, temp_folders, folder_addy = None, custom_file_ext = None):

	#not sure if this works it might throw an error
	if folder_addy == None:
		folder_addy = os.path.dirname(os.path.abspath(__file__))

	if custom_file_ext == None:
		custom_file_ext = 'cell'

	#now going to sift through each file and gather the relevant data
	data_dictionary = {}
	for element in temp_folders:
		temp_file = os.path.join(element, custom_file_ext + '.tif') #temp_file = element + '\\' + custom_file_ext + '.tif'

		print(temp_file)
		#try:

		# T, Z, C, Y, X - pretty sure it is this structure (z&c) & (y&x) might be switched
		temp_image_array = tiff.imread(temp_file)
		# we will first try to correctly segment and threshold our nucleus
		focus_metric_dict = intermediate_mask_maker(temp_image_array, element, settings)
		# we have made the mask now lets try calculating the standard deviation,SDI, and other metrics!
		feature_extracter(temp_image_array, element, settings, focus_metric_dict)

		# temp_dictionary = data_extracter(settings,temp_file,element,custom_file_ext = None)
		# data_dictionary.update(temp_dictionary)
		#except FileNotFoundError:
		#	print('File did not exist or labeled incorrectly:',temp_file)


# after making all the dictionaries for each cell, now it is time to compile everything together and see what we get!
def compiler(settings, temp_folders, NETosisSTATE, folder_addy = None, custom_file_ext = None):
	
	metrics = ['SD','TotInt','MeanInt','SDI','SD_Med','SD_Mean','Circularity','Roundness','Area', 'Edge_TotInt','Edge_MeanInt'] #this is hard coded and can be changed :)
	metrics = ['TotInt','Area','MeanInt', 'Edge_TotInt','Edge_MeanInt'] #this is hard coded and can be changed :)

	#not sure if this works it might throw an error
	if folder_addy == None:
		folder_addy = os.path.dirname(os.path.abspath(__file__))

	if custom_file_ext == None:
		custom_file_ext = 'data_dictionary_output'

	#this determines which state everything is initialized to! you could prob code in to normalize to the beginning of each state but rn i dont want that
	NORMSTATE = 1

	for j in range(len(settings['extrapolate_channel'])): #sifting through the channels i want to extrapolate
		for l in range(len(settings['extrapolate_channel'][j])):

			mask_channel_name = settings['channels_names'][settings['mask_channel'][j]]
			extrapolate_channel_name = settings['channels_names'][settings['extrapolate_channel'][j][l]]
			combo_label = f"{extrapolate_channel_name}-{mask_channel_name}"

			# #filtering data based on good Z plane and the correct channel i want to look at
			# filtered_df = df_dict[(df_dict['Source'].str.contains(name)) & (df_dict['ChangeZPlane'] == 1)].reset_index(drop=True)
			# # filtered_df.to_csv('file_name.csv', index=False)
			
			# Create a new subfolder for each mask channel
			output_folder = os.path.join(folder_addy, mask_channel_name)
			os.makedirs(output_folder, exist_ok=True)

			# Create structure to hold all data across cells for each metric and sheet
			all_metric_data = {
			    metric: {'Raw': [], 'Normalized': [], 'Interpolated': []}
			    for metric in metrics
			}

			
			#now going to sift through each file and gather the relevant data
			for element in temp_folders:
				temp_file = os.path.join(element, custom_file_ext + '.csv') #temp_file = element + '\\' + custom_file_ext + '.csv'
				df_dict = pd.read_csv(temp_file) #turning it into a dataframe yo
				cell_name = os.path.basename(element) #cell_name = element.split('\\')[-1]

				name = mask_channel_name + ':' + extrapolate_channel_name + ':Z'
				filtered_df = df_dict[(df_dict['Source'].str.contains(name)) & (df_dict['ChangeZPlane'] == 1)].reset_index(drop=True)

				
				if NETosisSTATE:
					valid_states = filtered_df['NETosisState'].dropna().unique()
					for metric in metrics:
						# Per cell: init dfs for each type
						raw_df_total = pd.DataFrame()
						norm_df_total = pd.DataFrame()
						interp_df_total = pd.DataFrame()

						for k in sorted(valid_states.astype(int)):
							filtered_filtered_df = filtered_df[filtered_df['NETosisState'] == k].reset_index(drop=True)
							if filtered_filtered_df.empty:
								continue

							times = filtered_filtered_df['Time[min]']
							values = filtered_filtered_df[metric]
							states = filtered_filtered_df['NETosisState']

							# RAW
							raw_df_temp = pd.DataFrame({
							    'Time[min]': times,
							    'NETosisState': states,
							    cell_name: values
							})
							raw_df_total = pd.concat([raw_df_total, raw_df_temp], ignore_index=True)

						# Append to all cells' data for the metric
						all_metric_data[metric]['Raw'].append(raw_df_total)

						# NORMALIZED to first value in state 0
						state_0 = filtered_df[filtered_df['NETosisState'] == NORMSTATE].reset_index(drop=True)
						if not state_0.empty:
							baseline = state_0[metric].iloc[0]
							if baseline == 0:
								norm_factor = 1
							else:
								norm_factor = baseline

							for k in sorted(valid_states.astype(int)):
								sub_df = filtered_df[filtered_df['NETosisState'] == k].reset_index(drop=True)
								times = sub_df['Time[min]']
								states = sub_df['NETosisState']
								norm_values = sub_df[metric] / norm_factor

								norm_df_temp = pd.DataFrame({
								    'Time[min]': times,
								    'NETosisState': states,
								    cell_name: norm_values
								})
								norm_df_total = pd.concat([norm_df_total, norm_df_temp], ignore_index=True)

							all_metric_data[metric]['Normalized'].append(norm_df_total)

							# INTERPOLATION using normalized data
							for k in sorted(valid_states.astype(int)):
								state_df = norm_df_total[norm_df_total['NETosisState'] == k].reset_index(drop=True)
								norm_vals = state_df[cell_name]

								if len(norm_vals) >= 2: #you can’t interpolate if there’s only one data point
									interp_func = interp1d(range(len(norm_vals)), norm_vals, kind='linear')				#X-axis is not the actual time in minutes, but just the index position within that state (0, 1, 2, …). Y-axis is the metric values.
									new_x = np.linspace(0, len(norm_vals) - 1, 100)										#Resample to exactly 100 points: Regardless of whether a cell had 5 frames or 50 frames in this state → both get stretched/compressed to produce 100 evenly spaced values.
									new_y = interp_func(new_x)

									interp_df = pd.DataFrame({
									    'NETosisState': [k] * 100,
									    cell_name: new_y
									})
									interp_df_total = pd.concat([interp_df_total, interp_df], ignore_index=True)

							all_metric_data[metric]['Interpolated'].append(interp_df_total)


			# After all elements are processed, save per metric
			for metric in metrics:
				save_path = os.path.join(output_folder, f"{mask_channel_name}-{extrapolate_channel_name}-{metric}_data.xlsx")
				with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
					# Raw sheet
					if all_metric_data[metric]['Raw']:
						raw_df_combined = pd.DataFrame()
						for df in all_metric_data[metric]['Raw']:
							if df.empty:
								continue
							cell_name = df.columns[-1]  # metric column name is the cell name
							df.columns = [f'Time[min]_{cell_name}', f'NETosisState_{cell_name}', f'{cell_name}']
							raw_df_combined = pd.concat([raw_df_combined, df], axis=1)
						raw_df_combined.to_excel(writer, sheet_name='Raw', index=False)

					# Normalized sheet
					if all_metric_data[metric]['Normalized']:
						norm_df_combined = pd.DataFrame()
						for df in all_metric_data[metric]['Normalized']:
							if df.empty:
								continue
							cell_name = df.columns[-1]
							df.columns = [f'Time[min]_{cell_name}', f'NETosisState_{cell_name}', f'{cell_name}']
							norm_df_combined = pd.concat([norm_df_combined, df], axis=1)
						norm_df_combined.to_excel(writer, sheet_name='Normalized', index=False)

					# Interpolated sheet
					if all_metric_data[metric]['Interpolated']:
						interp_combined = pd.concat(all_metric_data[metric]['Interpolated'], axis=1)
						interp_combined = interp_combined.loc[:, ~interp_combined.columns.duplicated()]
						interp_combined.to_excel(writer, sheet_name='Interpolated', index=False)


# going to make a function that can visualize everything quickly
def func_plotter(settings, temp_folders, folder_addy = None, custom_file_ext = None):
	
	metrics = ['SD','TotInt','MeanInt','SDI','SD_Med','Circularity','Roundness','Area'] #this is hard coded and can be changed :)

	#not sure if this works it might throw an error
	if folder_addy == None:
		folder_addy = os.path.dirname(os.path.abspath(__file__))

	if custom_file_ext == None:
		custom_file_ext = 'data_dictionary_output'

	cell_dict = {}

	for i in range(len(settings['extrapolate_channel'])): #sifting through the channels i want to extrapolate
		for j in range(len(settings['extrapolate_channel'][i])):
			
			for k in range(4): #theres max 4 events that happen during NETosis
				
				for metric in metrics: #going to sift through each metric to interpolate it and then also normalize it
				
					name = settings['channels_names'][j] + ':State' + str(k) + ':Metric' + metric
					cell_dict[name] = []
					name_norm = settings['channels_names'][j] + ':State' + str(k) + ':Metric' + metric + ':NORM'
					cell_dict[name_norm] = []

					for element in temp_folders:
						temp_file = os.path.join(element, settings['channels_names'][j] + '_processed.xlsx') #temp_file = element + '\\' + settings['channels_names'][j] + '_processed.xlsx' #this is the file directory
						excel_file = pd.ExcelFile(temp_file) #opening up the excel file here

						#OG = pd.read_excel(excel_file, sheet_name='OG_data')
						interp = pd.read_excel(excel_file, sheet_name='Interpolated')

						try:
							cell_dict[name].append(interp[name])
							cell_dict[name_norm].append(interp[name_norm])
						except:
							a=5 #nothing sandwhich here just to get the coded running
							#this means that this cell did not get through all stages of NETosis

	# print(cell_dict['DNA_1x1:State1:MetricSD'])
	# print(len(cell_dict['DNA_1x1:State1:MetricSD']))

	# for i in range(len(cell_dict['DNA_1x1:State2:MetricSD:NORM'])):
	# 	t = np.linspace(0,len(cell_dict['DNA_1x1:State2:MetricSD:NORM'])-1, 100)
	# 	plt.plot(t,cell_dict['DNA_1x1:State2:MetricSD:NORM'][i])
	# 	plt.show()

	# t = np.linspace(0,len(cell_dict['DNA_1x1:State2:MetricSD:NORM'])-1, 100)
	# y = np.mean(cell_dict['DNA_1x1:State2:MetricSD:NORM'],axis=0)
	# y_std = np.std(cell_dict['DNA_1x1:State2:MetricSD:NORM'],axis=0)
	# plt.plot(t,y,color='blue')
	# plt.fill_between(t, y - y_std, y + y_std, color='blue', alpha=0.2)
	# y = np.mean(cell_dict['BF_1x1:State2:MetricSD:NORM'],axis=0)
	# y_std = np.std(cell_dict['BF_1x1:State2:MetricSD:NORM'],axis=0)
	# plt.plot(t,y,color='red')
	# plt.fill_between(t, y - y_std, y + y_std, color='red', alpha=0.2)
	# plt.show()

	# t = np.linspace(0,len(cell_dict['DNA_1x1:State0:MetricSD_Med:NORM'])-1, 100)
	# y = np.mean(cell_dict['DNA_1x1:State1:MetricSD_Med:NORM'],axis=0)
	# y_std = np.std(cell_dict['DNA_1x1:State1:MetricSD_Med:NORM'],axis=0)
	# plt.plot(t,y,color='blue')
	# plt.fill_between(t, y - y_std, y + y_std, color='blue', alpha=0.2)
	# y = np.mean(cell_dict['BF_1x1:State1:MetricSD_Med:NORM'],axis=0)
	# y_std = np.std(cell_dict['BF_1x1:State1:MetricSD_Med:NORM'],axis=0)
	# plt.plot(t,y,color='red')
	# plt.fill_between(t, y - y_std, y + y_std, color='red', alpha=0.2)
	# plt.show()

	# t = np.linspace(0,len(cell_dict['DNA_1x1:State0:MetricSD_Med'])-1, 100)
	# y = np.mean(cell_dict['DNA_1x1:State1:MetricSD_Med'],axis=0)
	# y_std = np.std(cell_dict['DNA_1x1:State1:MetricSD_Med'],axis=0)
	# plt.plot(t,y,color='blue')
	# plt.fill_between(t, y - y_std, y + y_std, color='blue', alpha=0.2)
	# y = np.mean(cell_dict['BF_1x1:State1:MetricSD_Med'],axis=0)
	# y_std = np.std(cell_dict['BF_1x1:State1:MetricSD_Med'],axis=0)
	# plt.plot(t,y,color='red')
	# plt.fill_between(t, y - y_std, y + y_std, color='red', alpha=0.2)
	# plt.show()	

	# t = np.linspace(0,len(cell_dict['DNA_1x1:State0:MetricRoundness'])-1, 100)
	# y = np.mean(cell_dict['DNA_1x1:State1:MetricRoundness'],axis=0)
	# y_std = np.std(cell_dict['DNA_1x1:State1:MetricRoundness'],axis=0)
	# plt.plot(t,y,color='blue')
	# plt.fill_between(t, y - y_std, y + y_std, color='blue', alpha=0.2)
	# y = np.mean(cell_dict['BF_1x1:State1:MetricRoundness'],axis=0)
	# y_std = np.std(cell_dict['BF_1x1:State1:MetricRoundness'],axis=0)
	# plt.plot(t,y,color='red')
	# plt.fill_between(t, y - y_std, y + y_std, color='red', alpha=0.2)
	# plt.show()

	# t = np.linspace(0,len(cell_dict['DNA_1x1:State0:MetricTotInt:NORM'])-1, 100)
	# y = np.mean(cell_dict['DNA_1x1:State1:MetricTotInt:NORM'],axis=0)
	# y_std = np.std(cell_dict['DNA_1x1:State1:MetricTotInt:NORM'],axis=0)
	# plt.plot(t,y,color='blue')
	# plt.fill_between(t, y - y_std, y + y_std, color='blue', alpha=0.2)
	# y = np.mean(cell_dict['BF_1x1:State1:MetricTotInt:NORM'],axis=0)
	# y_std = np.std(cell_dict['BF_1x1:State1:MetricTotInt:NORM'],axis=0)
	# plt.plot(t,y,color='red')
	# plt.fill_between(t, y - y_std, y + y_std, color='red', alpha=0.2)
	# plt.show()	

	# t = np.linspace(0,len(cell_dict['DNA_1x1:State1:MetricSDI'])-1, 100)
	# y = np.mean(cell_dict['DNA_1x1:State1:MetricSDI'],axis=0)
	# y_std = np.std(cell_dict['DNA_1x1:State1:MetricSDI'],axis=0)
	# plt.plot(t,y,color='blue')
	# plt.fill_between(t, y - y_std, y + y_std, color='blue', alpha=0.2)
	# y = np.mean(cell_dict['BF_1x1:State1:MetricSDI'],axis=0)
	# y_std = np.std(cell_dict['BF_1x1:State1:MetricSDI'],axis=0)
	# plt.plot(t,y,color='red')
	# plt.fill_between(t, y - y_std, y + y_std, color='red', alpha=0.2)
	# plt.show()




##here are the settings ! dictionary to say which channels are which
settings = {
	'channel_num' : 4,
	'channels' : ['DIC','647','488','561'],
	'channels_names' : ['DIC','DNA','Cholesterol','Mask'],
	'mask_channel' : [1,3], #if you are doing more, just add the channel associated number
	'extrapolate_channel' : [[2],[2]], #this should be in the same order as channels_names (if need more, add another list)
	#the extrapolate channel list inside list should correspond to the mask_channel list
	'ionomycin' : True,
	'erosion_factor': 0,
	'erosion_shape': 2, #this is like 2x2 or 3x3 etc
	'win_size_pref': [10,5,15],
	'NETosisStageAnnot': True
}

print(settings)












#the following files just quickly re-orgnizes the data so it is easier to work with
#it puts everything in a dictionary, dataframe, and csv which can be manipulated later
folder_addy = None
cell_folders, folder_addy = folder_finder(folder_addy)

#print(cell_folders,folder_addy)
#cell_files = file_list_process(cell_folders) #not needed cuz of the way i named everything
#print(cell_files) #not needed cuz of the way i named everything

files_looper(settings,cell_folders,folder_addy)
compiler(settings,cell_folders,settings['NETosisStageAnnot'],folder_addy) #this is based on NETosis state so will not work otherwise
# func_plotter(settings,cell_folders,folder_addy)

