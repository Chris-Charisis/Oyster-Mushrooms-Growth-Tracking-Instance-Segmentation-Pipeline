import cv2
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
import copy
import time
# from shapely.geometry import Polygon
from Metrics_Utilities import *
# from shapely.errors import GEOSException


##Create the working folder structure
def initialize_workspace(working_folder):
	os.makedirs(working_folder, exist_ok=True)
	os.makedirs(os.path.join(working_folder, "Predictions"), exist_ok=True)
	# os.makedirs(os.path.join(working_folder, "Untracked"), exist_ok=True)
	os.makedirs(os.path.join(working_folder, "Tracked"), exist_ok=True)
	os.makedirs(os.path.join(working_folder, "Substrate"), exist_ok=True)
	os.makedirs(os.path.join(working_folder, "Images"), exist_ok=True)

	return


##Making lengths of all polygon arrays equal
def equalize_polygons(polygons,polygons_info):

	#Finding longest set of polygons
	max_length = 0
	for poly in polygons:
		if len(poly) > max_length:
			max_length = len(poly)

	#Adding null points to make the list equal sizes
	i = 0
	for poly in polygons:
		while len(poly) < max_length:
			poly.append([0])
			polygons_info[i].append([0])
		i += 1

	return polygons, polygons_info


##Conversion of pixel area to 'real' area using real substrate size
def pixel_relative_area(pixel_area,averaged_width_pixels, averaged_height_pixels,substrate_real_width,substrate_real_height):

	return pixel_area*substrate_real_width*substrate_real_height/(averaged_height_pixels*averaged_width_pixels)


##Plotting the growth curves
def plot_growth(polygons,lines,working_folder):

	#Initializing x-axis
	x_axis = np.linspace(0,len(lines[-1]),num = len(lines[-1]))

	colors = cm.get_cmap('tab20', 20)

	fig, axs = plt.subplots()
	for i in range(len(polygons[-1])):
		axs.plot(x_axis,lines[i], label = 'Cluster {}'.format(i),color=colors(i))
	#Displaying the graphs
	axs.set_xlabel('Image Number')
	axs.set_ylabel('Relative Size by Pixel Number')
	axs.legend()
	plt.savefig(working_folder + 'Cluster Growth Curves.png')
	plt.show()

	return


##Isolate the cluster from the original image
def process_cluster(image_copy,poly,bounding,working_folder,test_img,j,label,labels_to_crop):
	#Check if cluster label is selected for cropping or all labels should be cropped
	if labels_to_crop[0]!=-1 and label not in labels_to_crop:
		return None,None
		
	#Get shape of the image
	height = image_copy.shape[0]
	width = image_copy.shape[1]

	minx = bounding[0]
	miny = bounding[1]
	maxx = bounding[2]
	maxy = bounding[3]
	
	#Limiting the upper boundaries to the maximum width and height
	limit_factor = 0.025
	limit_increase = 1 + limit_factor
	limit_decrease = 1 - limit_factor
	if maxy*limit_increase > height:
		uppery = height
	else:
		uppery = int(maxy*limit_increase)
	if maxx*limit_increase > width:
		upperx = width
	else:
		upperx = int(maxx*limit_increase)
		
	#Copying the cropped image section
	box_image = image_copy[int(miny*limit_decrease) : uppery, int(minx*limit_decrease) : upperx]
	#Converting from the polygon coordinates of the full picture to polygon coordinates in the box image
	local_poly = poly.copy()
	local_poly[:,0] = local_poly[:,0] - minx*limit_decrease
	local_poly[:,1] = local_poly[:,1] - miny*limit_decrease

	#Saving the bounded section of the image
	#cv2.polylines(box_image, np.int32([local_poly]), True, (255, 0, 0), 10)

	#Saving isolated cluster image
	os.makedirs(working_folder + "/Clusters/",exist_ok=True)
	cv2.imwrite(working_folder + "/Clusters/cluster_{}_{}".format(j,test_img), cv2.cvtColor(box_image,cv2.COLOR_RGB2BGR))

	return box_image,local_poly


# ##Saving the image information for an individual cluster in numpy array format
# def save_cluster_array(sizing_image,poly,centre,box_image,local_poly,working_folder,test_img,i,j):
# 	cv2.polylines(sizing_image, np.int32([poly]), True, (255, 0, 0), 10)
# 	cv2.putText(sizing_image, str(j), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)
# 	#Localizing polygon mask
# 	binary_mask = np.zeros((box_image.shape[0],box_image.shape[1]),int)
# 	for point in local_poly:
# 		binary_mask[point[1],point[0]] = 1
# 	array = [box_image[:,:,0],box_image[:,:,1],box_image[:,:,2],binary_mask]
# 	os.makedirs(working_folder + "/Arrays/",exist_ok=True)
# 	np.save(working_folder + "/Arrays/cluster_{}_{}".format(j,test_img),array)

# 	return


##Saving the various image types
def save_tracked_image(working_folder,test_img,full_image,):
	cv2.imwrite(working_folder + "/Tracked/tracked_" + test_img, cv2.cvtColor(full_image,cv2.COLOR_RGB2BGR))

	return


# ##Show results after filtering and before trackiing
# def save_untracked_image(img,polygons,working_folder,test_img):

# 	#Copying image
# 	untracked_img = img.copy()

# 	#Outling the polygons before tracking
# 	i = 0
# 	for poly in polygons[-1]:
# 		centre = Polygon(poly).centroid
# 		cv2.polylines(untracked_img, np.int32([poly]), True, (255, 0, 0), 10)
# 		#cv2.putText(untracked_img, 'Pred {} {}'.format(i,poly[0]), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
# 		cv2.putText(untracked_img, 'Pred {}'.format(i), (int(centre.x),int(centre.y)), cv2.FONT_HERSHEY_COMPLEX, 4, (0,255,0), 6, cv2.LINE_AA)
# 		i += 1

# 	#Saving the image with untracked polygons
# 	cv2.imwrite(working_folder + "/Untracked/untracked_" + test_img, cv2.cvtColor(untracked_img,cv2.COLOR_RGB2BGR))

# 	return

def numpy_centroid(raw_poly: np.ndarray):
    # 1. make it (N, 2) and float
    pts = np.squeeze(raw_poly).astype(float)

    # 2. need at least 3 unique vertices
    if len(np.unique(pts, axis=0)) < 3:
        raise ValueError("Polygon must have at least three distinct vertices")

    # 3. close the ring if it isn’t already
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # 4. shoelace area and centroid formula
    x, y = pts[:-1, 0], pts[:-1, 1]
    x_next, y_next = pts[1:, 0], pts[1:, 1]

    cross = x * y_next - x_next * y           # area summands
    area  = cross.sum() * 0.5                 # signed area

    if np.isclose(area, 0.0):
        raise ValueError("Polygon area is zero; centroid is undefined")

    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)

    return cx, cy

def numpy_area(raw_poly: np.ndarray) -> float:
    # 1. flatten (N,1,2) → (N,2) and cast to float
    pts = np.squeeze(raw_poly).astype(float)

    # 2. need at least 3 unique vertices
    if len(np.unique(pts, axis=0)) < 3:
        raise ValueError("Polygon must have at least three distinct vertices")

    # 3. close the ring if necessary
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # 4. shoelace / surveyor's formula
    x, y       = pts[:-1, 0], pts[:-1, 1]
    x_next, y_next = pts[1:, 0], pts[1:, 1]

    area = 0.5 * np.abs((x * y_next - x_next * y).sum())

    if np.isclose(area, 0.0):
        raise ValueError("Polygon area is zero; cannot compute centroid/area")

    return area

def cluster_sizing_and_visualization(image_for_visualization,polygons,polygons_info,labels_to_crop,cluster_sizing_option,crop_cluster_option,working_folder,test_img,img_num,averaged_width_pixels,averaged_height_pixels,substrate_real_width,substrate_real_height):
	if crop_cluster_option:
		image_without_cluster_annotation = copy.copy(image_for_visualization)
	
	#Saving the images with predicted polygons
	#Polygons from current image
	for j,poly in enumerate(polygons[-1]):
		
		#Skip polygon placeholders
		if len(poly) > 1:
			poly.reshape(-1,2)
			# min_0 = np.amin(poly[:,0])
			# max_0 = np.amax(poly[:,0])
			# min_1 = np.amin(poly[:,1])
			# max_1 = np.amax(poly[:,1])
			# centre_x = (max_0 + min_0) // 2
			# centre_y = (max_1 + min_1) // 2
			centre_x, centre_y = numpy_centroid(poly)
			if cluster_sizing_option:
				#Calculating estimated relative cluster area from pixel area
				relative_cluster_area = pixel_relative_area(numpy_area(poly),averaged_width_pixels,averaged_height_pixels,substrate_real_width,substrate_real_height)
				
				#Updating cluster segments
				cluster_segments = ([img_num+1,j,numpy_area(poly),relative_cluster_area,polygons_info[-1][j]])
				
				#Update dynamic cluster sizing csv
				write_cluster_sizing(cluster_segments,working_folder,test_img,img_num)

			#Saving the image with outlined clusters
			cv2.polylines(image_for_visualization,np.int32([poly]),True,(255, 0, 0),4)
			cv2.putText(image_for_visualization,'{}'.format(j),(int(centre_x),int(centre_y)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1,cv2.LINE_AA)
			#Saving the image information for an individual cluster in numpy array format
			if crop_cluster_option:
				#Isolate and save the cluster from the original image
				process_cluster((image_without_cluster_annotation,cv2.COLOR_RGB2BGR)[0],poly,polygons_info[-1][j][5],working_folder,test_img,j,polygons_info[-1][j][0],labels_to_crop)

	#Saving image with tracked instances
	save_tracked_image(working_folder,test_img,image_for_visualization)

	return

def plot_area_growth_curves(working_folder,label_mapping,label_colors,time_interval,show_plots=False):
	
	# time.sleep(1)
	# Load generated growth monitoring size information file
	growth_data = pd.read_csv(working_folder + "Cluster_Sizing.csv")
	
	plt.figure(figsize=(10, 6))
	num_clusters = int(max(growth_data["Cluster Tracking ID"]) + 1)
	# Use "tab20" colormap for distinct colors
	colors = cm.get_cmap("tab20", num_clusters)

	for cluster_id, group in growth_data.groupby("Cluster Tracking ID"):
		plt.plot(time_interval * group["Image #"], group["Relative Cluster Area"], label=f"Cluster {cluster_id}",
					color=colors(cluster_id))

	plt.ylabel("Relative Cluster Area (cm\u00b2)", fontsize=14, weight="bold")
	plt.xlabel("Time (hours)", fontsize=14, weight="bold")
	plt.legend(title="Cluster ID", title_fontsize=14, fontsize=9)
	plt.savefig(working_folder + "clusters_relative_area.png", dpi=300, bbox_inches="tight")
	if show_plots:
		plt.show()

	# Determine the number of unique clusters
	unique_clusters = growth_data["Cluster Tracking ID"].unique()
	num_clusters = len(unique_clusters)

	# Set up the figure and subplots
	fig, axes = plt.subplots(nrows=int(math.ceil(num_clusters / 3)), ncols=3, figsize=(18, num_clusters * 2))
	axes = axes.flatten()

	# Create scatter plots for each cluster value in subplots
	for i, cluster_id in enumerate(unique_clusters):
		cluster_df = growth_data[growth_data["Cluster Tracking ID"] == cluster_id]

		ax = axes[i]
		# Plot each scatter point based on class
		for label_value in cluster_df["Class"].unique():
			label_group = cluster_df[cluster_df["Class"] == label_value]
			ax.scatter(label_group["Image #"], label_group["Relative Cluster Area"],
						label=label_mapping.get(label_value), color=label_colors.get(label_value))

		ax.set_xlabel("Time (hours)", fontsize=14, weight="bold")
		ax.set_ylabel("Relative Cluster Area (cm\u00b2)", fontsize=14, weight="bold")
		ax.legend(title="Class", title_fontsize=16, fontsize=12, loc="lower right")

	# Remove any unused subplots
	for j in range(i + 1, len(axes)):
		fig.delaxes(axes[j])

	plt.tight_layout()
	plt.savefig(working_folder + "individual_relative_cluster_area.png", dpi=300)
	if show_plots:
		plt.show()

	return

