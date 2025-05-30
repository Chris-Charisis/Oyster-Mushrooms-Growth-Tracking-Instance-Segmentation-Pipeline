from shapely.geometry import Polygon
import numpy as np
from Filtering_Utilities import dict_to_det_data_sample
import cv2

#Sorting clusters for tracking
def cluster_sort(polygons,polygons_info,baseline,image_result):

	#Establishing baseline
	if baseline == []:
		to_delete = []
		polygons_cropped = [[]]
		polygons_info_cropped = [[]]
		#Check for polygons to set baseline
		if polygons[-1] != []:
			i = 0
			for polygon in polygons[-1]:
				#Check that recognised mushrooms are immature
				if polygons_info[0][i][0] == 0:
					baseline.append([polygon,polygons_info[0][i]])
				#Remove mature clusters
				else:
					# print("Mature cluster detected, removing from baseline: ", polygons_info[0][i])
					to_delete.append(i)
				i += 1
			#New polygons/polygons info with mature clusters removed
			i = 0
			for base in baseline:
				if i not in to_delete:
					polygons_cropped[0].append(base[0])
					polygons_info_cropped[0].append(base[1])
				i += 1

		image_result = image_result.cpu().numpy().to_dict()

		image_result["pred_instances"]["bboxes"] = np.delete(image_result["pred_instances"]["bboxes"],to_delete, axis=0)
		image_result["pred_instances"]["scores"] = np.delete(image_result["pred_instances"]["scores"],to_delete, axis=0)
		image_result["pred_instances"]["masks"] = np.delete(image_result["pred_instances"]["masks"],to_delete, axis=0)
		image_result["pred_instances"]["labels"] = np.delete(image_result["pred_instances"]["labels"],to_delete, axis=0)

		image_result = dict_to_det_data_sample(image_result)
		#Exit the function after establishing baseline or if no polygons
		return polygons_cropped,polygons_info_cropped,baseline,image_result

	polygons[-1],polygons_info[-1],to_delete = polygon_sort(polygons[-1],polygons_info[-1],baseline)

	#Updating baseline
	for j in range(len(polygons[-1])):
		if len(polygons[-1][j]) > 1:
			if j < (len(baseline)):
				baseline[j] = (polygons[-1][j],polygons_info[-1][j])
			else:
				baseline.append([polygons[-1][j],polygons_info[-1][j]])


	image_result = image_result.cpu().numpy().to_dict()

	image_result["pred_instances"]["bboxes"] = np.delete(image_result["pred_instances"]["bboxes"],to_delete, axis=0)
	image_result["pred_instances"]["scores"] = np.delete(image_result["pred_instances"]["scores"],to_delete, axis=0)
	image_result["pred_instances"]["masks"] = np.delete(image_result["pred_instances"]["masks"],to_delete, axis=0)
	image_result["pred_instances"]["labels"] = np.delete(image_result["pred_instances"]["labels"],to_delete, axis=0)

	image_result = dict_to_det_data_sample(image_result)

	return polygons,polygons_info,baseline,image_result

def mask_from_poly(poly, out_shape, scale=1.0):
    """Rasterise polygon → boolean mask."""
    poly = np.squeeze(poly).astype(np.int32)
    poly = (poly * scale).astype(np.int32)
    mask = np.zeros(out_shape, np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)

def raster_inter_union(polyA, polyB, resolution=1.0):
    """
    Approx. intersection + union areas at chosen resolution (pixels / unit).
    """
    # 1. find overall bounding box
    both = np.vstack([polyA.reshape(-1, 2), polyB.reshape(-1, 2)])
    xmin, ymin = both.min(0)
    xmax, ymax = both.max(0)

    w = int(np.ceil((xmax - xmin) * resolution)) + 3
    h = int(np.ceil((ymax - ymin) * resolution)) + 3

    # 2. translate coords to mask space
    shift = np.array([[-xmin, -ymin]])
    mA = mask_from_poly(polyA + shift, (h, w), resolution)
    mB = mask_from_poly(polyB + shift, (h, w), resolution)

    inter = np.logical_and(mA, mB).sum() / (resolution**2)
    union = np.logical_or(mA, mB).sum()   / (resolution**2)
    return inter, union

#Calculating intersection over union for coordiante list 
def coordinate_iou(poly,base):

	# poly1 = Polygon(poly).buffer(0)
	# poly2 = Polygon(base).buffer(0)
	intersect, union = raster_inter_union(poly, base, resolution=1.0)
	#Getting intersection of both polygons
	# intersect = poly1.intersection(poly2).area
	# print(intersect, union)
	if intersect == 0:
		return 0
	
	#Getting union and intersection over union
	# union = poly1.union(poly2).area
	iou = intersect / union

	return iou

#Using intersection over union method to track the same mushrooms for coordinate lists 
def polygon_sort(polygons,polygons_info,baseline,iou_baseline = 0.2):

	temp = [[] for _ in range(len(baseline))]
	included = []

	#Iterate through the 'baseline polygons'
	i = 0
	for base in baseline:
		#Set maximum iou to iou_baseline (minimum acceptable iou)
		iou_max = iou_baseline
		best_fit = 0
		location = -1
		#Iterate through the next set of bounding boxes
		j = 0
		for polygon in polygons:
			#Looking through normal or empty boxes
			if len(base[0]) > 1:
				poly_iou = coordinate_iou(polygon,base[0])
				if poly_iou > iou_max:
					iou_max = poly_iou
					best_fit = [polygon,polygons_info[j]]
					location = j
			j += 1   

		if location != -1:
			included.append(location)
		
		#Setting best fit box 
		if iou_max == 1:
			temp[i] = [[0],[0]]
		elif iou_max > iou_baseline:
			temp[i] = best_fit
		else:
			temp[i] = [[0],[0]]

		i += 1
		
	for i in range(len(polygons)):
		#Adding new polygons
		if i not in included:
			if polygons_info[i][0] == 0:
				temp.append([polygons[i],polygons_info[i]])

	polygons_temp = [x[0] for x in temp]
	info_temp = [x[1] for x in temp]

	to_delete = []

	i = 0
	for poly in polygons:
		included = False
		for temp in polygons_temp:
			if np.all(poly[0] == temp[0]):
				included = True
		if not included:
			to_delete.append(i)
		i += 1

	return polygons_temp, info_temp, to_delete