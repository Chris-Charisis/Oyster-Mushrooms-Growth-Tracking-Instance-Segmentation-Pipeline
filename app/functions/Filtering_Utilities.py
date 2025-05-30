import math
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.structures.instance_data import InstanceData
import numpy as np
from PIL import Image, ImageStat
import torch 
import torchvision.ops.boxes as bops


#Calculating iou for bounding boxes
def box_iou(box_1,box_2):

	#Getting intersection width and height
	intersection_width = min(box_1[2],box_2[2]) - max(box_1[0],box_2[0]) 
	intersection_height = min(box_1[3],box_2[3]) - max(box_1[1],box_2[1]) 

	#No intersection
	if intersection_height <= 0 or intersection_width <= 0:
		return 0

	#Caclulating intersection area and area of each box
	intersection_area = intersection_width*intersection_height

	box_1_area = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
	box_2_area = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])

	#One box completely within another box
	if box_1_area == intersection_area or box_2_area == intersection_area:
		return 1

	union_area = box_1_area + box_2_area - intersection_area

	return intersection_area/union_area


#Adjust image brightness for processing
def calculate_brightness(im_file):
    im = Image.open(im_file)        
    stat = ImageStat.Stat(im)        
    r,g,b = stat.mean        
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

#check if brightness outlier exists
def brightness_filter(path_to_image,last_valid_ema,last_valid_residuals,alpha=0.1,z_thresh=30,residual_window_size=5):
    # Calculate brightness value
    brightness = calculate_brightness(path_to_image)

    # If no valid EMA, initialize it
    if last_valid_ema is None:
        return False, brightness, [0.0]

    # Predict EMA and compute residual
    ema = alpha * brightness + (1 - alpha) * last_valid_ema
    residual = brightness - last_valid_ema

    # Compute z-score from residual history
    if len(last_valid_residuals) >= 5:
        residuals_to_use = last_valid_residuals[-residual_window_size:]
        mean = np.mean(residuals_to_use)
        std = np.std(residuals_to_use)
        z = (residual - mean) / std if std > 0 else 0.0
        is_outlier = z > z_thresh
    else:
        is_outlier = False

    if is_outlier:
        print("Image with outlier brightness detected and skipped: ", path_to_image)
        print("--------------------------------------------")
        # Do not update EMA or residuals
        return True, last_valid_ema, last_valid_residuals
    else:
        # Update EMA and residuals
        updated_residuals = last_valid_residuals + [residual]
        if len(updated_residuals) > residual_window_size:
            updated_residuals = updated_residuals[-residual_window_size:]
        return False, ema, updated_residuals


##Dict to MMDetection InstanceData class
def dict_to_instance_data(instance_dict):
    instance_data = InstanceData()
    for key, value in instance_dict.items():
        setattr(instance_data, key, value)
    return instance_data

##Dict to MMDetection DetDataSample class
def dict_to_det_data_sample(data_dict):
    det_data_sample = DetDataSample()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            #If the value is a dictionary, we assume it is an InstanceData
            setattr(det_data_sample, key, dict_to_instance_data(value))
        else:
            setattr(det_data_sample, key, value)
    return det_data_sample


##Filter predictions with low condidence score
def delete_low_confidence_predictions(result,confidence_score_threshold):
    result = result.cpu().numpy().to_dict()
    low_conf_indices = []
    #Find which predictions are of low classification/confidence score and keep their index for deletion
    for idx in range(len(result["pred_instances"]["scores"])):
        if result["pred_instances"]["scores"][idx]<confidence_score_threshold:
            low_conf_indices.append(idx)
        elif np.sum(result["pred_instances"]["masks"][idx])/result["pred_instances"]["masks"][idx].size < 0.0006:
            low_conf_indices.append(idx)

    #Delete from all components of the result variable the instances with low classification/confidence score
    result["pred_instances"]["bboxes"] = np.delete(result["pred_instances"]["bboxes"],low_conf_indices, axis=0)
    result["pred_instances"]["scores"] = np.delete(result["pred_instances"]["scores"],low_conf_indices, axis=0)
    result["pred_instances"]["masks"] = np.delete(result["pred_instances"]["masks"],low_conf_indices, axis=0)
    result["pred_instances"]["labels"] = np.delete(result["pred_instances"]["labels"],low_conf_indices, axis=0)

    return dict_to_det_data_sample(result)


#Filter overlapping predictions with lower confidence score
def delete_overlapping_with_lower_confidence(result,iou_threshold):
    result = result.cpu().numpy().to_dict()
    to_delete = []
    #Iterate through all existing pairs of predictions
    for idx in range(len(result["pred_instances"]["bboxes"])):
        for idy in range(idx+1,len(result["pred_instances"]["bboxes"])):
            #Create the pair of bounding boxes to be examined
            pred1_bbox = torch.tensor([result["pred_instances"]["bboxes"][idx]], dtype=torch.float)
            pred2_bbox = torch.tensor([result["pred_instances"]["bboxes"][idy]], dtype=torch.float)
            #calculate iou of bounding boxes pair
            iou = bops.box_iou(pred1_bbox, pred2_bbox)
            #If iou is above a defined threshold the two prediction are referring to the same instance,
            #so we find the one with the lower classification/confidence score and keep its index to be deleted
            if iou>iou_threshold:
                if result["pred_instances"]["scores"][idx]>result["pred_instances"]["scores"][idy]:
                    to_delete.append(idy)
                else:
                    to_delete.append(idx)
    
    #Delete from all components of the result variable the overlapping instances with classification/confidence score
    result["pred_instances"]["bboxes"] = np.delete(result["pred_instances"]["bboxes"],to_delete, axis=0)
    result["pred_instances"]["scores"] = np.delete(result["pred_instances"]["scores"],to_delete, axis=0)
    result["pred_instances"]["masks"] = np.delete(result["pred_instances"]["masks"],to_delete, axis=0)
    result["pred_instances"]["labels"] = np.delete(result["pred_instances"]["labels"],to_delete, axis=0)

    return dict_to_det_data_sample(result)


##Expand a bounding box by a given scale factor
def expand_box(box, scale_factor):
    #Calculate the width and height of the box
    width = box[2] - box[0]
    height = box[3] - box[1]

    #Calculate the expansion in both width and height
    delta_w = width * scale_factor
    delta_h = height * scale_factor

    #Expand the box
    x_min_expanded = box[0] - delta_w / 2
    y_min_expanded = box[1] - delta_h / 2
    x_max_expanded = box[2] + delta_w / 2
    y_max_expanded = box[3] + delta_h / 2

    #Return the expanded box
    expanded_box = torch.tensor([[x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded]])
    return expanded_box


##Filter out erroneous cluster predictions from background and residual areas from harvesting
def delete_post_background_clusters(result,substrate_result, post_harvest_polygons_info_base,iou_threshold,verbose=False):
    result = result.cpu().numpy().to_dict()
    substrate_bbox = expand_box(substrate_result.cpu()["bboxes"][0],0.05)
    # print("Substrate result:", substrate_result)
    # print(substrate_bbox)
    to_delete = []
    for idy in range(len(result["pred_instances"]["bboxes"])):
        prediction_bbox = torch.tensor([result["pred_instances"]["bboxes"][idy]], dtype=torch.float)
        substrate_iou = bops.box_iou(substrate_bbox, prediction_bbox)
        #First check is to have common area with the expanded substrate, this filters out the far away instances
        if substrate_iou==0 and not post_harvest_polygons_info_base:
            to_delete.append(idy)
            continue
            
        preds_iou = []
        pred_intersection_iou = []
        base_intersection_iou = []
        for idx in range(len(post_harvest_polygons_info_base)):
            baseline_bbox = torch.tensor([post_harvest_polygons_info_base[idx][5]], dtype=torch.float)
            preds_iou.append(bops.box_iou(baseline_bbox, prediction_bbox))

            #Compute intersection bbox coordinates
            x_min_inter = torch.max(baseline_bbox[0][0], prediction_bbox[0][0])
            y_min_inter = torch.max(baseline_bbox[0][1], prediction_bbox[0][1])
            x_max_inter = torch.min(baseline_bbox[0][2], prediction_bbox[0][2])
            y_max_inter = torch.min(baseline_bbox[0][3], prediction_bbox[0][3])
            
            #Intersection box
            intersection_box = torch.tensor([[x_min_inter, y_min_inter, x_max_inter, y_max_inter]])

            pred_intersection_iou.append(bops.box_iou(prediction_bbox, intersection_box))
            base_intersection_iou.append(bops.box_iou(baseline_bbox, intersection_box))
            
            
        if any(pred_iou>=iou_threshold for pred_iou in preds_iou):
            if verbose:
                print("Matched cluster", idy)
            continue
        elif all(pred_iou==0 for pred_iou in preds_iou) and substrate_iou!=0:
            if verbose:
                print("New candidate cluster", idy)
            continue      
        elif substrate_iou==0:
            if verbose:
                print("Away from substrate",idy)
            to_delete.append(idy) 
        else:
            #Find the non-zero iou for intersection against baseline and prediction
            base_inter_iou_index = [i for i, e in enumerate(base_intersection_iou) if e != 0]
            pred_inter_iou_index = [i for i, e in enumerate(pred_intersection_iou) if e != 0]
            
            if 1 in pred_intersection_iou:
                if verbose:
                    print("Residual harvested area filtered", idy)
                to_delete.append(idy)
            elif all(base_inter_iou<=0.5 for base_inter_iou in base_intersection_iou):
                if verbose:
                    print("Background occluded filtered", idy)
                to_delete.append(idy)
            else:
                if verbose:
                    print("Retracked occluded or temporarily hidden cluster", idy)
                continue

    #Delete from all components of the result variable the overlapping instances with classification/confidence score
    result["pred_instances"]["bboxes"] = np.delete(result["pred_instances"]["bboxes"],to_delete, axis=0)
    result["pred_instances"]["scores"] = np.delete(result["pred_instances"]["scores"],to_delete, axis=0)
    result["pred_instances"]["masks"] = np.delete(result["pred_instances"]["masks"],to_delete, axis=0)
    result["pred_instances"]["labels"] = np.delete(result["pred_instances"]["labels"],to_delete, axis=0)

    return dict_to_det_data_sample(result)
