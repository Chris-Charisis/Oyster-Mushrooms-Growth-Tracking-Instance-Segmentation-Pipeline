import csv
import json
import numpy as np
from Tracking_Utilities import coordinate_iou
import torch
import torchvision.ops.boxes as bops


##Convert COCO to VOC annotation format
def COCO_to_VOC_bbox(bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    return [x1,y1,x2,y2]


##Read and ingest annotations for tracking
def annotation_tracking(file):
	
    #Set input and output paths
    tracked_annotations_save_path = "{}_tracked.json".format(file.replace('.json',''))
    #Open input annotation file
    with open(file) as f:
        annotations = json.load(f)

    #Initialize variable for baseline bboxes
    baseline_bboxes = []
    for idx,annotation in enumerate(annotations["annotations"]):
        #Convert COCO x,y,h,w to VOC x1,y1,x2,y2 for correct results in the similarity function
        converted_bbox = COCO_to_VOC_bbox(annotation["bbox"])
        #When baseline_bboxes is not empty
        if baseline_bboxes:
            #Calculate bbox iou between the new bbox and all existing baseline bboxes
            bboxes_iou = []
            for baseline_bbox in baseline_bboxes:
                bboxes_iou.append(bops.box_iou(torch.tensor([converted_bbox], dtype=torch.float), torch.tensor([baseline_bbox], dtype=torch.float))[0][0].item())
            #If all baseline bbox low iou with the new bbox then we have a new instance and we add the new bbox to the baseline
            if (np.array(bboxes_iou)<0.4).all():
                #Add the new bbox to the baseline
                baseline_bboxes.append(converted_bbox)
                #Add the tracking_id field to the annotation variable, which is the found index position
                annotation["tracking_id"] = len(baseline_bboxes)-1
            #Else if there is one baseline bbox with high iou with the new bbox then we have an instance match
            else:
                #Numpy array to list for code simplicity
                #Find the index of the baseline bbox that matches, bboxes_iou and baseline_bboxes have the same order
                respective_baseline_bbox_index = bboxes_iou.index(max(bboxes_iou))
                #Update the index position of the baseline_bboxes variable with the new bbox
                baseline_bboxes[respective_baseline_bbox_index] = converted_bbox
                #Add the tracking_id field to the annotation variable, which is the found index position
                annotation["tracking_id"] = respective_baseline_bbox_index
        #If baseline_bboxes is empty add the first element to initialize the process
        else:
            #Add the first bbox to the baseline
            baseline_bboxes.append(converted_bbox)
            #Add the tracking_id field to the annotation variable, which is the found index position
            annotation["tracking_id"] = 0

    #Save the new tracked annotations
    with open(tracked_annotations_save_path,"w") as f:
        json.dump(annotations,f)

    #Read the new json file
    annotations, tracking_annotations = get_annotations_json(tracked_annotations_save_path)

    return annotations, tracking_annotations


##Establishing cluster_segments excel file
def establish_cluster_sizing(working_folder):
    with open(working_folder + 'Cluster_Sizing.csv', 'w') as csv_file:
        #Creating the csv writer
        writer = csv.writer(csv_file)
        #Writing the first row with all the headers
        writer.writerow(['Image filename',"Image #",'Cluster Tracking ID','Class','Cluster Pixel Area','Relative Cluster Area','Cluster Height','Cluster Width','Relative Height','Relative Width'])

    return


##Establishing csv file for multiple object tracking accuracy
def establish_mota(working_folder):
    with open(working_folder + 'MOTA_Metrics.csv', 'w') as csv_file:
        #Creating the csv writer
        writer = csv.writer(csv_file)
        #Writing the first row with all the headers
        writer.writerow(['Image #','MOTA 50','FP 50','Cumulative FP 50','FN 50','Cumulative FN 50','IDS 50', 'Cumulative IDS 50','GT 50','Cumulative GT 50'])

    return


##Getting annotations from text file (JSON Coco format)
def get_annotations_json(text_file):

	#Pathway to json file
	with open(text_file, 'r') as file:
		#Reading from json file
		annotation_data = json.load(file)

	#Annotations for each image
	annotations = [[] for _ in range(len(annotation_data['images']))]
	tracking_annotations = [[] for _ in range(len(annotation_data['images']))]
	
	for annotation in annotation_data['annotations']:

		#Pairing each pair of x,y points together
		points = annotation['segmentation'][0]
		segment = []
		pair = []
		for point in points:
			pair.append(int(point))
			if len(pair) == 2:
				segment.append(pair)
				pair = []
		
		#Full polygon added to corresponding image
		annotations[annotation['image_id']-1].append([segment,annotation['category_id']])
		tracking_annotations[annotation['image_id']-1].append([segment,annotation['tracking_id'],annotation['category_id']])

	return annotations, tracking_annotations


##Calculate TP, FP and FN for the multi-class task
def calculate_multiclass_TP_FP_FN(annotations,polygons,polygons_info):
    per_class = {
        0 : {
            "TP" : [0,0,0,0,0,0,0,0,0,0],
            "FP" : [0,0,0,0,0,0,0,0,0,0],
            "FN" : [0,0,0,0,0,0,0,0,0,0]
        },
        1 : {
            "TP" : [0,0,0,0,0,0,0,0,0,0],
            "FP" : [0,0,0,0,0,0,0,0,0,0],
            "FN" : [0,0,0,0,0,0,0,0,0,0]
        },
        2 : {
            "TP" : [0,0,0,0,0,0,0,0,0,0],
            "FP" : [0,0,0,0,0,0,0,0,0,0],
            "FN" : [0,0,0,0,0,0,0,0,0,0]
        }
    }
    
    #Iterate through all images
    for idx in range(len(annotations)):
        polygon_metrics = []
        polygon_metrics_info = []
        #Check every polygon of the image if it is not a placeholder
        for i,poly in enumerate(polygons[idx]):
            if isinstance(poly, int):
                continue          
            if len(poly) > 1:
                polygon_metrics.append(poly)
                polygon_metrics_info.append(polygons_info[idx][i])

        if not polygon_metrics:
            # print("empty predictions", idx)
            continue

        else:
            #For every threshold
            for k,iou_threshold in enumerate(range(50,100,5)):
                iou_threshold /= 100

                #To check whether an annotation/polygon have been recognized/matched 
                annotations_check = [False for _ in range(len(annotations[idx]))]

                #Iterate valid polygons
                for j,polygon in enumerate(polygon_metrics):
                    matched = False
                    #Iterate annotations
                    for i,annotation in enumerate(annotations[idx]):
                        iou = coordinate_iou(annotation[0],polygon)
                        
                        #Each annotation should only be detected once. Extra detections are false positives
                        #Label condition also being checked
                        if iou >= iou_threshold and annotations_check[i] == False and annotation[1] == (polygon_metrics_info[j][0]+1):
                            per_class[polygon_metrics_info[j][0]]["TP"][k] += 1
                            annotations_check[i] = True
                            matched = True
                            break
                    if not matched:
                        per_class[polygon_metrics_info[j][0]]["FP"][k] += 1

                for idy,check in enumerate(annotations_check):
                    #If the annotation was matched move to the next one
                    if check:
                        continue
                    else:
                        per_class[annotations[idx][idy][1]-1]["FN"][k] += 1
    
    return per_class


##Calculate TP, FP and FN for the single-class task
def calculate_singleclass_TP_FP_FN(annotations,polygons,polygons_info):
    per_class = {
        0 : {
            "TP" : [0,0,0,0,0,0,0,0,0,0],
            "FP" : [0,0,0,0,0,0,0,0,0,0],
            "FN" : [0,0,0,0,0,0,0,0,0,0]
        }
    }
    #For every annotated imge
    for idx in range(len(annotations)):

        #Removing placeholder [0] from list for metrics
        polygon_metrics = []
        polygon_metrics_info = []
        #Check every polygon of the image if it is not a placeholder
        for i,poly in enumerate(polygons[idx]):
            if isinstance(poly, int):
                continue
            if len(poly) > 1:
                polygon_metrics.append(poly)
                polygon_metrics_info.append(polygons_info[idx][i])

        if not polygon_metrics:
            print("empty predictions", idx)
        else:
            #For every threshold
            for k,iou_threshold in enumerate(range(50,100,5)):
                iou_threshold /= 100

                #To check whether an annotation/polygon have been recognized/matched 
                annotations_check = [False for _ in range(len(annotations[idx]))]

                #Iterate valid polygons
                for j,polygon in enumerate(polygon_metrics):
                    matched = False
                    #Iterate annotations
                    for i,annotation in enumerate(annotations[idx]):
                        iou = coordinate_iou(annotation[0],polygon)

                        #Each annotation should only be detected once. Extra detections are false positives
                        #Label condition also being checked
                        if iou >= iou_threshold and annotations_check[i] == False:
                            per_class[0]["TP"][k] += 1
                            annotations_check[i] = True
                            matched = True
                            break
                    if not matched:
                        per_class[0]["FP"][k] += 1

                for idy,check in enumerate(annotations_check):
                    #If the annotation was matched move to the next one
                    if check:
                        continue
                    else:
                        per_class[0]["FN"][k] += 1

    return per_class


##Element-wise addition of two nested dictionaries
def add_dicts(dict1, dict2):
    result = {}

    #Get all unique keys from both dictionaries
    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        result[key] = {}

        #Get inner dictionaries
        dict1_inner = dict1.get(key, {})
        dict2_inner = dict2.get(key, {})

        #Get all keys in the inner dictionaries
        inner_keys = set(dict1_inner.keys()).union(set(dict2_inner.keys()))

        for inner_key in inner_keys:
            #Default to zero list
            list1 = dict1_inner.get(inner_key, [0] * 10)
            list2 = dict2_inner.get(inner_key, [0] * 10)

            #Perform element-wise addition
            result[key][inner_key] = [a + b for a, b in zip(list1, list2)]

    return result


##Compute AP for a single class using 11-point interpolation (COCO method)
def compute_ap(tp, fp, fn):
    tp = np.array(tp, dtype=np.float32)
    fp = np.array(fp, dtype=np.float32)
    # fn = np.array(fn, dtype=np.float32)

    #Compute precision and recall for each IoU threshold
    precision = tp / (tp + fp)
    # recall = tp / (tp + fn)

    # #Sort recall values to ensure they are non-decreasing
    # sorted_indices = np.argsort(recall)
    # recall = recall[sorted_indices]
    # precision = precision[sorted_indices]

    # #Ensure precision is non-increasing (COCO rule)
    # precision = np.maximum.accumulate(precision[::-1])[::-1]

    # #Compute AP using 11-point interpolation
    # ap = 0
    # #11 recall levels: [0, 0.1, ..., 1]
    # recall_levels = np.linspace(0, 1, 11)
    # for r in recall_levels:
    #     max_precision = np.max(precision[recall >= r]) if np.any(recall >= r) else 0
    #     ap += max_precision / 11

    return np.mean(precision)


##Compute COCO-style Recall for a single class
def compute_coco_recall(tp, fn):
    tp = np.array(tp, dtype=np.float32)
    fn = np.array(fn, dtype=np.float32)

    #Compute recall at each IoU threshold
    recall_per_threshold = tp / (tp + fn)

    #Return average recall across all IoU thresholds
    return np.mean(recall_per_threshold)


##Compute AP and Recall for each class
def compute_mAP_mAR(results):
    ap_per_class = {}
    recall_per_class = {}

    for cls, data in results.items():
        ap_per_class[cls] = compute_ap(data['TP'], data['FP'], data['FN'])
        recall_per_class[cls] = compute_coco_recall(data['TP'], data['FN'])

    #Compute mAP (Mean Average Precision) as the mean of all class APs
    mAP = np.mean(list(ap_per_class.values()))

    #Compute Mean Recall (averaged across all classes)
    mean_recall = np.mean(list(recall_per_class.values()))

    if len(ap_per_class)>1:
        #Print results
        print("AP per class:")
        for cls, ap in ap_per_class.items():
            print(f"Class {cls}: AP = {ap:.3f}")
    
        print("Recall per class:")
        for cls, recall in recall_per_class.items():
            print(f"Class {cls}: Recall = {recall:.3f}")

    F1_SCore = 2*mAP*mean_recall/(mAP+mean_recall)

    print(f"\nMean Average Precision (mAP): {mAP:.3f}")
    print(f"Mean Recall: {mean_recall:.3f}")
    print(f"F1-Score: {F1_SCore:.3f}")
    
    return


##Calculate tracking metric MOTA
def get_tracking_metrics(annotations,polygons,mota_metrics_50, motaTracker_50):
    GT = len(annotations)
    mota_metrics_50[3].append(GT)

    max_id = -1
    for annotation in annotations:
            if annotation[1] > max_id:
                max_id = annotation[1]
	
    i = 0
    while i < 1:
        if i == 0:
            iou_threshold = 0.5
        mota_id = [[] for _ in range(max(len(polygons),len(annotations)))]
        #Track how many of the polygons have corresponding ground truth annotations
        polygon_check = 0
        FN = 0
        IDS = 0
        for annotation in annotations:

            tracking_id = annotation[1]
            annotation = annotation[0]			
            #To check if an annotation has been tracked
            annotation_tracked = False
            max_iou = 0
            index = 0
            max_index = -1
            for polygon in polygons:

                if len(polygon) > 1:

                    iou = coordinate_iou(annotation,polygon)
                
                    if iou > iou_threshold:
                        #Increment polygon check (max once per annotation)
                        if not annotation_tracked:
                            polygon_check += 1
                        #Annotation has been tracked
                        annotation_tracked = True
                        #In case of overlapping annotations
                        if iou > max_iou:
                            max_iou = iou
                            max_index = index

                index += 1
            
            if max_index != -1:
                mota_id[max_index].append(tracking_id)
                #Check that tracker has been established
                if i == 0:
                    if len(motaTracker_50) > 0:
                        #Check that tracker isn't being compared to a new cluster
                        if len(motaTracker_50[-1]) > max_index: 
                            if len(motaTracker_50[-1][max_index]) > 0 and motaTracker_50[-1][max_index][0] != tracking_id:
                                IDS += 1

            #Annotation has not been tracked	
            if not annotation_tracked:
                FN += 1

        polygon_count = 0
        for polygon in polygons:
            if len(polygon) > 1:
                polygon_count += 1

        FP = polygon_count - polygon_check

        if i == 0:
            mota_metrics_50[0].append(FP)
            mota_metrics_50[1].append(FN)
            mota_metrics_50[2].append(IDS)
            motaTracker_50.append(mota_id)
        
        i += 1

    return mota_metrics_50, motaTracker_50


##Writing information from cluster_segments to excel file
def write_cluster_sizing(segment,working_folder,test_img,img_num):
    with open(working_folder + 'Cluster_Sizing.csv', 'a') as csv_file:
        print(csv_file)
        #Creating the csv writer
        writer = csv.writer(csv_file,lineterminator='\n')
        #Writing new row
        writer.writerow([test_img,img_num+1,segment[1],segment[4][0],segment[2],segment[3],segment[4][1],segment[4][2],segment[4][3],segment[4][4]])
    
    return


##Calculating MOTA from timelapse collected information
def compute_mota(mota_metrics_50):

    cum_FP_50 = sum(mota_metrics_50[0])
    cum_FN_50 = sum(mota_metrics_50[1])
    cum_IDS_50 = sum(mota_metrics_50[2])
    cum_GT_50 = sum(mota_metrics_50[3])

    mota_50 = 1 - ((cum_FP_50 + cum_FN_50 + cum_IDS_50)/cum_GT_50)

    return round(mota_50,3)


def calculate_metrics(annotations, pre_filtering_polygons, pre_filtering_polygons_info, post_filtering_polygons,
                      post_filtering_polygons_info,mota_metric):
    # Calculate metrics for test timelapse
    pre_multiclass_TP_FP_FN = calculate_multiclass_TP_FP_FN(annotations, pre_filtering_polygons,
                                                            pre_filtering_polygons_info)
    pre_singleclass_TP_FP_FN = calculate_singleclass_TP_FP_FN(annotations, pre_filtering_polygons,
                                                              pre_filtering_polygons_info)
    post_multiclass_TP_FP_FN = calculate_multiclass_TP_FP_FN(annotations, post_filtering_polygons,
                                                             post_filtering_polygons_info)
    post_singleclass_TP_FP_FN = calculate_singleclass_TP_FP_FN(annotations, post_filtering_polygons,
                                                               post_filtering_polygons_info)
    print("Multi-class pre-filtering metrics:")
    compute_mAP_mAR(pre_multiclass_TP_FP_FN)
    print("----------------------")
    print("Multi-class post-filtering metrics:")
    compute_mAP_mAR(post_multiclass_TP_FP_FN)
    print("----------------------")
    print("Single-class pre-filtering metrics:")
    compute_mAP_mAR(pre_singleclass_TP_FP_FN)
    print("----------------------")
    print("Single-class post-filtering metrics:")
    compute_mAP_mAR(post_singleclass_TP_FP_FN)
    print("----------------------")
    print("MOTA metric:")
    print(compute_mota(mota_metric))

    return