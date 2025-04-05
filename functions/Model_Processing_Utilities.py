from concave_hull import concave_hull_indexes
import cv2
from mmdet.apis import init_detector,inference_detector
from mmdet.registry import VISUALIZERS
from mmengine import Config
import numpy as np
import os
import torch 

##Checking if CUDA is available on the device running the program
def check_cuda():
	if torch.cuda.is_available():
		use_device = "cuda"
	else:
		use_device = "cpu"
	print(use_device)
	return use_device

##Loading models prediction
def load_models(mushroom_model_config_folder,substrate_model_config_folder,reference_model_config_folder,use_device):
	
	#Load the trained model
	mushroom_weights = [x for x in os.listdir(mushroom_model_config_folder) if x.endswith("pth")][0]
	substrate_weights = [x for x in os.listdir(substrate_model_config_folder) if x.endswith("pth")][0]
	if reference_model_config_folder is not None:
		reference_weights = [x for x in os.listdir(reference_model_config_folder) if x.endswith("pth")][0]

	#Load the configuration file
	mushroom_model_config_file = mushroom_model_config_folder + [x for x in os.listdir(mushroom_model_config_folder) if x.endswith(".py")][0]
	substrate_model_config_file = substrate_model_config_folder + [x for x in os.listdir(substrate_model_config_folder) if x.endswith(".py")][0]
	if reference_model_config_folder is not None:
		reference_config_file = reference_model_config_folder + [x for x in os.listdir(reference_model_config_folder) if x.endswith(".py")][0]

	#Initiate the model
	mushroom_model = init_detector(Config.fromfile(mushroom_model_config_file), mushroom_model_config_folder + mushroom_weights, device=use_device)
	substrate_model = init_detector(Config.fromfile(substrate_model_config_file), substrate_model_config_folder + substrate_weights, device=use_device)
	reference_model = None
	if reference_model_config_folder is not None:
		reference_model = init_detector(Config.fromfile(reference_config_file), reference_model_config_folder + reference_weights, device=use_device)

	#Initiate visualizer(run the block only once in jupyter notebook)
	visualizer = VISUALIZERS.build(mushroom_model.cfg.visualizer)
	#The dataset_meta is loaded from the model's configuration and then passed to the model in init_detector
	visualizer.dataset_meta = mushroom_model.dataset_meta

	return mushroom_model,substrate_model,reference_model,visualizer

##Process substrate, calculate size and save results
def process_substrate(substrate_model,reference_model,img,image_for_visualization,save_substrate_bbox_image,save_reference_bbox_image,working_folder,test_img,detected_width_pixels,detected_height_pixels):
	#Substrate segmentation inference
	substrate_result = inference_detector(substrate_model, img).pred_instances.cpu().numpy()

	if reference_model is  None:
		#Calculate substrate height/width data
		detected_width_pixels.append(substrate_result[0]["bboxes"][0][2] - substrate_result[0]["bboxes"][0][0])
		detected_height_pixels.append(substrate_result[0]["bboxes"][0][3] - substrate_result[0]["bboxes"][0][1])

	else:
		reference_result = inference_detector(reference_model, img).pred_instances.cpu().numpy()
		for idx in range(len(reference_result["labels"])):
			if reference_result["scores"][idx]>0.9:
				detected_width_pixels.append(reference_result[idx]["bboxes"][0][2] - reference_result[idx]["bboxes"][0][0])
				detected_height_pixels.append(reference_result[idx]["bboxes"][0][3] - reference_result[idx]["bboxes"][0][1])

	#Calculate the substrate height/width average
	averaged_height_pixels = np.mean(detected_height_pixels)
	averaged_width_pixels = np.mean(detected_width_pixels)		
		
	#Saving substrate bbox image 
	if save_substrate_bbox_image:
		substrate_img = img.copy()
		#Draw bounding boxes on substrate images
		for result in substrate_result:
			sub_result = result["bboxes"][0]
			cv2.rectangle(substrate_img,(int(sub_result[0]),int(sub_result[1])),(int(sub_result[2]),int(sub_result[3])),(0,0,255),5)

		#Save substrate images
		cv2.imwrite(working_folder + "/Substrate/substrate_" + test_img, substrate_img)
		
	#Saving reference bbox image 
	if save_reference_bbox_image and reference_model is not None:
		reference_img = img.copy()
		#Draw bounding boxes on substrate images
		for result in reference_result:
			ab_ref_result = result["bboxes"][0]
			cv2.rectangle(reference_img,(int(ab_ref_result[0]),int(ab_ref_result[1])),(int(ab_ref_result[2]),int(ab_ref_result[3])),(0,0,255),5)
		
		#Create the folder containing the reference images
		os.makedirs(working_folder + "/reference/",exist_ok=True)
		#Save reference images
		cv2.imwrite(working_folder + "/reference/reference_" + test_img, reference_img)
		
	return substrate_result,averaged_height_pixels,averaged_width_pixels


# ##Process substrate, calculate size and save results
# def process_substrate(substrate_model,reference_model,img,image_for_visualization,save_substrate_bbox_image,working_folder,test_img,detected_width_pixels,detected_height_pixels):
# 	#Substrate segmentation inference
# 	substrate_result = inference_detector(substrate_model, img).pred_instances

# 	if reference_model is not None:
# 		#Calculate substrate height/width data
# 		detected_width_pixels.append(substrate_result[0]["bboxes"].cpu().numpy()[0][2] - substrate_result[0]["bboxes"].cpu().numpy()[0][0])
# 		detected_height_pixels.append(substrate_result[0]["bboxes"].cpu().numpy()[0][3] - substrate_result[0]["bboxes"].cpu().numpy()[0][1])

# 		#Calculate the substrate height/width average
# 		averaged_height_pixels = np.mean(detected_height_pixels)
# 		averaged_width_pixels = np.mean(detected_width_pixels)
# 	else:
# 		reference_result = inference_detector(reference_model, img).pred_instances

# 	#Saving substrate bbox image 
# 	if save_substrate_bbox_image:
# 		substrate_img = img.copy()
# 		#Draw bounding boxes on substrate images
# 		for result in substrate_result:
# 			sub_result = result["bboxes"].cpu().numpy()[0]
# 			cv2.rectangle(substrate_img,(int(sub_result[0]),int(sub_result[1])),(int(sub_result[2]),int(sub_result[3])),(0,0,255),5)

# 		#Save substrate images
# 		cv2.imwrite(working_folder + "/Substrate/substrate_" + test_img, substrate_img) #cv2.cvtColor(substrate_img,cv2.COLOR_RGB2BGR))

# 	return substrate_result,averaged_height_pixels,averaged_width_pixels


##Processing image polygons and information
def process_results(image_result, averaged_width_pixels, averaged_height_pixels, substrate_real_width, substrate_real_height):

	#Converting how inference information is saved
	img_result = []
	for result in image_result.pred_instances:
		img_result.append([result[0]["scores"][0],result[0]["bboxes"][0],result[0]["labels"][0],result[0]["masks"][0]])

	#To store all the results from the image
	results = []
	results_info = []

	#Converting from a boolean mask to a coordinate mask
	for result in img_result:
		masks = result[3]
		if not np.array_equal(masks,[]):
			#Converting the true/false mask matrix in each result to a coordinate list 
			points = np.argwhere(masks).tolist()
			#Flipping across the coordinates from (y,x) to (x,y)
			points = np.flip(points,1)
			#Reducing the size of the coordinate list
			points = points[0::10]
			#Finding the concave hull (outline) of the mask 
			hull = concave_hull_indexes(points)
			#Appending the points that make the outline
			results.append(points[hull])

			#Caclulating cluster pixel sizing
			cluster_bbox = np.array([result[1][0],result[1][1],result[1][2],result[1][3]]).astype(int)
			pixel_cluster_width = result[1][2] - result[1][0]
			pixel_cluster_height = result[1][3] - result[1][1]

			#Getting cluster label
			cluster_label = result[2]

			#Use the last element of the averaged substrate lengths to approximate the actual cluster length and width
			cluster_width = round(pixel_cluster_width*substrate_real_width/averaged_width_pixels,3)
			cluster_height = round(pixel_cluster_height*substrate_real_height/averaged_height_pixels,3)

			#Results info include: Cluster label, bbox height/width, height/width, bbox coordinates 
			results_info.append([cluster_label,pixel_cluster_height,pixel_cluster_width,cluster_height,cluster_width,cluster_bbox])

	return results, results_info