##Initialization of variables used in growth_monitoring_pipeline.py

#Tracking clusters and cluster information
post_filtering_polygons = []
post_filtering_polygons_info = []
metrics_post_filtering_polygons = []
metrics_post_filtering_polygons_info = []
pre_filtering_polygons = []
pre_filtering_polygons_info = []
post_harvest_post_filtering_polygons_info_base = []

#Baseline for tracking
baseline = []

#Tracking brightness of images for filtering
last_valid_brightness_ema = None
last_valid_brightness_residuals = []

#Saving pixel height of the substrate in images
detected_width_pixels = []
detected_height_pixels = []

#Tracking MOTA Metrics
#False Positive, False Negative, ID Switch, Ground Truth
mota_metric = [[],[],[],[]]
motaTracker = []

#From the farm substrate (50 cm x 36cm x 19cm)
# substrate_real_width = 50
# substrate_real_height = 36
substrate_real_width = 33
substrate_real_height = 28.5

#Select which class(es) of clusters to be cropped as separate images
#Set to -1 to crop all classes, else class numbers based on COCO annotations are from 0,1,2 etc.
labels_to_crop = [1]

#Generate images with cluster heights/widths measured in pixels
cluster_sizing_option = True
#Generate images with cropped clusters
crop_cluster_option = False
#Save images with substrate bbox
save_substrate_bbox_image = False
#Save images with reference bbox(es)
save_reference_bbox_image = False
#Save images with original prediction ids before tracking
save_untracked = False

#Confidence thresholds
confidence_score_threshold = 0.2
overlapping_iou_threshold = 0.3
post_harvest_occluded_iou_overlap = 0.7

#Print filtering actions on detected clusters
# verbose = True