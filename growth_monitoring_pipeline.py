import os
import sys
import time
import math
import copy
import argparse
import pandas as pd
import mmcv
from natsort import natsorted
import matplotlib.pyplot as plt
from mmdet.apis import inference_detector

# Import utilities
sys.path.insert(1, "./functions/")
sys.path.insert(1, "./configs/")
from Information_Processing_Utilities import *
from Model_Processing_Utilities import *
from Filtering_Utilities import *
from Tracking_Utilities import *
from Metrics_Utilities import *
from config_paths import *
import config_variables

def parse_args():
    parser = argparse.ArgumentParser(description="Mushroom Growth Monitoring Pipeline")
    parser.add_argument("--verbose", action="store_true", default=False, required=False, help="Enable verbose output.")
    parser.add_argument("--calculate_metrics", action="store_true", default=True, required=False, help="Calculate metrics mAP,mAR,MOTA.")
    parser.add_argument("--save_plots", action="store_true", default=True, required=False, help="Save locally growth curve plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    test_set_path = PATHS["test_set_path"]
    initialize_workspace(PATHS["working_folder"])
    establish_cluster_sizing(PATHS["working_folder"])
    use_device = check_cuda()

    # Path to COCO annotation json file, creates automatically tracking annotations
    annotations, tracking_annotations = annotation_tracking(PATHS["annotations_path"])

    mushroom_model, substrate_model, reference_model, visualizer = load_models(
        PATHS["mushroom_model_config_folder"], PATHS["substrate_model_config_folder"],
        PATHS["reference_model_config_folder"], use_device
    )

    test_set = natsorted(os.listdir(test_set_path))
    test_set = [x for x in test_set if x.endswith("JPG")]
    start_time = time.time()
    for img_num, test_img in enumerate(test_set):

        skip_image, config_variables.last_valid_brightness_ema, config_variables.last_valid_brightness_residuals = brightness_filter(test_img,
            path_to_image=test_set_path + test_img,
            last_valid_ema=config_variables.last_valid_brightness_ema,
            last_valid_residuals=config_variables.last_valid_brightness_residuals,
        )

        if skip_image:
            config_variables.post_filtering_polygons.append([[0]])
            config_variables.post_filtering_polygons_info.append([[0]])
            config_variables.pre_filtering_polygons.append([[0]])
            config_variables.pre_filtering_polygons_info.append([[0]])
            continue

        # Load the image
        img = mmcv.imread(test_set_path + test_img)
        # Color correction of the images for visualization
        image_for_visualization = mmcv.image.bgr2rgb(img)
        ##Process substrate, calculate size and save results
        substrate_result, averaged_height_pixels, averaged_width_pixels = process_substrate(substrate_model,
                                                                                            reference_model,
                                                                                            img,
                                                                                            image_for_visualization,
                                                                                            config_variables.save_substrate_bbox_image,
                                                                                            config_variables.save_reference_bbox_image,
                                                                                            PATHS["working_folder"],
                                                                                            test_img,
                                                                                            config_variables.detected_width_pixels,
                                                                                            config_variables.detected_height_pixels)

        # Mushroom segmentation inference
        image_result = inference_detector(mushroom_model, img)

        # Visualize predictions before filtering
        visualizer.add_datasample(
            "result",
            image_for_visualization,
            data_sample=image_result,
            draw_gt=None,
            wait_time=0,
            out_file=PATHS["working_folder"] + "Predictions/before_filtering_predictions_" + test_img,
            pred_score_thr=0.01
        )

        # Keep predictions information before filtering
        before_filtering_results, before_filtering_results_info = process_results(image_result.cpu().numpy(),
                                                                                  averaged_width_pixels,
                                                                                  averaged_height_pixels,
                                                                                  config_variables.substrate_real_width,
                                                                                  config_variables.substrate_real_height)

        config_variables.pre_filtering_polygons.append(before_filtering_results)
        config_variables.pre_filtering_polygons_info.append(before_filtering_results_info)

        # Apply filters
        image_result = delete_low_confidence_predictions(image_result, config_variables.confidence_score_threshold)
        # If no confident predictions are found, add placeholders for metrics and move to next frame
        if image_result.pred_instances.masks.size == 0:
            print("Finished processing image #{}: {}".format(img_num + 1, test_img))
            print("No confident predictions were found.")
            print("--------------------------------------------")
            config_variables.metrics_post_filtering_polygons.append([])
            config_variables.metrics_post_filtering_polygons_info.append([])
            continue

        image_result = delete_overlapping_with_lower_confidence(image_result,
                                                                config_variables.overlapping_iou_threshold)
        image_result = delete_post_background_clusters(img_num, image_result,
                                                       substrate_result,
                                                       config_variables.post_harvest_post_filtering_polygons_info_base,
                                                       config_variables.post_harvest_occluded_iou_overlap,
                                                       args.verbose)

        # -----------------------------------------------------------------------------------------------------------------
        # Processing of cluster data to be used from tracking algorithm data structures
        results, results_info = process_results(image_result,
                                                averaged_width_pixels,
                                                averaged_height_pixels,
                                                config_variables.substrate_real_width,
                                                config_variables.substrate_real_height)

        # Keep the hull information for all clusters
        config_variables.post_filtering_polygons.append(results)
        config_variables.post_filtering_polygons_info.append(results_info)

        # Show image with numbered clusters before tracking
        if config_variables.save_untracked:
            save_untracked_image(image_for_visualization,
                                 config_variables.post_filtering_polygons,
                                 PATHS["working_folder"],
                                 test_img)

        # Tracking of predicted clusters
        config_variables.post_filtering_polygons, config_variables.post_filtering_polygons_info, config_variables.baseline, image_result = cluster_sort(
            config_variables.post_filtering_polygons,
            config_variables.post_filtering_polygons_info,
            config_variables.baseline,
            image_result)

        config_variables.metrics_post_filtering_polygons.append(config_variables.post_filtering_polygons[-1])
        config_variables.metrics_post_filtering_polygons_info.append(config_variables.post_filtering_polygons_info[-1])

        # Check no prediction remains
        if not config_variables.post_filtering_polygons_info[-1]:
            config_variables.post_filtering_polygons.pop()
            config_variables.post_filtering_polygons_info.pop()

            if args.verbose:
                print("No valid predictions remain after filtering.")
                print("Finished processing image #{}: {}".format(img_num + 1, test_img))
                print("--------------------------------------------")

            continue

        # Visualize predictions after filtering
        visualizer.add_datasample(
            "result",
            image_for_visualization,
            data_sample=image_result,
            draw_gt=None,
            wait_time=0,
            out_file=PATHS["working_folder"] + "Predictions/after_filtering_predictions_" + test_img,
            pred_score_thr=config_variables.confidence_score_threshold
        )

        # Post tracking MOTA metric calculation
        config_variables.mota_metric, config_variables.motaTracker = get_tracking_metrics(tracking_annotations[img_num],
                                                                                          config_variables.post_filtering_polygons[
                                                                                              -1],
                                                                                          config_variables.mota_metric,
                                                                                          config_variables.motaTracker)

        # Visualize and crop tracked clusters
        cluster_sizing_and_visualization(image_for_visualization,
                                         config_variables.post_filtering_polygons,
                                         config_variables.post_filtering_polygons_info,
                                         config_variables.labels_to_crop,
                                         config_variables.cluster_sizing_option,
                                         config_variables.crop_cluster_option,
                                         PATHS["working_folder"],
                                         test_img,
                                         img_num,
                                         averaged_width_pixels,
                                         averaged_height_pixels,
                                         config_variables.substrate_real_width,
                                         config_variables.substrate_real_height)

        # Equalizing polygon list
        config_variables.post_filtering_polygons, config_variables.post_filtering_polygons_info = equalize_polygons(
            config_variables.post_filtering_polygons,
            config_variables.post_filtering_polygons_info)

        # Creating post-processing bbox baseline
        if not config_variables.post_harvest_post_filtering_polygons_info_base:
            config_variables.post_harvest_post_filtering_polygons_info_base = copy.deepcopy(
                config_variables.post_filtering_polygons_info[-1])
        else:
            for i in range(len(config_variables.post_filtering_polygons_info[-1])):
                if config_variables.post_filtering_polygons_info[-1][i] == [0]:
                    continue
                if i < len(config_variables.post_harvest_post_filtering_polygons_info_base):
                    config_variables.post_harvest_post_filtering_polygons_info_base[i] = copy.deepcopy(
                        config_variables.post_filtering_polygons_info[-1][i])
                else:
                    config_variables.post_harvest_post_filtering_polygons_info_base.append(
                        copy.deepcopy(config_variables.post_filtering_polygons_info[-1][i]))

        if args.verbose:
            print("Finished processing image #{}: {}".format(img_num + 1, test_img))
            print("--------------------------------------------")

    end_time = time.time()
    if args.verbose:
        print("Total timelapse processing time: ", end_time - start_time)
        print("Approximate processing time per image: ", (end_time - start_time) / len(test_set))

    # Calculate metrics
    calculate_metrics(annotations, config_variables.pre_filtering_polygons, config_variables.pre_filtering_polygons_info, config_variables.post_filtering_polygons,
                      config_variables.post_filtering_polygons_info,config_variables.mota_metric)

    # Update the class mapping for the legend accordingly
    label_mapping = {0: "Immature", 1: "Well-defined", 2: "Overstayed"}
    label_colors = {0: "green", 1: "red", 2: "blue"}
    # Plot growth monitoring data
    plot_area_growth_curves("./results/Cluster_Sizing.csv",label_mapping,label_colors, time_interval=2, show_plots=False)

if __name__ == "__main__":
    main()
