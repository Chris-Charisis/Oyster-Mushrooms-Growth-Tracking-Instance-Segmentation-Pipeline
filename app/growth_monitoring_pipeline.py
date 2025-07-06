# Import libraries
import os
import sys
import copy
import io
import zipfile
import pathlib
import mmcv
from mmdet.apis import inference_detector
from fastapi import FastAPI, Form, HTTPException,status
import asyncio
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware
import requests
from PIL import Image
from io import BytesIO
from fastapi.responses import StreamingResponse
from datetime import datetime
from dotenv import dotenv_values


# Import custom utilities
sys.path.insert(1, "./functions/")
sys.path.insert(1, "./configs/")
from Information_Processing_Utilities import *
from Model_Processing_Utilities import *
from Filtering_Utilities import *
from Tracking_Utilities import *
from Metrics_Utilities import *
from config_paths import *
from config_variables import *


# Setup FastAPI state variables
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up the FastAPI application...")
    app.state.config_variables = copy.deepcopy(CONFIG_VARIABLES)  
    app.state.PATHS = copy.deepcopy(PATHS)
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    app.state.PATHS["working_folder"] += date_time + "/"
    initialize_workspace(app.state.PATHS["working_folder"])
    establish_cluster_sizing(app.state.PATHS["working_folder"])
    use_device = check_cuda()
    app.state.mushroom_model, app.state.substrate_model, app.state.reference_model, app.state.visualizer = load_models(
        app.state.PATHS["mushroom_model_config_folder"], app.state.PATHS["substrate_model_config_folder"],
        app.state.PATHS["reference_model_config_folder"], use_device)
    app.state.time_interval = 0
    app.state.reset_lock = asyncio.Lock()
    yield # app is live
    
# Create FastAPI app instance with lifespan management
app = FastAPI(lifespan=lifespan)

# Enable CORS (Cross-Origin Resource Sharing) to allow requests from any domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define hard reset function
async def hard_reset(app: FastAPI):
    async with app.state.reset_lock:
        # 1. Reset the working folder
        app.state.PATHS = copy.deepcopy(PATHS)
        now = datetime.now() # current date and time
        print("Hard reset initiated at:", now.strftime("%Y_%m_%d_%H_%M_%S"))
        # Generate a unique folder name based on the current date and time
        # This ensures that each reset creates a new working folder
        # and does not overwrite previous results.
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        app.state.PATHS["working_folder"] += date_time + "/"

        # 2. Initialize workspace
        initialize_workspace(app.state.PATHS["working_folder"])
        establish_cluster_sizing(app.state.PATHS["working_folder"])

        # 3. Reset in-memory state
        app.state.config_variables = copy.deepcopy(CONFIG_VARIABLES)
        app.state.time_interval = 0

        return

# Define the reset endpoint
@app.post("/reset/")
async def reset_workspace(token: str = Form(...)):
    # Load the environment variables
    config = dotenv_values(".env")
    # Simple token validation
    if token != config["RESET_TOKEN"]:
        return {
            "status_code": status.HTTP_401_UNAUTHORIZED,
            "message": "Unauthorized access. Invalid token."
        }

    # Check if the reset lock is already acquired
    # This prevents multiple reset requests from being processed simultaneously
    if app.state.reset_lock.locked():
        # Optional ‘Retry-After’ header to help the caller automate retries
        return {
            "status_code" : status.HTTP_409_CONFLICT,
            "message" : "Workspace is busy (reset in progress or frames queuing). Retry-After: 15 seconds.",
        }

    # If the lock is not acquired, proceed with the hard reset
    await hard_reset(app)

    return {
        "status_code" : status.HTTP_200_OK, 
        "message": "Workspace reset completed."
    }

# Define the image processing endpoint
@app.post("/process_image/")
async def process_image(url: str = Form(...)):
    ## Check if the URL is valid
    try:
        # make an HTTP GET request to the image URL
        response = requests.get(url)

        # check if the request was successful
        response.raise_for_status()

        # extract and save the image from the message
        image = Image.open(BytesIO(response.content))
        image = image.rotate(180)
        
        path_to_image = app.state.PATHS["working_folder"] + "/Images/downloaded_image_" + str(app.state.time_interval) + ".png"
        image.save(path_to_image)

        # read the image in an acceptable format from MMDetection library
        img = mmcv.imread(path_to_image) 

    except requests.exceptions.RequestException as e:
        return {
            "status_code": status.HTTP_400_BAD_REQUEST,
            "message": f"Failed to download image from URL: {str(e)}"}
    
    try:
        # Check if the image is a brightness outlier
        skip_image, app.state.config_variables["last_valid_brightness_ema"], app.state.config_variables["last_valid_brightness_residuals"] = brightness_filter(path_to_image=path_to_image,
            last_valid_ema=app.state.config_variables["last_valid_brightness_ema"],
            last_valid_residuals=app.state.config_variables["last_valid_brightness_residuals"],
        )

        if skip_image:
            app.state.config_variables["post_filtering_polygons"].append([[0]])
            app.state.config_variables["post_filtering_polygons_info"].append([[0]])
            app.state.time_interval += 1
            return {
                "status_code": status.HTTP_204_NO_CONTENT,
                "message": "Image skipped due to brightness filtering."
            }
        

        # Color correction of the images for visualization
        image_for_visualization = mmcv.image.bgr2rgb(img)
        ##Process substrate, calculate size and save results
        substrate_result, averaged_height_pixels, averaged_width_pixels = process_substrate(app.state.substrate_model,
                                                                                            app.state.reference_model,
                                                                                            img,
                                                                                            app.state.config_variables["save_substrate_bbox_image"],
                                                                                            app.state.config_variables["save_reference_bbox_image"],
                                                                                            app.state.PATHS["working_folder"],
                                                                                            "downloaded_image_" + str(app.state.time_interval) + ".png",
                                                                                            app.state.config_variables["detected_width_pixels"],
                                                                                            app.state.config_variables["detected_height_pixels"])

        # Mushroom segmentation inference
        image_result = inference_detector(app.state.mushroom_model, img)

        # Apply filters
        image_result = delete_low_confidence_predictions(image_result, app.state.config_variables["confidence_score_threshold"])
        # If no confident predictions are found, add placeholders for metrics and move to next frame
        if image_result.pred_instances.masks.size == 0:
            print("No confident predictions were found.")
            print("--------------------------------------------")
            app.state.time_interval += 1
            return {
                "status_code": status.HTTP_204_NO_CONTENT,
                "message": "No confident predictions were found after filtering."
            }
        image_result = delete_overlapping_with_lower_confidence(image_result,
                                                                app.state.config_variables["overlapping_iou_threshold"])
        image_result = delete_post_background_clusters(image_result,
                                                        substrate_result,
                                                        app.state.config_variables["post_harvest_post_filtering_polygons_info_base"],
                                                        app.state.config_variables["post_harvest_occluded_iou_overlap"],
                                                        )

        # -----------------------------------------------------------------------------------------------------------------
        # Processing of cluster data to be used from tracking algorithm data structures
        results, results_info = process_results(image_result,
                                                averaged_width_pixels,
                                                averaged_height_pixels,
                                                app.state.config_variables["substrate_real_width"],
                                                app.state.config_variables["substrate_real_height"])

        # Keep the hull information for all clusters
        app.state.config_variables["post_filtering_polygons"].append(results)
        app.state.config_variables["post_filtering_polygons_info"].append(results_info)

        # Tracking of predicted clusters
        app.state.config_variables["post_filtering_polygons"], app.state.config_variables["post_filtering_polygons_info"], app.state.config_variables["baseline"], image_result = cluster_sort(
            app.state.config_variables["post_filtering_polygons"],
            app.state.config_variables["post_filtering_polygons_info"],
            app.state.config_variables["baseline"],
            image_result)

        # config_variables["metrics_post_filtering_polygons"].append(config_variables["post_filtering_polygons"][-1])
        # config_variables["metrics_post_filtering_polygons_info"].append(config_variables["post_filtering_polygons_info"][-1])
        # print(app.state.config_variables["post_filtering_polygons_info"])

        # Check no prediction remains
        if not app.state.config_variables["post_filtering_polygons_info"][-1]:
            app.state.config_variables["post_filtering_polygons"].pop()
            app.state.config_variables["post_filtering_polygons_info"].pop()
            print("No valid predictions remain after filtering.")
            print("--------------------------------------------")
            app.state.time_interval += 1
            return {
                "status_code": status.HTTP_204_NO_CONTENT,
                "message": "No valid predictions remain after full filtering.",
                "prediction": None,
                "curves": None
            }

        # Visualize predictions after filtering
        app.state.visualizer.add_datasample(
            "result",
            image_for_visualization,
            data_sample=image_result,
            draw_gt=None,
            wait_time=0,
            out_file=app.state.PATHS["working_folder"] + "Predictions/after_filtering_predictions_downloaded_image_" + str(app.state.time_interval) + ".png",
            pred_score_thr=app.state.config_variables["confidence_score_threshold"]
        )

        # Post tracking MOTA metric calculation
        # config_variables["mota_metric"], config_variables["motaTracker"] = get_tracking_metrics(tracking_annotations[img_num],
        #                                                                                   config_variables["post_filtering_polygons"][
        #                                                                                       -1],
        #                                                                                   config_variables["mota_metric"],
        #                                                                                   config_variables["motaTracker"])
        # Visualize and crop tracked clusters
        cluster_sizing_and_visualization(image_for_visualization,
                                            app.state.config_variables["post_filtering_polygons"],
                                            app.state.config_variables["post_filtering_polygons_info"],
                                            app.state.config_variables["labels_to_crop"],
                                            app.state.config_variables["cluster_sizing_option"],
                                            app.state.config_variables["crop_cluster_option"],
                                            app.state.PATHS["working_folder"],
                                            "after_filtering_predictions_downloaded_image_" + str(app.state.time_interval) + ".png",
                                            app.state.time_interval,
                                            averaged_width_pixels,
                                            averaged_height_pixels,
                                            app.state.config_variables["substrate_real_width"],
                                            app.state.config_variables["substrate_real_height"])

        
        # Equalizing polygon list
        app.state.config_variables["post_filtering_polygons"], app.state.config_variables["post_filtering_polygons_info"] = equalize_polygons(
            app.state.config_variables["post_filtering_polygons"],
            app.state.config_variables["post_filtering_polygons_info"])

        
        # Creating post-processing bbox baseline
        if not app.state.config_variables["post_harvest_post_filtering_polygons_info_base"]:
            app.state.config_variables["post_harvest_post_filtering_polygons_info_base"] = copy.deepcopy(
                app.state.config_variables["post_filtering_polygons_info"][-1])
        else:
            for i in range(len(app.state.config_variables["post_filtering_polygons_info"][-1])):
                if app.state.config_variables["post_filtering_polygons_info"][-1][i] == [0]:
                    continue
                if i < len(app.state.config_variables["post_harvest_post_filtering_polygons_info_base"]):
                    app.state.config_variables["post_harvest_post_filtering_polygons_info_base"][i] = copy.deepcopy(
                        app.state.config_variables["post_filtering_polygons_info"][-1][i])
                else:
                    app.state.config_variables["post_harvest_post_filtering_polygons_info_base"].append(
                        copy.deepcopy(app.state.config_variables["post_filtering_polygons_info"][-1][i]))


        # Update the class mapping for the legend accordingly
        label_mapping = {0: "Immature", 1: "Well-defined", 2: "Overstayed"}
        label_colors = {0: "green", 1: "red", 2: "blue"}
        # Plot growth monitoring data
        plot_area_growth_curves(app.state.PATHS["working_folder"],label_mapping,label_colors, time_interval=2, show_plots=False)


        # Return two images to the post request with status code 200
        pred_path   = pathlib.Path(app.state.PATHS["working_folder"] + "Predictions/after_filtering_predictions_downloaded_image_" + str(app.state.time_interval) + ".png")
        tracked_path   = pathlib.Path(app.state.PATHS["working_folder"] + "Tracked/tracked_after_filtering_predictions_downloaded_image_" + str(app.state.time_interval) + ".png")
        curves_path = pathlib.Path(app.state.PATHS["working_folder"] + "clusters_relative_area.png")
        csv_path = pathlib.Path(app.state.PATHS["working_folder"] + "Cluster_Sizing.csv")

        # Increment the time interval for the next request
        app.state.time_interval += 1

        original_image_name = url.split("/")[-1]
        original_image_name = original_image_name.split(".")[0]  # Remove the file extension
        
        # build an in-memory zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.write(pred_path,   arcname="prediction.png")
            zf.write(curves_path, arcname="curves.png")
            zf.write(tracked_path, arcname="tracked.png")
            zf.write(csv_path, arcname="cluster_growth.csv")
        buf.seek(0)

        # Set the headers for the response to indicate a file download
        headers = {"Content-Disposition": 'attachment; filename="' + original_image_name + '.zip"'}
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers=headers,
            status_code=status.HTTP_200_OK)
    
    except Exception as e:
        return {
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": f"An error occurred during image processing: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)