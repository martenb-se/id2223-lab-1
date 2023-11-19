# Service
import modal

# Environment check
import sys
import os

# Nice printing
from pprint import pprint

# Nice logging
# Note: If not running using "modal_daily-wine-feature-pipeline.sh", then copy ../nice_log.py to this script's current
#       folder. Otherwise, modal cannot copy it to the remote environment when running or deploying there.
import nice_log as NiceLog
from nice_log import BGColors

# My own exceptions (makes it nicer and quicker in the reports from Model)
from project_exceptions import (
    HopsworksNoAPIKey, HopsworksLoginError, HopsworksGetFeatureStoreError, HopsworksGetFeatureGroupError,
    HopsworksQueryReadError, HopsworksFeatureGroupInsertError)

# IDE help
from hsfs import feature_store, feature_group, feature_view
from hopsworks import project as HopsworksProject
from hsfs.constructor import query as hsfs_query
from great_expectations.core.expectation_validation_result import ExpectationSuiteValidationResult
from hsml import model_registry
from hsml.python import model as hsml_model

# Error help
from hopsworks import RestAPIError

# Settings
# - Modal
modal_stub_name = "wine_batch_inference_daily"
modal_image_libraries = ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image"]
# - Modal Deployment
model_run_every_n_days = 1  # Note: must use `modal deploy` to run scheduled app
# - Hopsworks
hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment
# - - Names
# - - - Models
model_wine_dir = "wine_models"
model_red_wine_name = "wine_red_model"
model_red_wine_version = 1
model_white_wine_name = "wine_white_model"
model_white_wine_version = 1
# - - - Feature Groups
fg_wine_name = "wine"
fg_wine_version = 1
fg_type_red = "red"
fg_type_white = "white"
# - - - Feature Views
fw_wine_red_name = "wine_red"
fw_wine_red_version = 1
fw_wine_white_name = "wine_white"
fw_wine_white_version = 1
# - - - Monitor
fg_monitor_name = "wine_predictions"
fg_monitor_version = 1
dir_wine_saves = "latest_wine"
# file_wine_predict_save = f"{dir_wine_saves}/latest_wine.png"  # Don't know of a good picture for a rating
# file_wine_actual_save = f"{dir_wine_saves}/actual_wine.png"  # Don't know of a good picture for a rating
hopsworks_images_location = f"Resources/images/{dir_wine_saves}"
file_dataframe_save = "df_recent.png"
file_confusion_matrix_save = "confusion_matrix.png"
num_monitor_entries_to_export = 4

LOCAL = True

# Running REMOTELY in Modal's environment
if "MODAL_ENVIRONMENT" in os.environ:
    NiceLog.info(f"Running in {BGColors.HEADER}REMOTE{BGColors.ENDC} Modal environment")
    LOCAL = False
# Running LOCALLY with 'modal run' to deploy to Modal's environment
elif "/modal" in os.environ["_"]:
    from dotenv import load_dotenv

    if sys.argv[1] == "run":
        NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} using 'modal run' to run stub once in Modal's "
                     f"remote environment")

    elif sys.argv[1] == "deploy":
        NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} using 'modal deploy' to deploy to Modal's "
                     f"remote environment")
    else:
        NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} using 'modal {sys.argv[1]}'")

    LOCAL = False
# Running LOCALLY in Python
else:
    from dotenv import load_dotenv

    NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} in Python environment.")





def g():
    from pathlib import Path
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    # Make sure directory exists
    Path(dir_wine_saves).mkdir(parents=True, exist_ok=True)

    NiceLog.header(f"Running function to predict the daily wine (and get the ground truth), save latest prediction, "
                   f"get a table and a confusion matrix of all predictions and upload data to Hopsworks!")
    NiceLog.info("Logging in to Hopsworks...")

    try:
        project: HopsworksProject.Project = hopsworks.login()
    except RestAPIError as e:
        NiceLog.error(f"Failed to log in to Hopsworks. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to log in to Hopsworks. Reason: {e}")
        return

    NiceLog.success("Logged in to Hopsworks!")
    NiceLog.info(f"Active hopsworks project: {BGColors.HEADER}{project.name}{BGColors.ENDC} ({project.description})")
    NiceLog.info(f"Project created at: {BGColors.HEADER}{project.created}{BGColors.ENDC}")

    NiceLog.info(f"Getting {BGColors.HEADER}{project.name}{BGColors.ENDC} feature store...")
    try:
        fs: feature_store.FeatureStore = project.get_feature_store()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature store from Hopsworks project. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature store from Hopsworks project. Reason: {e}")
        return

    NiceLog.success("Gotten feature store!")
    NiceLog.info(f"Feature store is named: {BGColors.HEADER}{fs.name}{BGColors.ENDC})")

    NiceLog.info(f"Getting {BGColors.HEADER}{project.name}{BGColors.ENDC} model registry...")
    try:
        mr: model_registry.ModelRegistry = project.get_model_registry()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get model registry from Hopsworks project. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get model registry from Hopsworks project. Reason: {e}")
        return

    #
    # ----------------------- LATEST ADDITION
    #

    NiceLog.info(f"Getting latest wine addition!")
    NiceLog.info(f"Getting {BGColors.HEADER}{fg_wine_name}{BGColors.ENDC} (version {fg_wine_version}) feature group...")
    try:
        wine_fg: feature_group.FeatureGroup = fs.get_feature_group(name=fg_wine_name, version=fg_wine_version)
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
        return
    NiceLog.info(f"Feature group is named: {BGColors.HEADER}{wine_fg.name}{BGColors.ENDC} ({wine_fg.description})")

    NiceLog.info(f"Getting feature group as a DataFrame...")
    try:
        wine_df: DataFrame = wine_fg.read()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project as DataFrame. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project as DataFrame. "
                      f"Reason: {e}")
        return
    NiceLog.success(f"Gotten feature group as a DataFrame.")
    pprint(wine_df.describe())

    # wine_batch_data: pd.DataFrame = wine_red_fw.get_batch_data()
    # NiceLog.ok(f"Gotten feature view as a DataFrame.")
    # pprint(wine_batch_data.describe())

    offset = 1
    latest_type = wine_df.iloc[-offset]["type"]
    latest_quality = wine_df.iloc[-offset]["quality"]

    NiceLog.info(f"Latest addition was a {BGColors.HEADER}{latest_type}{BGColors.ENDC} wine. Using appropriate model.")

    wine_quality = 0
    if latest_type == fg_type_red:
        #
        # ----------------------- RED WINE - MODEL
        #

        NiceLog.info(f"Getting model named {BGColors.HEADER}{model_red_wine_name}{BGColors.ENDC} (version {model_red_wine_version})...")
        try:
            model_red_wine: hsml_model.Model = mr.get_model(model_red_wine_name, version=model_red_wine_version)
        except RestAPIError as e:
            NiceLog.error(f"Failed to get model from Hopsworks model registry. Reason: {e}")
            return
        except Exception as e:
            NiceLog.error(f"Unexpected error when trying to get model from Hopsworks model registry. Reason: {e}")
            return
        NiceLog.info(
            f"Model description: {BGColors.HEADER}{model_red_wine.description}{BGColors.ENDC} "
            f"(training accuracy: {model_red_wine.training_metrics['accuracy']})")

        NiceLog.info(f"Downloading model...")
        model_dir = model_red_wine.download()
        NiceLog.success(f"Model downloaded to: {BGColors.HEADER}{model_dir}{BGColors.ENDC}")

        red_wine_local_model = joblib.load(model_dir + f"/{model_red_wine_name}.pkl")
        NiceLog.ok(f"Initialized locally downloaded model "
                   f"({BGColors.HEADER}{model_dir + '/iris_model.pkl'}{BGColors.ENDC})")

        #
        # ----------------------- RED WINE - FW
        #

        NiceLog.info(f"Getting {BGColors.HEADER}{fw_wine_red_name}{BGColors.ENDC} (version {fw_wine_red_version}) feature view...")
        try:
            wine_red_fw: feature_view.FeatureView = fs.get_feature_view(name=fw_wine_red_name, version=fw_wine_red_version)
        except RestAPIError as e:
            NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
            return
        except Exception as e:
            NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
            return
        NiceLog.info(f"Feature group is named: {BGColors.HEADER}{wine_red_fw.name}{BGColors.ENDC} ({wine_red_fw.description})")

        red_wine_batch_data: pd.DataFrame = wine_red_fw.get_batch_data()
        NiceLog.ok(f"Gotten feature view as a DataFrame.")
        pprint(red_wine_batch_data.describe())

        #
        # ----------------------- RED WINE - PREDICT
        #

        NiceLog.info(f"Predicting wine qualities stored in feature view using model {BGColors.HEADER}{model_red_wine_name}{BGColors.ENDC} "
                     f"(version {model_red_wine_version})...")
        red_white_y_pred = red_wine_local_model.predict(red_wine_batch_data)
        NiceLog.ok("Done")

        wine_quality = red_white_y_pred[red_white_y_pred.size - offset]
        NiceLog.info(f"Latest wine's quality prediction is: {BGColors.HEADER}{wine_quality}{BGColors.ENDC}")

        #
        # ----------------------- RED WINE - ACTUAL
        #

        # wine_red_label = wine_df[wine_df["type"] == fg_type_red].iloc[-offset]["type"]
        NiceLog.info(f"Latest wine's quality was actually: {BGColors.HEADER}{latest_quality}{BGColors.ENDC}")

    elif latest_type == fg_type_white:
        #
        # ----------------------- WHITE WINE - MODEL
        #

        NiceLog.info(f"Getting model named {BGColors.HEADER}{model_white_wine_name}{BGColors.ENDC} (version {model_white_wine_version})...")
        try:
            model_white_wine: hsml_model.Model = mr.get_model(model_white_wine_name, version=model_white_wine_version)
        except RestAPIError as e:
            NiceLog.error(f"Failed to get model from Hopsworks model registry. Reason: {e}")
            return
        except Exception as e:
            NiceLog.error(f"Unexpected error when trying to get model from Hopsworks model registry. Reason: {e}")
            return
        NiceLog.info(
            f"Model description: {BGColors.HEADER}{model_white_wine.description}{BGColors.ENDC} "
            f"(training accuracy: {model_white_wine.training_metrics['accuracy']})")

        NiceLog.info(f"Downloading model...")
        model_dir = model_white_wine.download()
        NiceLog.success(f"Model downloaded to: {BGColors.HEADER}{model_dir}{BGColors.ENDC}")

        white_wine_local_model = joblib.load(model_dir + f"/{model_white_wine_name}.pkl")
        NiceLog.ok(f"Initialized locally downloaded model "
                   f"({BGColors.HEADER}{model_dir + '/iris_model.pkl'}{BGColors.ENDC})")

        #
        # ----------------------- WHITE WINE - FW
        #

        NiceLog.info(f"Getting {BGColors.HEADER}{fw_wine_white_name}{BGColors.ENDC} (version {fw_wine_white_version}) feature view...")
        try:
            wine_white_fw: feature_view.FeatureView = fs.get_feature_view(name=fw_wine_white_name, version=fw_wine_white_version)
        except RestAPIError as e:
            NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
            return
        except Exception as e:
            NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
            return
        NiceLog.info(f"Feature group is named: {BGColors.HEADER}{wine_white_fw.name}{BGColors.ENDC} ({wine_white_fw.description})")

        white_wine_batch_data: pd.DataFrame = wine_white_fw.get_batch_data()
        NiceLog.ok(f"Gotten feature view as a DataFrame.")
        pprint(white_wine_batch_data.describe())

        #
        # ----------------------- WHITE WINE - PREDICT
        #

        NiceLog.info(f"Predicting wine qualities stored in feature view using model {BGColors.HEADER}{model_white_wine_name}{BGColors.ENDC} "
                     f"(version {model_white_wine_version})...")
        white_white_y_pred = white_wine_local_model.predict(white_wine_batch_data)
        NiceLog.ok("Done")

        wine_quality = white_white_y_pred[white_white_y_pred.size - offset]
        NiceLog.info(f"Latest wine's quality prediction is: {BGColors.HEADER}{wine_quality}{BGColors.ENDC}")

        #
        # ----------------------- WHITE WINE - ACTUAL
        #

        # wine_white_label = wine_df[wine_df["type"] == fg_type_white].iloc[-offset]["type"]
        NiceLog.info(f"Latest wine's quality was actually: {BGColors.HEADER}{latest_quality}{BGColors.ENDC}")

    else:
        NiceLog.warn("No model exist for wine type.")
        return

    #
    # ----------------------- MONITOR - INSERT
    #
    time_now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    monitor_fg: feature_group.FeatureGroup = fs.get_or_create_feature_group(
        name=fg_monitor_name,
        version=fg_monitor_version,
        primary_key=["datetime"],
        description="Wine Quality Prediction/Outcome Monitoring"
    )
    NiceLog.ok(
        f"Got or created feature group: {BGColors.HEADER}{monitor_fg.name}{BGColors.ENDC} ({monitor_fg.description})")

    monitor_data = {
        'type': [latest_type],
        'prediction': [wine_quality],
        'label': [latest_quality],
        'datetime': [time_now],
    }
    monitor_df = pd.DataFrame(monitor_data)
    NiceLog.ok(f"Created a DataFrame from latest prediction and ground truth.")
    pprint(monitor_df)

    NiceLog.info(f"Inserting the wine quality prediction and ground truth to the "
                 f"{BGColors.HEADER}{fg_monitor_name}{BGColors.ENDC} feature group "
                 f"{BGColors.HEADER}asynchronously{BGColors.ENDC}...")
    fg_insert_info = monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})
    fg_insert_validation_info: ExpectationSuiteValidationResult = fg_insert_info[1]

    #
    # ----------------------- MONITOR - READ + ADD
    #

    NiceLog.info(f"Getting monitoring history feature group as a DataFrame...")
    try:
        history_df: DataFrame = monitor_fg.read()
    except RestAPIError as e:
        NiceLog.error(
            f"Failed to get monitoring history feature group from Hopsworks project as DataFrame. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(
            f"Unexpected error when trying to get monitoring history feature group from Hopsworks project as DataFrame. "
            f"Reason: {e}")
        return
    NiceLog.success(f"Gotten monitoring history feature group as a DataFrame.")
    pprint(history_df.describe())

    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])
    NiceLog.ok(f"Added prediction to the monitoring history feature group:")
    pprint(history_df)

    #
    # ----------------------- MONITOR - MOST RECENT FETCH and UPLOAD
    #

    df_recent = history_df.tail(4)
    NiceLog.info(f"{num_monitor_entries_to_export} most recent entries in monitoring history DataFrame:")
    pprint(df_recent)

    NiceLog.info(f"Exporting {num_monitor_entries_to_export} most recent entries to: {file_dataframe_save}")
    dfi.export(df_recent, file_dataframe_save, table_conversion = 'matplotlib')

    NiceLog.info(f"Uploading image of {num_monitor_entries_to_export} most recent monitor entries to: {hopsworks_images_location}")
    dataset_api = project.get_dataset_api()
    try:
        dataset_api.upload(file_dataframe_save, hopsworks_images_location, overwrite=True)
    except RestAPIError as e:
        NiceLog.error(f"Failed to upload image of most recent monitor entries to Hopsworks. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error and failed to upload image of most recent monitor entries to Hopsworks. Reason: {e}")
        return
    NiceLog.success(f"Image of most recent monitor entries uploaded to Hopsworks")

    #
    # ----------------------- MONITOR - MOST RECENT INFO
    #

    monitor_predictions = history_df[['prediction']]
    monitor_labels = history_df[['label']]
    NiceLog.info(f"{num_monitor_entries_to_export} most recent predictions:")
    pprint(monitor_predictions)
    NiceLog.info(f"{num_monitor_entries_to_export} most recent labels:")
    pprint(monitor_labels)

    #
    # ----------------------- CONFUSION MATRIX
    #

    # Skip for now...

# Initialize
if not LOCAL:
    NiceLog.info(f"Setting Modal stub name to: {BGColors.HEADER}{modal_stub_name}{BGColors.ENDC}")
    stub = modal.Stub(modal_stub_name)

    NiceLog.info(f"Creating a Modal image with Python libraries: {BGColors.HEADER}{', '.join(modal_image_libraries)}"
                 f"{BGColors.ENDC}")
    image = modal.Image.debian_slim().pip_install(modal_image_libraries)

    NiceLog.info(f"Stub should run every: {BGColors.HEADER}{model_run_every_n_days}{BGColors.ENDC} day(s)")

    if sys.argv[1] == "run":
        NiceLog.info(f"But this is just a {BGColors.HEADER}one time{BGColors.ENDC} test.")


    @stub.function(image=image, schedule=modal.Period(days=model_run_every_n_days),
                   secret=modal.Secret.from_name(hopsworks_api_key_modal_secret_name))
    def f():
        g()

# Load local environment
else:
    NiceLog.info("Loading local environment...")
    if load_dotenv() and 'HOPSWORKS_API_KEY' in os.environ:
        NiceLog.success("Loaded variables from .env file!")
    else:
        if 'HOPSWORKS_API_KEY' not in os.environ:
            NiceLog.error("Add add HOPSWORKS_API_KEY to your .env file!")
        else:
            NiceLog.error("Failed to load .env file!")
        exit(1)

if __name__ == "__main__":
    if LOCAL:
        g()
    else:
        stub.deploy(modal_stub_name)
        with stub.run():
            f()

