import modal

# Directory creation
from pathlib import Path

# Environment check
import os

# Nice printing
from pprint import pprint

# Nice logging
import nice_log as NiceLog
from nice_log import BGColors

# IDE help
from hsfs import feature_store, feature_group, feature_view as hsfs_feature_view
from hsml import model_registry
from hsml.python import model as hsml_model
from hopsworks import project as HopsworksProject
from great_expectations.core.expectation_validation_result import ExpectationSuiteValidationResult
from pandas.core.frame import DataFrame

# Error help
from hopsworks import RestAPIError

LOCAL = True

# Running REMOTELY in Modal's environment
if "MODAL_ENVIRONMENT" in os.environ:
    NiceLog.info(f"Running in {BGColors.HEADER}REMOTE{BGColors.ENDC} Modal environment")
    LOCAL = False
# Running LOCALLY with 'modal run' to deploy to Modal's environment
elif "/modal" in os.environ["_"]:
    from dotenv import load_dotenv

    NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} using 'modal run' to deploy to Modal's remote "
                 f"environment")
    LOCAL = False
# Running LOCALLY in Python
else:
    from dotenv import load_dotenv

    NiceLog.info(f"Running {BGColors.HEADER}LOCALLY{BGColors.ENDC} in Python environment.")


def g():
    model_name = "iris_model"
    model_version = 1
    fw_name = "iris"
    fw_version = 1
    fg_name = "iris"
    fg_version = 1
    fg_monitor_name = "iris_predictions"
    fg_monitor_version = 1
    dir_flower_saves = "./latest_flower"
    file_flower_predict_save = f"{dir_flower_saves}/latest_iris.png"
    file_flower_actual_save = f"{dir_flower_saves}/actual_iris.png"
    file_dataframe_save = f"{dir_flower_saves}/df_recent.png"
    file_confusion_matrix_save = f"{dir_flower_saves}/confusion_matrix.png"
    hopsworks_images_location = "Resources/images"
    num_monitor_entries_to_export = 4

    # Make sure directory exists
    Path(dir_flower_saves).mkdir(parents=True, exist_ok=True)

    def flower_url(cur_flower):
        return ("https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" +
                cur_flower + ".png")

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

    NiceLog.header(f"Running function to predict the daily flower (and get the ground truth), save latest prediction, "
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

    NiceLog.info(f"Getting model named {BGColors.HEADER}{model_name}{BGColors.ENDC} (version {model_version})...")
    try:
        model: hsml_model.Model = mr.get_model(model_name, version=model_version)
    except RestAPIError as e:
        NiceLog.error(f"Failed to get model from Hopsworks model registry. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get model from Hopsworks model registry. Reason: {e}")
        return
    NiceLog.info(
        f"Model description: {BGColors.HEADER}{model.description}{BGColors.ENDC} "
        f"(training accuracy: {model.training_metrics['accuracy']})")

    NiceLog.info(f"Downloading model...")
    model_dir = model.download()
    NiceLog.success(f"Model downloaded to: {BGColors.HEADER}{model_dir}{BGColors.ENDC}")

    local_model = joblib.load(model_dir + "/iris_model.pkl")
    NiceLog.ok(f"Initialized locally downloaded model "
               f"({BGColors.HEADER}{model_dir + '/iris_model.pkl'}{BGColors.ENDC})")

    feature_view: hsfs_feature_view.FeatureView = fs.get_feature_view(name="iris", version=1)

    NiceLog.info(f"Getting {BGColors.HEADER}{fw_name}{BGColors.ENDC} (version {fw_version}) feature view...")
    try:
        feature_view: hsfs_feature_view.FeatureView = fs.get_feature_view(name=fw_name, version=fw_version)
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
        return
    NiceLog.info(f"Feature group is named: {BGColors.HEADER}{feature_view.name}{BGColors.ENDC} ({feature_view.description})")

    batch_data: DataFrame = feature_view.get_batch_data()
    NiceLog.ok(f"Gotten feature view as a DataFrame.")
    pprint(batch_data.describe())

    #
    # PREDICT
    #

    NiceLog.info(f"Predicting flowers stored in feature view using model {BGColors.HEADER}{model_name}{BGColors.ENDC} "
                 f"(version {model_version})...")
    y_pred = local_model.predict(batch_data)
    NiceLog.ok("Done")

    #print(y_pred)

    offset = 1
    flower = y_pred[y_pred.size - offset]
    NiceLog.info(f"Latest flower's prediction is: {BGColors.HEADER}{flower}{BGColors.ENDC}")

    NiceLog.info(f"Saving an image of latest predicted flower to: {file_flower_predict_save}")
    img = Image.open(requests.get(flower_url(flower), stream=True).raw)
    img.save(file_flower_predict_save)
    NiceLog.ok("Done")

    NiceLog.info(f"Uploading image of latest predicted flower to Hopsworks at: {hopsworks_images_location}")
    dataset_api = project.get_dataset_api()
    try:
        dataset_api.upload(file_flower_predict_save, hopsworks_images_location, overwrite=True)
    except RestAPIError as e:
        NiceLog.error(f"Failed to upload latest predicted flower to Hopsworks. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error and failed to upload latest predicted flower to Hopsworks. Reason: {e}")
        return
    NiceLog.success(f"Latest flower prediction uploaded to Hopsworks")

    #
    # ACTUAL
    #

    NiceLog.info(f"Getting latest predicted flower ground truth!")
    NiceLog.info(f"Getting {BGColors.HEADER}{fg_name}{BGColors.ENDC} (version {fg_version}) feature group...")
    try:
        iris_fg: feature_group.FeatureGroup = fs.get_feature_group(name=fg_name, version=fg_version)
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
        return
    NiceLog.info(f"Feature group is named: {BGColors.HEADER}{iris_fg.name}{BGColors.ENDC} ({iris_fg.description})")

    NiceLog.info(f"Getting feature group as a DataFrame...")
    try:
        iris_fg_df: DataFrame = iris_fg.read()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project as DataFrame. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project as DataFrame. "
                      f"Reason: {e}")
        return
    NiceLog.success(f"Gotten feature group as a DataFrame.")
    pprint(iris_fg_df.describe())

    #print(df)

    label = iris_fg_df.iloc[-offset]["variety"]
    NiceLog.info(f"Latest flower's was actually: {BGColors.HEADER}{label}{BGColors.ENDC}")

    NiceLog.info(f"Saving an image of latest predicted flower to: {file_flower_actual_save}")
    img = Image.open(requests.get(flower_url(label), stream=True).raw)
    img.save(file_flower_actual_save)

    NiceLog.info(f"Uploading image of latest flower ground truth to Hopsworks at: {hopsworks_images_location}")
    dataset_api = project.get_dataset_api()
    try:
        dataset_api.upload(file_flower_actual_save, hopsworks_images_location, overwrite=True)
    except RestAPIError as e:
        NiceLog.error(f"Failed to upload latest flower ground truth to Hopsworks. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error and failed to upload latest flower ground truth to Hopsworks. Reason: {e}")
        return
    NiceLog.success(f"Latest flower ground truth uploaded to Hopsworks")

    #
    # MONITOR: INSERT
    #
    time_now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    monitor_fg: feature_group.FeatureGroup = fs.get_or_create_feature_group(
        name=fg_monitor_name,
        version=fg_monitor_version,
        primary_key=["datetime"],
        description="Iris flower Prediction/Outcome Monitoring"
    )
    NiceLog.ok(f"Got or created feature group: {BGColors.HEADER}{iris_fg.name}{BGColors.ENDC} ({iris_fg.description})")

    monitor_data = {
        'prediction': [flower],
        'label': [label],
        'datetime': [time_now],
    }
    monitor_df = pd.DataFrame(monitor_data)
    NiceLog.ok(f"Created a DataFrame from latest prediction and ground truth.")
    pprint(monitor_df)

    NiceLog.info(f"Inserting the flower prediction and ground truth to the "
                 f"{BGColors.HEADER}{fg_monitor_name}{BGColors.ENDC} feature group "
                 f"{BGColors.HEADER}asynchronously{BGColors.ENDC}...")
    fg_insert_info = monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})
    fg_insert_validation_info: ExpectationSuiteValidationResult = fg_insert_info[1]

    # Only if `"wait_for_job": True` above
    # if fg_insert_validation_info.success:
    #     NiceLog.success("Flower prediction and ground truth inserted into the feature group.")
    # else:
    #     NiceLog.error("Could not insert flower prediction and ground truth into group! More info")
    #     pprint(fg_insert_validation_info)
    #     return

    #
    # MONITOR: READ + ADD
    #

    NiceLog.info(f"Getting monitoring history feature group as a DataFrame...")
    try:
        history_df: DataFrame = monitor_fg.read()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get monitoring history feature group from Hopsworks project as DataFrame. Reason: {e}")
        return
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get monitoring history feature group from Hopsworks project as DataFrame. "
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
    # MONITOR: MOST RECENT FETCH and UPLOAD
    #

    df_recent = history_df.tail(4)
    NiceLog.info(f"{num_monitor_entries_to_export} most recent entries in monitoring history DataFrame:")
    pprint(df_recent)

    NiceLog.info(f"Exporting {num_monitor_entries_to_export} most recent entries to: {file_dataframe_save}")
    dfi.export(df_recent, file_dataframe_save, table_conversion = 'matplotlib')


    NiceLog.info(f"Uploading image of {num_monitor_entries_to_export} most recent monitor entries to: {hopsworks_images_location}")
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
    # MONITOR: MOST RECENT INFO
    #

    monitor_predictions = history_df[['prediction']]
    monitor_labels = history_df[['label']]
    NiceLog.info(f"{num_monitor_entries_to_export} most recent predictions:")
    pprint(monitor_predictions)
    NiceLog.info(f"{num_monitor_entries_to_export} most recent labels:")
    pprint(monitor_labels)

    #
    # CONFUSION MATRIX
    #

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    NiceLog.info(f"Number of different flower predictions to date: {str(monitor_predictions.value_counts().count())}")
    if monitor_predictions.value_counts().count() == 3:
        monitor_results = confusion_matrix(monitor_labels, monitor_predictions)

        df_cm = pd.DataFrame(
            monitor_results, ['True Setosa', 'True Versicolor', 'True Virginica'],
            ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()

        NiceLog.info(f"Saving Iris predictions confusion matrix to: {file_confusion_matrix_save}")
        fig.savefig(file_confusion_matrix_save)

        NiceLog.info(
            f"Uploading Iris predictions confusion matrix to: {hopsworks_images_location}")
        try:
            dataset_api.upload(file_confusion_matrix_save, hopsworks_images_location, overwrite=True)
        except RestAPIError as e:
            NiceLog.error(f"Failed to upload Iris predictions confusion matrix to Hopsworks. Reason: {e}")
            return
        except Exception as e:
            NiceLog.error(
                f"Unexpected error and failed to upload Iris predictions confusion matrix to Hopsworks. Reason: {e}")
            return
        NiceLog.success(f"Iris predictions confusion matrix uploaded to Hopsworks")

    else:
        NiceLog.warn("You need 3 different flower predictions to create the confusion matrix.")
        NiceLog.info("Run the batch inference pipeline more times until you get 3 different iris flower predictions")


    # Initialize
if not LOCAL:
    modal_stub_name = "iris_batch_inference_daily"
    modal_image_libraries = ["hopsworks", "joblib", "seaborn", "scikit-learn==1.1.1", "dataframe-image"]
    run_every_n_days = 1  # Note: must use `modal deploy` to run scheduled app
    hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment

    NiceLog.info(f"Setting Modal stub name to: {BGColors.HEADER}{modal_stub_name}{BGColors.ENDC}")
    stub = modal.Stub(modal_stub_name)

    NiceLog.info(f"Creating a Modal image with Python libraries: {BGColors.HEADER}{', '.join(modal_image_libraries)}"
                 f"{BGColors.ENDC}")
    image = modal.Image.debian_slim().pip_install(modal_image_libraries)

    NiceLog.info(f"Stub should run every: {BGColors.HEADER}{run_every_n_days}{BGColors.ENDC} day(s)")


    @stub.function(image=image, schedule=modal.Period(days=run_every_n_days),
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
        stub.deploy("iris_daily")
        with stub.run():
            f()
