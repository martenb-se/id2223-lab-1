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

# Error help
from hopsworks import RestAPIError

# Settings
# - Modal
modal_stub_name = "wine_daily"
modal_image_libraries = ["hopsworks"]
# - Modal Deployment
model_run_every_n_days = 1  # Note: must use `modal deploy` to run scheduled app
# - Hopsworks
hopsworks_api_key_modal_secret_name = "hopsworks-api-key"  # Load secret to environment
# Names
# - Models
model_red_wine_name = "wine_red_model"
model_red_wine_version = 1
model_white_wine_name = "wine_white_model"
model_white_wine_version = 1
# - Feature Groups
fg_wine_name = "wine"
fg_wine_version = 1
fg_type_red = "red"
fg_type_white = "white"
# - - - Feature Views
fw_wine_name = "wine"
fw_wine_version = 1

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


def calculate_feature_stats_by_quality(dataframe, quality):
    """
    Calculates the mean and standard deviation for each feature in the dataframe based on a specified quality.
    :param dataframe: The wine dataset.
    :param quality: The target quality.
    :return: A DataFrame containing the mean and standard deviation for each feature based on the chosen quality.
    """
    quality_df = dataframe[dataframe['quality'] == quality]

    if not quality_df.empty:
        mean = quality_df.mean()
        std = quality_df.std()
    else:
        # If there are no wines with the specified quality, use the overall mean and std from the whole dataset.
        mean = dataframe.mean()
        std = dataframe.std()

    return mean, std


def generate_wine_by_quality(dataframe, quality, type):
    """
    Generates a synthetic wine sample based on a specified quality.
    :param dataframe: The wine dataset.
    :param quality: The target quality.
    :param type: The target type
    :return: A DataFrame containing a single row representing the synthetic wine sample.
    """
    import random
    import pandas as pd

    mean, std = calculate_feature_stats_by_quality(dataframe, quality)
    wine_sample = {attribute: random.gauss(mean[attribute], std[attribute])
                   for attribute in mean.index if attribute != 'quality'}

    wine_sample['quality'] = quality
    wine_sample['type'] = type
    return pd.DataFrame([wine_sample])


def select_quality_based_on_distribution(dataframe):
    """
    Select a wine quality based on the distribution in the presented dataframe to make it more realistic.
    This ultimately means very few low and high quality wines will be generated.

    :param dataframe: The wine dataset to base the distributions on.
    :return: A selected quality number to generate.
    """
    import random

    quality_counts = dataframe['quality'].value_counts(normalize=True)
    return random.choices(quality_counts.index, weights=quality_counts, k=1)[0]


def generate_wine_by_distribution(wine_fg, wine_type):
    """
    Generates a synthetic wine sample based on the distribution of qualities.
    :param wine_fg:
    :param wine_type:
    :return:
    """
    import pandas as pd
    # Query
    wine_query: hsfs_query.Query = wine_fg.select_all()

    # Get DataFrame
    NiceLog.info(f"Getting {BGColors.HEADER}{wine_fg.name}{BGColors.ENDC} "
                 f"(version {wine_fg.version}) pandas DataFrame...")
    try:
        wine_df: pd.DataFrame = wine_query.read()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksQueryReadError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksQueryReadError()
    NiceLog.info(f"Feature group is named: {BGColors.HEADER}{wine_fg.name}{BGColors.ENDC} "
                 f"({wine_fg.description})")

    # Filter by type to get correct distributions
    wine_df = wine_df[wine_df["type"] == wine_type]
    wine_df = wine_df.drop(columns=["type"])

    # Generate a wine
    target_quality = select_quality_based_on_distribution(wine_df)
    return generate_wine_by_quality(wine_df, target_quality, wine_type)


def g():
    import hopsworks
    import random

    NiceLog.header(f"Running function to generate and insert an Wine into the Hopsworks feature group!")

    if "HOPSWORKS_API_KEY" not in os.environ:
        NiceLog.error(f"Failed to log in to Hopsworks. HOPSWORKS_API_KEY is not in the current environment.")
        raise HopsworksNoAPIKey()

    # Log in
    NiceLog.info("Logging in to Hopsworks...")
    try:
        # As long as os.environ["HOPSWORKS_API_KEY"] is set, Hopsworks should not ask for user input
        project: HopsworksProject.Project = hopsworks.login()
    except RestAPIError as e:
        NiceLog.error(f"Failed to log in to Hopsworks. Reason: {e}")
        raise HopsworksLoginError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to log in to Hopsworks. Reason: {e}")
        raise HopsworksLoginError()
    NiceLog.success("Logged in to Hopsworks!")

    # Get the feature store
    NiceLog.info(f"Getting {BGColors.HEADER}{project.name}{BGColors.ENDC} feature store...")
    try:
        fs: feature_store.FeatureStore = project.get_feature_store()
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature store from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureStoreError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature store from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureStoreError()

    NiceLog.success("Gotten feature store!")
    NiceLog.info(f"Feature store is named: {BGColors.HEADER}{fs.name}{BGColors.ENDC})")

    # Make wine (50% red, 50% white even if they are not distributed evenly)
    NiceLog.info("Generating a random wine...")
    if random.random() < 0.5:
        cur_wine_type = fg_type_red
    else:
        cur_wine_type = fg_type_white

    # Get Feature Group
    NiceLog.info(f"Getting {BGColors.HEADER}{fg_wine_name}{BGColors.ENDC} "
                 f"(version {fg_wine_version}) feature group...")
    try:
        wine_fg: feature_group.FeatureGroup = (
            fs.get_feature_group(name=fg_wine_name, version=fg_wine_version))
    except RestAPIError as e:
        NiceLog.error(f"Failed to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureGroupError()
    except Exception as e:
        NiceLog.error(f"Unexpected error when trying to get feature group from Hopsworks project. Reason: {e}")
        raise HopsworksGetFeatureGroupError()
    NiceLog.info(f"Feature group is named: {BGColors.HEADER}{wine_fg.name}{BGColors.ENDC} "
                 f"({wine_fg.description})")

    # Generate
    cur_wine = generate_wine_by_distribution(wine_fg, cur_wine_type)

    NiceLog.info(f"Generated a {BGColors.HEADER}{cur_wine_type}{BGColors.ENDC} wine:")
    pprint(cur_wine)

    NiceLog.info(
        f"Inserting the generated wine into the {BGColors.HEADER}{fg_wine_name}{BGColors.ENDC} feature group...")
    fg_insert_info = wine_fg.insert(cur_wine)
    fg_insert_job: feature_group.Job = fg_insert_info[0]
    fg_insert_validation_info: ExpectationSuiteValidationResult = fg_insert_info[1]

    if fg_insert_validation_info is None:
        NiceLog.info(f"Check job {fg_insert_job.name} manually at the provided link.")
    else:
        if fg_insert_validation_info.success:
            NiceLog.success("Wine inserted into the feature group.")
        else:
            NiceLog.error("Could not insert wine into group! More info: ")
            pprint(fg_insert_validation_info)
            raise HopsworksFeatureGroupInsertError()


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
