import modal

# Environment check
import os

# Nice printing
from pprint import pprint

# Nice logging
import nice_log as NiceLog
from nice_log import BGColors

# IDE help
from hsfs import feature_store, feature_group
from hopsworks import project as HopsworksProject
from great_expectations.core.expectation_validation_result import ExpectationSuiteValidationResult

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


def generate_flower(name, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min,
                    petal_len_max, petal_len_min, petal_width_max, petal_width_min):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
                       "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
                       "petal_length": [random.uniform(petal_len_max, petal_len_min)],
                       "petal_width": [random.uniform(petal_width_max, petal_width_min)]
                       })
    df['variety'] = name
    return df


def get_random_iris_flower():
    """
    Returns a DataFrame containing one random iris flower
    """
    import random

    virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df = generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0, 3)
    if pick_random >= 2:
        iris_df = virginica_df
        NiceLog.ok("get_random_iris_flower(): Virginica added")
    elif pick_random >= 1:
        iris_df = versicolor_df
        NiceLog.ok("get_random_iris_flower(): Versicolor added")
    else:
        iris_df = setosa_df
        NiceLog.ok("get_random_iris_flower(): Setosa added")

    return iris_df


def g():
    import hopsworks

    NiceLog.header(f"Running function to generate and insert an Iris flower into the Hopsworks feature group!")
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

    NiceLog.info("Generating a random IRIS flower...")
    iris_df = get_random_iris_flower()
    NiceLog.ok(f"Generated flower:")
    pprint(iris_df)

    fg_name = "iris"
    fg_version = 1
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

    NiceLog.info(f"Inserting the generated flower into the {BGColors.HEADER}{fg_name}{BGColors.ENDC} feature group...")
    fg_insert_info = iris_fg.insert(iris_df)
    fg_insert_validation_info: ExpectationSuiteValidationResult = fg_insert_info[1]

    if fg_insert_validation_info.success:
        NiceLog.success("Flower inserted into the feature group.")
    else:
        NiceLog.error("Could not insert flower into group! More info")
        pprint(fg_insert_validation_info)


# Initialize
if not LOCAL:
    modal_stub_name = "iris_daily"
    modal_image_libraries = ["hopsworks"]
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
