def evaluate_pca(wine_df):
    """
    PCA
    Try PCA: https://scribe.rip/@sanchitamangale12/scree-plot-733ed72c8608
    :return:
    """
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Function to apply PCA and plot explained variance
    def apply_pca_and_plot(dataframe):
        # Separating out the features and standardizing them
        features = dataframe.iloc[:, :-1]
        features_standardized = StandardScaler().fit_transform(features)

        # Applying PCA
        pca = PCA()
        principal_components = pca.fit_transform(features_standardized)

        # Calculating explained variance
        explained_variance = pca.explained_variance_ratio_ * 100

        return principal_components, explained_variance

    # Applying PCA to red and white wine datasets
    principal_components, explained_variance = apply_pca_and_plot(wine_df)

    # Plotting the explained variance for both datasets
    plt.figure(figsize=(12, 6))

    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Number of Components')
    plt.ylabel('% Explained Variance')
    plt.title('Explained Variance for Wine')

    plt.tight_layout()
    plt.show()


def train_evaluation(df, target, use_pca_n_components=False, use_model=None, random_state=42,
                     use_resampler=False,
                     classifier_adjust_categories=False,
                     class_knn_n_neighbors=False,
                     reg_logistic_max_iter=1000,
                     reg_logistic_class_weight=False, reg_logistic_c=False, reg_logistic_penalty=False,
                     reg_logistic_solver=False,
                     debug=False, silent=False):
    """

    :param df:
    :type df: pd.DataFrame
    :param target:
    :param use_pca_n_components:
    :param use_model:
    :param random_state:
    :param classifier_adjust_categories: (bins, names), example: ((0, 5, 6, 10), ['low', 'medium', 'high'])
    :return:
    """
    import pandas as pd
    import math
    import seaborn as sns
    from matplotlib import pyplot as plt

    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score

    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

    import numpy as np

    is_classification_model = use_model == "KNeighborsClassifier" or use_model == "RandomForestClassifier"

    # --- [ Features Scalar ] ---------------
    # Scaler used for various methods
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop(target, axis=1))

    # --- [ PCA ] ---------------
    if use_pca_n_components:
        # Standardizing the features
        # scaled_features = scaler.fit_transform(df.drop(target, axis=1))

        # Applying PCA
        pca = PCA(n_components=use_pca_n_components)
        df_pca = pca.fit_transform(scaled_features)

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df_pca, df[target], test_size=0.2,
                                                            random_state=random_state, stratify=df[target])

    # --- [ No PCA ] ------------
    else:
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2,
                                                            random_state=random_state, stratify=df[target])

        # Checking if all quality values are present in both sets
        original_classes = set(df[target].unique())
        train_classes = set(y_train.unique())
        test_classes = set(y_test.unique())

        if not original_classes.issubset(train_classes) or not original_classes.issubset(test_classes):
            print("Original dataset:\n", df[target].value_counts().sort_index())
            print("Training dataset:\n", y_train.value_counts().sort_index())
            print("Testing dataset:\n", y_test.value_counts().sort_index())
            raise Exception(f"It seems that not all {target} values existing in the original "
                            f"dataset are present in both training and testing. "
                            f"Modify the 'random_state' to see if it changes.")

    # --- [ Classification Adjustment ] -------------
    y_train_cat = y_train
    y_test_cat = y_test
    if is_classification_model:
        # Adjusting the function to categorize quality scores with a more inclusive range
        def categorize_quality_adjusted(dataframe):
            # Adjusting bins to be more inclusive
            categories = pd.cut(dataframe[target],
                                classifier_adjust_categories[0], labels=classifier_adjust_categories[1],
                                include_lowest=True)
            return categories

        # Re-categorizing the quality for classification models
        if classifier_adjust_categories:
            y_train_cat = categorize_quality_adjusted(pd.DataFrame(y_train))
            y_test_cat = categorize_quality_adjusted(pd.DataFrame(y_test))

    # --- [ Resampling ] ---------------------
    if debug and use_resampler:
        if is_classification_model:
            print("Before resample (X_train.shape, y_train.shape):\n", (X_train.shape, y_train_cat.shape))
            print("Training dataset:\n", y_train_cat.value_counts().sort_index())
        else:
            print("Before resample (X_train.shape, y_train.shape):\n", (X_train.shape, y_train.shape))
            print("Training dataset:\n", y_train.value_counts().sort_index())

    # Oversampling
    if use_resampler == "oversample":
        if debug:
            print("Using RandomOverSampler")
        ros = RandomOverSampler(random_state=random_state)
        if is_classification_model:
            X_train, y_train_cat = ros.fit_resample(X_train, y_train_cat)
        else:
            X_train, y_train = ros.fit_resample(X_train, y_train)


    # Undersampling
    elif use_resampler == "undersample":
        if debug:
            print("Using RandomUnderSampler")
        rus = RandomUnderSampler(random_state=random_state)
        if is_classification_model:
            X_train, y_train_cat = rus.fit_resample(X_train, y_train_cat)
        else:
            X_train, y_train = rus.fit_resample(X_train, y_train)

    # SMOTE
    elif use_resampler == "smote":
        if debug:
            print("Using SMOTE")
        smote = SMOTE(random_state=random_state)
        try:
            if is_classification_model:
                X_train, y_train_cat = smote.fit_resample(X_train, y_train_cat)
            else:
                X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"Cannot use SMOTE on dataset! Got error: {e}")
            if not silent:
                return 0, pd.DataFrame({'A': []})
            else:
                return 0

    if debug and use_resampler:
        if is_classification_model:
            print("After resample (X_train.shape, y_train.shape):\n", (X_train.shape, y_train_cat.shape))
            print("Training dataset:\n", y_train_cat.value_counts().sort_index())
        else:
            print("After resample (X_train.shape, y_train.shape):\n", (X_train.shape, y_train.shape))
            print("Training dataset:\n", y_train.value_counts().sort_index())

    # --- [ Test/Train Scaling ] ---------------
    if not use_pca_n_components:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        # Already scaled by PCA
        X_train_scaled = X_train
        X_test_scaled = X_test

    # --- [ Model ] -------------
    if use_model == "KNeighborsClassifier":
        # It is recommended to scale the data when using KNN:
        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
        # Use scaled data

        # Cross Validate if n_neighbors are not defined
        if not class_knn_n_neighbors:
            print("No 'n_neighbors' set. Looking for best value...")

            # Max n_neighbors to test ~ sqrt of size of dataset.
            # https://stats.stackexchange.com/questions/534999/why-is-k-sqrtn-a-good-solution-of-the-number-of-neighbors-to-consider
            neighbors_range = range(1, math.floor(math.sqrt(X_train.shape[0])),
                                    2)  # Trying odd values from 1 to sqrt(size)
            cv_scores = []

            for n in neighbors_range:
                classifier = KNeighborsClassifier(n_neighbors=n)
                scores = cross_val_score(classifier, X_train_scaled, y_train_cat, cv=5, scoring='accuracy')
                cv_scores.append(scores.mean())

            # Finding the optimal number of neighbors
            optimal_n = neighbors_range[cv_scores.index(max(cv_scores))]
            print(f"Optimal number of neighbors: {optimal_n}")

            return optimal_n

        else:
            # Retraining KNeighborsClassifier for classification
            classifier = KNeighborsClassifier(n_neighbors=class_knn_n_neighbors).fit(X_train_scaled,
                                                                                     y_train_cat)  # y_train_cat.values.ravel()

            try:
                # Evaluating Classification Model
                y_pred_class = classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test_cat, y_pred_class)
                results = confusion_matrix(y_test_cat, y_pred_class)
                metrics = classification_report(y_test_cat, y_pred_class, output_dict=True)
            except Exception as e:
                print(f"Cannot use KNeighborsClassifier with n_neighbors={class_knn_n_neighbors} on dataset! "
                      f"Got error: {e}")
                if not silent:
                    return 0, pd.DataFrame({'A': []})
                else:
                    return 0

    elif use_model == "RandomForestClassifier":
        # Does not need to use scaled data

        # Retraining RandomForestClassifier for classification
        classifier = RandomForestClassifier(random_state=random_state)
        classifier.fit(X_train, y_train_cat)

        # Evaluating Classification Model
        y_pred_class = classifier.predict(X_test)
        accuracy = accuracy_score(y_test_cat, y_pred_class)
        results = confusion_matrix(y_test_cat, y_pred_class)
        metrics = classification_report(y_test_cat, y_pred_class, output_dict=True)

    elif use_model == "LinearRegression":
        regressor = LinearRegression()
        regressor.fit(X_train, y_train.values.ravel())

        # Evaluating Regression Model
        y_pred = regressor.predict(X_test)

        # Round predicted values to integers
        y_pred = np.rint(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        results = confusion_matrix(y_test, y_pred)
        metrics = classification_report(y_test, y_pred, output_dict=True)

    elif use_model == "LogisticRegression":
        if reg_logistic_class_weight is False or not reg_logistic_c or not reg_logistic_penalty or not reg_logistic_solver:
            print("Parameters not set. Using GridSearchCV to find best...")

            # Setting up the parameter grid to tune
            param_grid = {}
            if not reg_logistic_solver:
                param_grid = {
                    'class_weight': [None, "balanced"],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    # 'penalty': ['l1', 'l2'],
                    # 'solver': ['lbfgs', 'liblinear']
                }
            elif reg_logistic_solver == 'lbfgs':
                param_grid = {
                    'class_weight': [None, "balanced"],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
            elif reg_logistic_solver == 'liblinear':
                param_grid = {
                    'class_weight': [None, "balanced"],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }

            # Initializing the Logistic Regression model
            regressor = LogisticRegression()

            # Setting up the grid search with cross-validation
            grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='accuracy')

            # Fitting grid search
            grid_search.fit(X_train_scaled, y_train)

            # Best parameters and best score
            print("Best Parameters:", grid_search.best_params_)
            print("Best Score:", grid_search.best_score_)

            # Optionally: Evaluate the best model on the test set
            best_logreg = grid_search.best_estimator_
            y_pred = best_logreg.predict(X_test_scaled)
            # ... (use y_pred to evaluate the model on the test set)

            return best_logreg

        else:
            regressor = LogisticRegression(
                max_iter=reg_logistic_max_iter,
                class_weight=reg_logistic_class_weight,
                C=reg_logistic_c,
                penalty=reg_logistic_penalty,
                solver=reg_logistic_solver
            ).fit(X_train_scaled, y_train)

            # Evaluating Regression Model
            y_pred = regressor.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            results = confusion_matrix(y_test, y_pred)
            metrics = classification_report(y_test, y_pred, output_dict=True)

    else:
        raise Exception("No model selected!")

    if not silent:  # Now it became a bad name...
        print(f"Accuracy: {accuracy:.2%}")
        # Create the confusion matrix as a figure, we will later store it as a PNG image file
        true_vals = [f"True {prediction}" for prediction in metrics if
                     prediction not in ['accuracy', 'macro avg', 'weighted avg']]
        pred_vals = [f"Pred {prediction}" for prediction in metrics if
                     prediction not in ['accuracy', 'macro avg', 'weighted avg']]
        df_cm = pd.DataFrame(results, true_vals, pred_vals)
        ### cm = sns.heatmap(df_cm, annot=True)
        ### plt.figure()
        ### cm = sns.heatmap(df_cm, annot=True)
        ### # fig = cm.get_figure()
        ### cm.get_figure()
        ### plt.show()

        return accuracy, df_cm

    return accuracy
