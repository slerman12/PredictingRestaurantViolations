import warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Pre-process data
def all_data(create_file=False):
    # Las Vegas 3 Year Inspections
    all_restaurant_data = pd.read_csv("data/Las Vegas 3 Year Inspections.csv", encoding="ISO-8859-1")

    # Violation codes cheat sheat data
    cheat_sheet = pd.read_csv("data/Cheat Sheet.csv")

    # New data set to hold all provided information regarding restaurant violations
    data = all_restaurant_data.merge(cheat_sheet, on="VIOL VIOLATION CODE")

    # Convert column to date time format
    data["DTA ACTIVITY DATE"] = pd.to_datetime(data["DTA ACTIVITY DATE"])

    # Sort by DTA Activity Date
    data = data.sort_values(by="DTA ACTIVITY DATE")

    # data = data.groupby("GP FACILITY ID").agg(
    #         {"DTA ACTIVITY DATE": lambda x: (x.diff() > pd.Timedelta(days=15)).cumsum()})

    # For each unique restaurant
    for restaurant in data["GP FACILITY ID"].unique():
        # If the difference between two inspection dates is greater than 15 days, mark as unique inspection
        data.loc[data["GP FACILITY ID"] == restaurant, "INSPECTION ID"] = restaurant + ", " + (
            data.loc[data["GP FACILITY ID"] == restaurant, "DTA ACTIVITY DATE"].diff() > pd.Timedelta(
                    days=15)).cumsum().astype(str)

    # data.sort_values(by="GP FACILITY ID").to_csv("test3.csv", index=False)

    # Null inspections catch
    data.loc[pd.isnull(data["INSPECTION ID"]), "INSPECTION ID"] = "NaN"

    # Numerical encode unique inspections
    data["INSPECTION ID"] = preprocessing.LabelEncoder().fit_transform(data["INSPECTION ID"])

    # Sort data
    data = data.sort_values(["GP FACILITY ID", "INSPECTION ID", "VIOL VIOLATION STATUS", "VIOL VIOLATION CODE"])

    # Add Grade column
    data.loc[data["DTA RESULT CODE"] == 31, "GRADE"] = "A"
    data.loc[data["DTA RESULT CODE"] == 35, "GRADE"] = "B"
    data.loc[data["DTA RESULT CODE"] == 30, "GRADE"] = "C"
    data.loc[(data["DTA RESULT CODE"] == 86) | (data["DTA RESULT CODE"] == 87), "GRADE"] = "Closure"

    # Fix risk categories
    data.loc[data["GP RISK CATEGORY"] == '1-Jan', "GP RISK CATEGORY"] = 0
    data.loc[data["GP RISK CATEGORY"] == '2-Jan', "GP RISK CATEGORY"] = 1
    data.loc[data["GP RISK CATEGORY"] == '3-Jan', "GP RISK CATEGORY"] = 2
    data.loc[data["GP RISK CATEGORY"] == '2', "GP RISK CATEGORY"] = 3
    data.loc[data["GP RISK CATEGORY"] == '3', "GP RISK CATEGORY"] = 4
    data.loc[data["GP RISK CATEGORY"] == '1-Apr', "GP RISK CATEGORY"] = 5
    data.loc[data["GP RISK CATEGORY"] == '2-Apr', "GP RISK CATEGORY"] = 6
    data.loc[data["GP RISK CATEGORY"] == '3-Apr', "GP RISK CATEGORY"] = 7

    # Initialize imminent health hazard feature
    data["Imminent Health Hazard"] = 0

    # Set imminent health hazard feature
    data.loc[data["VIOLATION NUMBER ON INSPECTION REPORT"] == "Imminent Health Hazard", "Imminent Health Hazard"] = 1

    # Create CSV
    # data.to_csv("data/clean_data_old1.csv", index=False)

    # Numerically encode repeat violations
    data.loc[data["VIOL REPEAT VIOLATION"] == "Y", "VIOL REPEAT VIOLATION"] = 1
    data.loc[(data["VIOL REPEAT VIOLATION"] == "N") | (data["VIOL REPEAT VIOLATION"].isnull()),
             "VIOL REPEAT VIOLATION"] = 0

    # Numerically encode irregular Hours
    data.loc[data["GP IRREGULAR HOURS"] == "Y", "GP IRREGULAR HOURS"] = 1
    data.loc[data["GP IRREGULAR HOURS"] == "N", "GP IRREGULAR HOURS"] = 0

    # Create CSV
    # data.to_csv("data/clean_data_old1.csv", index=False)

    # Load data
    # clean_data = pd.read_csv("data/clean_data_old1.csv")
    # raw_data = pd.read_csv("data/Las Vegas 3 Year Inspections.csv", encoding="ISO-8859-1")

    # Merge activity date back into dataset
    # data = clean_data.merge(raw_data[['FACILITY ID', 'FACILITY DISTRICT', 'GP SQUARE FOOTAGE', 'GP PE',
    #                                   'VIOL VIOLATION STATUS', 'VIOL VIOLATION CODE', 'DTA SCORE',
    #                                   'DTA ACTIVITY DATE', 'DTA SERVICE']],
    #                         left_on=['Restaurant ID', 'District', 'Square Footage', 'GP PE', 'Violation Status',
    #                                  'Violation Code', 'Demerit Total'],
    #                         right_on=['FACILITY ID', 'FACILITY DISTRICT', 'GP SQUARE FOOTAGE', 'GP PE',
    #                                   'VIOL VIOLATION STATUS', 'VIOL VIOLATION CODE', 'DTA SCORE'], copy=False).drop(
    #         ['FACILITY ID', 'FACILITY DISTRICT', 'GP SQUARE FOOTAGE',
    #          'VIOL VIOLATION STATUS', 'VIOL VIOLATION CODE', 'DTA SCORE'], axis=1)

    # Only include violation status 2
    # data = data[data["Violation Status"] == 2]

    # Create CSV
    # data.to_csv("data/CleanData2.csv", index=False)

    # Sort data
    data = data.sort_values(["FACILITY ID", "INSPECTION ID", "DTA ACTIVITY DATE", "VIOL VIOLATION CODE"])

    # Rename columns
    data = data.rename(columns={'FACILITY ID': 'Restaurant ID', 'FACILITY DISTRICT': 'District',
                                'GP SQUARE FOOTAGE': 'Square Footage', 'VIOL VIOLATION STATUS': 'Violation Status',
                                'VIOL VIOLATION CODE': 'Violation Code', 'DTA SCORE': 'Demerit Total',
                                'INSPECTION ID': 'Inspection ID', 'GRADE': 'Grade', 'DTA SERVICE': 'DTA Service',
                                'DTA ACTIVITY DATE': 'Date', 'GP IRREGULAR HOURS': 'Irregular Hours',
                                'GP RISK CATEGORY': 'Risk Category', 'VIOL REPEAT VIOLATION': 'Repeat Violation'})

    # Create CSV
    if create_file:
        data.to_csv("data/all_data_{}.csv".format(datetime.date.today().strftime("_%d_%B_%Y")), index=False)

    # Return data
    return data


# Process data
def clean_data(data, create_file=False):
    # Sort data
    data = data.sort_values(["Restaurant ID", "Inspection ID", "Date", "Violation Status", "Violation Code"])

    # Drop certain features
    data = data.drop(
            ["FACILITY NAME", "FACILITY ADDRESS", "FACILITY CITY", "FACILITY STATE", "FACILITY ZIP", "GP RECORD ID",
             "GP FACILITY ID", "GP PROGRAM IDENTIFIER", "DTA RESULT CODE", "VIOL VIOLATION ID",
             "VIOL VIOLATION DESCRIPTION_x", "VIOLATION NUMBER ON INSPECTION REPORT", "VIOL VIOLATION DESCRIPTION_y"],
            axis=1)

    # Rearrange columns
    data = data[
        ["Restaurant ID", "District", "Risk Category", "Square Footage", "GP PE", "Irregular Hours", "Inspection ID",
         "Date", "DTA Service", "Violation Status", "Violation Code", "Demerit Value", "Violation Type",
         "Imminent Health Hazard", "Repeat Violation", "Demerit Total", "Grade"]]

    # Create CSV
    # data.to_csv("data/clean_data_old1.csv", index=False)

    # # Convert column to date time format
    data["Date"] = pd.to_datetime(data["Date"])

    # Add binary followup feature
    data["Followup"] = 0
    data.loc[data["Date"] != data.groupby(["Inspection ID"])["Date"].transform("min"), "Followup"] = 1

    # Rearrange columns
    data = data[
        ['Restaurant ID', 'District', 'Risk Category', 'Square Footage', 'GP PE', 'Irregular Hours', 'Inspection ID',
         'Followup', 'Date', 'DTA Service', 'Violation Status', 'Violation Code', 'Demerit Value', 'Violation Type',
         'Imminent Health Hazard', 'Repeat Violation', 'Demerit Total', 'Grade']].rename(
            columns={"GP PE": "Program Element"})

    # Create CSV
    # data.to_csv("data/clean_data_old2.csv", index=False)

    # District groups
    district_groups = {"NW": [28, 19, 53, 12, 14, 37, 26], "NE": [16, 17, 5, 1, 35, 30, 13, 2, 4, 64, 32],
                       "C": [27, 24, 97, 91, 48, 96, 90, 41, 40, 94, 93, 52, 51, 92, 98, 95, 8, 3, 31, 18, 39, 55, 33,
                             54, 56],
                       "SW": [62, 6, 36, 43, 46, 42, 67, 47, 7, 63, 45, 65, 11, 44, 59, 66, 15, 49, 60, 61, 38],
                       "SE": [10, 57, 58, 34, 9, 21, 20, 23, 22]}

    # Add group categories to data
    for key, value in district_groups.items():
        data.loc[data["District"].isin(value), "District Group"] = key

    # Rearrange columns
    data = data[
        ['Restaurant ID', 'District', 'District Group', 'Risk Category', 'Square Footage', 'Program Element',
         'Irregular Hours', 'Inspection ID', 'Followup', 'Date', 'DTA Service', 'Violation Status', 'Violation Code',
         'Demerit Value', 'Violation Type', 'Imminent Health Hazard', 'Repeat Violation', 'Demerit Total', 'Grade']]

    # Turn string mixed numbers into floats
    data.loc[data["Square Footage"].notnull(), "Square Footage"] = \
        data["Square Footage"].astype(str).str.replace("`", "").str.replace(",", "").astype(float)

    # Only use inspections with DTA Service 916, since the only other value 919 is not important
    data = data[data["DTA Service"] == 916]

    # Drop DTA Service field
    data = data.drop("DTA Service", axis=1)

    # Use 0 for null Irregular Hours
    data["Irregular Hours"] = data["Irregular Hours"].fillna(0)

    # Set unique district group for 0 districts
    data.loc[data["District"] == 0, "District Group"] = "0"

    # Create CSV
    if create_file:
        data.to_csv("data/clean_data{}.csv".format(datetime.date.today().strftime("_%d_%B_%Y")), index=False)

    # Return data
    return data


# Prepare data for modeling
def training_data(data, create_file=False):
    # Numerically encode training data
    for cur_value, new_value in {"A": 0, "B": 1, "C": 2, "Closure": 3}.items():
        data.loc[data["Grade"] == cur_value, "Grade"] = new_value

    # Create training data
    data = data[data["Followup"] == 0].groupby(["Restaurant ID", "Inspection ID", "Program Element"]).max().groupby(
            level=["Restaurant ID", "Inspection ID"]).agg(
            {"Square Footage": np.nansum, "Risk Category": np.max, "Grade": np.max, "Demerit Total": np.mean,
             "District": "last", "District Group": "last", "Irregular Hours": "last"})

    # Rearrange columns
    data = data[["District", "District Group", "Square Footage", "Risk Category", "Irregular Hours", "Demerit Total",
                 "Grade"]]

    # Reset index
    data = data.reset_index()

    # Remove negligible minority of restaurants with null districts, risk categories, or grades
    data = data[(data["District"].notnull()) & (data["Risk Category"].notnull()) & (data["Grade"].notnull())]

    # Add column for restaurant size categories (very small , … , very large) based on histogram of sizes
    data.loc[(data["Square Footage"] < 100), "Size"] = 0
    data.loc[(data["Square Footage"] >= 100) & (data["Square Footage"] < 350), "Size"] = 1
    data.loc[(data["Square Footage"] >= 350) & (data["Square Footage"] < 1150), "Size"] = 2
    data.loc[(data["Square Footage"] >= 1150) & (data["Square Footage"] < 1950), "Size"] = 3
    data.loc[(data["Square Footage"] >= 1950), "Size"] = 4

    # Rearrange columns
    data = data[["Restaurant ID", "Inspection ID", "District", "District Group", "Square Footage", "Size",
                 "Risk Category", "Irregular Hours", "Demerit Total", "Grade"]]

    # Numerically encode district groups
    data.loc[data["District Group"] == "0", "District Group"] = 0
    data.loc[data["District Group"] == "NW", "District Group"] = 1
    data.loc[data["District Group"] == "NE", "District Group"] = 2
    data.loc[data["District Group"] == "SW", "District Group"] = 3
    data.loc[data["District Group"] == "SE", "District Group"] = 4
    data.loc[data["District Group"] == "C", "District Group"] = 5

    # Remove negligible minority of restaurants with null district groups
    data = data[data["District Group"].notnull()]

    # Number of inspections per restaurant
    num_inspections_data = data.groupby(["Restaurant ID"]).size().reset_index(name='count')

    # New restaurants (those with only a single 916 & non-followup inspection)
    new_restaurants = num_inspections_data.loc[num_inspections_data["count"] == 1, "Restaurant ID"].unique()

    # New restaurant feature
    data["New Restaurant"] = 0
    data.loc[data["Restaurant ID"].isin(new_restaurants), "New Restaurant"] = 1

    # Create training data
    if create_file:
        data.to_csv("data/training_data{}.csv".format(datetime.date.today().strftime("_%d_%B_%Y")), index=False)

        # Last (non-followup) inspections only
        last_inspections = data.groupby(["Restaurant ID"]).agg({"Inspection ID": max})["Inspection ID"]
        last_inspections_data = data[data["Inspection ID"].isin(last_inspections)]

        # Initialize "prior B or worse" and "prior C or worse" features
        last_inspections_data["Prior B or Worse"] = 0
        last_inspections_data["Prior C or Worse"] = 0

        # All prior inspections
        prior__inspections_data = data[~data["Inspection ID"].isin(last_inspections)]

        # Prior inspections that received B or worse, and C or worse
        b_or_worse = prior__inspections_data.loc[prior__inspections_data["Grade"] >= 1, "Restaurant ID"].unique()
        c_or_worse = prior__inspections_data.loc[prior__inspections_data["Grade"] >= 2, "Restaurant ID"].unique()

        # Set prior B/C or worse features
        last_inspections_data.loc[last_inspections_data["Restaurant ID"].isin(b_or_worse), "Prior B or Worse"] = 1
        last_inspections_data.loc[last_inspections_data["Restaurant ID"].isin(c_or_worse), "Prior C or Worse"] = 1

        # Create CSV
        last_inspections_data.to_csv("data/training_data_last_inspections_only{}.csv".format(
                datetime.date.today().strftime("_%d_%B_%Y")), index=False)

        # Return data
        return last_inspections_data

    # Return data
    return data


# Statistics
def stats(histogram=None, bar=None, show=True):
    # Histogram
    if histogram is not None:
        # Plot info for each histogram
        for plot_info in histogram["info"]:
            # Plot histogram
            plt.hist(plot_info["data"], bins="auto" if "bins" not in plot_info else plot_info["bins"],
                     alpha=0.5 if "alpha" not in plot_info else plot_info["alpha"],
                     label=None if "label" not in plot_info else plot_info["label"])

        # Legend
        if "legend" in histogram:
            plt.legend(loc="upper right")

        # Title
        if "title" in histogram:
            plt.title(histogram["title"])

        # X label
        if "xlabel" in histogram:
            plt.xlabel(histogram["xlabel"])

        # Y label
        if "ylabel" in histogram:
            plt.ylabel(histogram["ylabel"])

        # Show plot
        if show:
            plt.show()

    if bar is not None:
        # x Positions
        xpos = np.arange(len(bar["labels"]))

        # Create bar graph
        plt.bar(xpos, bar["yvalues"], align='center', alpha=0.5)
        plt.xticks(xpos, bar["labels"])

        # Title
        if "title" in bar:
            plt.title(bar["title"])

        # X label
        if "xlabel" in bar:
            plt.xlabel(bar["xlabel"])

        # Y label
        if "ylabel" in bar:
            plt.ylabel(bar["ylabel"])

        if show:
            plt.show()


# Modeling
def modeling(data, predictors, target):
    # Describe training data
    print("\nTRAINING DATA DESCRIPTION\n")
    print(data[predictors].describe())

    # Describe target data
    print("\nTARGET DESCRIPTION\n")
    print(data[target].describe())
    (print(data[target].value_counts()))

    # Algorithms for model
    algs = [
        RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, oob_score=True),
        LogisticRegression(),
        SVC(probability=True),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=25),
        GradientBoostingClassifier(n_estimators=10, max_depth=3)]

    # Algoirthm names
    names = ["Random Forest", "Logistic Regression", "SVM", "Gaussian Naive Bayes", "kNN", "Gradient Boosting"]

    for index, alg in enumerate(algs):
        # Alg name
        name = names[index]

        # Fit alg
        alg.fit(data[predictors], data[target])

        # Base score
        score = alg.score(data[predictors], data[target])
        print("Base Score: {} [{}]".format(score, name))

        # Out of bag estimate
        if name == "Random Forest":
            score = alg.oob_score_
            print("OOB Score: {} [{}]".format(score, name))

        # Cross validation
        scores = cross_val_score(alg, data[predictors], data[target], cv=5, scoring="accuracy", n_jobs=-1)
        print("Cross Validation: {:0.2f} (+/- {:0.2f}) [{}] ({})".format(abs(scores.mean()), scores.std(), name,
                                                                         "accuracy"))

        # Feature importances
        if name == "Random Forest":
            fi = zip(predictors, alg.feature_importances_)
            output = sorted(fi, key=lambda x: x[1])
            print("Feature Importances [" + name + "]")
            for feature, imp in output:
                print(feature, imp)

        # Split the data into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=1.0 / 5)

        # Print ratio of split
        print("{:g}/{:g} Split: ".format(100 - 100 / 5, 100 / 5))

        # Create predictions
        y_pred = alg.fit(X_train, y_train).predict(X_test)

        # Split classification report
        print("Classification Report [" + name + "]")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(classification_report(y_test, y_pred))

        # Split confusion matrix
        # Print algorithm name
        print("Confusion Matrix [" + name + "]")

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print("Not Normalized:")
        print(cm)

        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized:")
        print(cm_normalized)

        # Configure confusion matrix plot
        def plot_confusion_matrix(cm, title="Confusion matrix", cmap=plt.cm.Blues):
            plt.imshow(cm, interpolation="nearest", cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(4)
            plt.xticks(tick_marks, [0, 1, 2, 3], rotation=45)
            plt.yticks(tick_marks, [0, 1, 2, 3])
            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")

        # Plot normalized confusion matrix
        plt.figure()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_confusion_matrix(cm_normalized, title="Normalized Confusion Matrix\n[{}]".format(name))

        # Show confusion matrix plots
        # plt.show()


# Main method
if __name__ == "__main__":
    # Load data
    clean = pd.read_csv("data/clean_data_25_July_2017.csv")
    train = pd.read_csv("data/training_data_21_August_2017.csv")
    last_ins_train = pd.read_csv("data/training_data_first_inspections_only_21_August_2017.csv")

    # Compute histogram on restaurant sizes
    # stats(histogram={"info": [{"data": first_ins_train["Square Footage"]}], "xlabel": "Restaurant Size (Square Ft.)",
    #                  "ylabel": "Frequency", "title": "Restaurant Size Frequencies"})

    # Run
    last_ins_train = training_data(clean, create_file=True)

    # Number of inspections per restaurant
    # num_inspections = train.groupby(["Restaurant ID"]).agg({"Inspection ID": "count"})["Inspection ID"]

    # Histogram of # of inspections per restaurant
    # stats(histogram={"info": [{"data": num_inspections}], "xlabel": "Number of Inspections",
    #                  "ylabel": "Frequency", "title": "Number of Inspections Per Restaurant"})

    print("\nALL RESTAURANTS\n")
    print(last_ins_train[["Demerit Total", "Grade"]].describe())

    print("\nBY SIZE\n")
    print(last_ins_train.groupby("Size")[["Demerit Total", "Grade"]].describe())

    print("\nBY DISTRICT\n")
    print(last_ins_train.groupby("District Group")[["Demerit Total", "Grade"]].describe())

    # Frequency of violations
    print("\nFREQUENCY OF VIOLATIONS\n")
    print(clean.loc[clean["Violation Status"] == 2, "Violation Code"].value_counts())

    # Print outcome measure
    print("\nCLASSIFYING:\nA or B vs. C or Closure")

    # Prepare target
    last_ins_train.loc[(last_ins_train["Grade"] == 0) | (last_ins_train["Grade"] == 1), "Grade1"] = 0
    last_ins_train.loc[(last_ins_train["Grade"] == 2) | (last_ins_train["Grade"] == 3), "Grade1"] = 1

    # Model
    modeling(last_ins_train, ["District", "District Group", "Square Footage", "Size", "Risk Category",
                               "Irregular Hours", "New Restaurant", "Prior B or Worse", "Prior C or Worse"], "Grade1")

    # Print outcome measure
    print("\nCLASSIFYING:\nA vs. B, C, or Closure")

    # Prepare target
    last_ins_train.loc[(last_ins_train["Grade"] == 0), "Grade2"] = 0
    last_ins_train.loc[(last_ins_train["Grade"] >= 1), "Grade2"] = 1

    # Model
    modeling(last_ins_train, ["District", "District Group", "Square Footage", "Size", "Risk Category",
                              "Irregular Hours", "New Restaurant", "Prior B or Worse", "Prior C or Worse"], "Grade2")
