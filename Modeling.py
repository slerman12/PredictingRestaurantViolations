import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime


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
def training_data(data, create_file=False, first_inspections_only=False):
    # Numerically encode training data
    for cur_value, new_value in {"A": 0, "B": 1, "C": 2, "Closure": 3}.items():
        data.loc[data["Grade"] == cur_value, "Grade"] = new_value

    # Create training data
    data = data[data["Followup"] == 0].groupby(["Restaurant ID", "Inspection ID", "Program Element"]).max().groupby(
            level=["Restaurant ID", "Inspection ID"]).agg(
            {"Square Footage": np.nansum, "Risk Category": np.max, "Grade": np.max, "District": "last",
             "District Group": "last", "Irregular Hours": "last"})

    # Rearrange columns
    data = data[["District", "District Group", "Square Footage", "Risk Category", "Irregular Hours", "Grade"]]

    # Reset index
    data = data.reset_index()

    # Create training data
    if not first_inspections_only:
        # Create CSV
        if create_file:
            data.to_csv("data/training_data{}.csv".format(datetime.date.today().strftime("_%d_%B_%Y")), index=False)
    else:
        # First inspections only
        first_inspections = data.groupby(["Restaurant ID"]).agg({"Inspection ID": min})["Inspection ID"]
        data = data[data["Inspection ID"].isin(first_inspections)]

        # Create CSV
        if create_file:
            data.to_csv("data/training_data_first_inspections_only{}.csv".format(
                datetime.date.today().strftime("_%d_%B_%Y")), index=False)

    # Return data
    return data


# Statistics
def stats(data):
    return


# Modeling
def modeling(data):
    return


# Main method
if __name__ == "__main__":
    # Load data
    clean = pd.read_csv("data/clean_data_25_July_2017.csv")
    train = pd.read_csv("data/training_data_25_July_2017.csv")
    first_ins_train = pd.read_csv("data/training_data_first_inspections_only_25_July_2017.csv")

    # Run
    first_ins_train = training_data(clean, create_file=False, first_inspections_only=True)

    print(first_ins_train.isnull().sum())

    print(first_ins_train.loc[first_ins_train["District Group"].isnull(), "District"].unique())
