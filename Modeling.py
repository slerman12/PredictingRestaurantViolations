import pandas as pd
from sklearn import preprocessing

# Main method
if __name__ == "__main__":
    # # Las Vegas 3 Year Inspections
    # all_restaurant_data = pd.read_csv("data/Las Vegas 3 Year Inspections.csv", encoding="ISO-8859-1")
    #
    # # Violation codes cheat sheat data
    # cheat_sheet = pd.read_csv("data/Cheat Sheet.csv")
    #
    # # New data set to hold all provided information regarding restaurant violations
    # data = all_restaurant_data.merge(cheat_sheet, on="VIOL VIOLATION CODE")
    #
    # # Convert column to date time format
    # data["DTA ACTIVITY DATE"] = pd.to_datetime(data["DTA ACTIVITY DATE"])
    #
    # # Sort by DTA Activity Date
    # data = data.sort_values(by="DTA ACTIVITY DATE")
    #
    # # data = data.groupby("GP FACILITY ID").agg(
    # #         {"DTA ACTIVITY DATE": lambda x: (x.diff() > pd.Timedelta(days=15)).cumsum()})
    #
    # # For each unique restaurant
    # for restaurant in data["GP FACILITY ID"].unique():
    #     # If the difference between two inspection dates is greater than 15 days, mark as unique inspection
    #     data.loc[data["GP FACILITY ID"] == restaurant, "INSPECTION ID"] = restaurant + ", " + (
    #         data.loc[data["GP FACILITY ID"] == restaurant, "DTA ACTIVITY DATE"].diff() > pd.Timedelta(
    #             days=15)).cumsum().astype(str)
    #
    # # data.sort_values(by="GP FACILITY ID").to_csv("test3.csv", index=False)
    #
    # # Load merged data
    # # data = pd.read_csv("data/Merged Data.csv")
    #
    # # Null inspections catch
    # data.loc[pd.isnull(data["INSPECTION ID"]), "INSPECTION ID"] = "NaN"
    #
    # # Numerical encode unique inspections
    # data["INSPECTION ID"] = preprocessing.LabelEncoder().fit_transform(data["INSPECTION ID"])
    #
    # # Sort data
    # data = data.sort_values(["GP FACILITY ID", "INSPECTION ID", "VIOL VIOLATION STATUS", "VIOL VIOLATION CODE"])
    #
    # # Make CSV
    # # data.to_csv("data/AllData.csv", index=False)
    #
    # # Load data
    # # data = pd.read_csv("data/AllData.csv", encoding="ISO-8859-1")
    #
    # # Add Grade column
    # data.loc[data["DTA RESULT CODE"] == 31, "GRADE"] = "A"
    # data.loc[data["DTA RESULT CODE"] == 35, "GRADE"] = "B"
    # data.loc[data["DTA RESULT CODE"] == 30, "GRADE"] = "C"
    # data.loc[(data["DTA RESULT CODE"] == 86) | (data["DTA RESULT CODE"] == 87), "GRADE"] = "Closure"
    #
    # # Make CSV
    # # data.to_csv("data/AllData.csv", index=False)
    #
    # # Load data
    # # data = pd.read_csv("data/AllData.csv", dtype={"GP RISK CATEGORY": str})
    #
    # # Fix risk categories
    # data.loc[data["GP RISK CATEGORY"] == '1-Jan', "GP RISK CATEGORY"] = 0
    # data.loc[data["GP RISK CATEGORY"] == '2-Jan', "GP RISK CATEGORY"] = 1
    # data.loc[data["GP RISK CATEGORY"] == '3-Jan', "GP RISK CATEGORY"] = 2
    # data.loc[data["GP RISK CATEGORY"] == '2', "GP RISK CATEGORY"] = 3
    # data.loc[data["GP RISK CATEGORY"] == '3', "GP RISK CATEGORY"] = 4
    # data.loc[data["GP RISK CATEGORY"] == '1-Apr', "GP RISK CATEGORY"] = 5
    # data.loc[data["GP RISK CATEGORY"] == '2-Apr', "GP RISK CATEGORY"] = 6
    # data.loc[data["GP RISK CATEGORY"] == '3-Apr', "GP RISK CATEGORY"] = 7
    #
    # # Create CSV
    # # data.to_csv("data/AllData.csv", index=False)
    #
    # # Load data
    # # data = pd.read_csv("data/CleanData.csv")
    #
    # # Initialize imminent health hazard feature
    # data["Imminent Health Hazard"] = 0
    #
    # # Set imminent health hazard feature
    # data.loc[data["VIOLATION NUMBER ON INSPECTION REPORT"] == "Imminent Health Hazard", "Imminent Health Hazard"] = 1
    #
    # # Create CSV
    # # data.to_csv("data/CleanData.csv", index=False)
    #
    # # Load data
    # # data = pd.read_csv("data/CleanData.csv")
    #
    # # Numerically encode repeat violations
    # data.loc[data["VIOL REPEAT VIOLATION"] == "Y", "VIOL REPEAT VIOLATION"] = 1
    # data.loc[(data["VIOL REPEAT VIOLATION"] == "N") | (data["VIOL REPEAT VIOLATION"].isnull()), "VIOL REPEAT VIOLATION"] = 0
    #
    # # Numerically encode irregular Hours
    # data.loc[data["GP IRREGULAR HOURS"] == "Y", "GP IRREGULAR HOURS"] = 1
    # data.loc[data["GP IRREGULAR HOURS"] == "N", "GP IRREGULAR HOURS"] = 0
    #
    # # Create CSV
    # # data.to_csv("data/CleanData.csv", index=False)
    #
    # # Load data
    # # clean_data = pd.read_csv("data/CleanData.csv")
    # # raw_data = pd.read_csv("data/Las Vegas 3 Year Inspections.csv", encoding="ISO-8859-1")
    #
    # # Merge activity date back into dataset
    # # data = clean_data.merge(raw_data[['FACILITY ID', 'FACILITY DISTRICT', 'GP SQUARE FOOTAGE', 'GP PE',
    # #                                   'VIOL VIOLATION STATUS', 'VIOL VIOLATION CODE', 'DTA SCORE',
    # #                                   'DTA ACTIVITY DATE', 'DTA SERVICE']],
    # #                         left_on=['Restaurant ID', 'District', 'Square Footage', 'GP PE', 'Violation Status',
    # #                                  'Violation Code', 'Demerit Total'],
    # #                         right_on=['FACILITY ID', 'FACILITY DISTRICT', 'GP SQUARE FOOTAGE', 'GP PE',
    # #                                   'VIOL VIOLATION STATUS', 'VIOL VIOLATION CODE', 'DTA SCORE'], copy=False).drop(
    # #         ['FACILITY ID', 'FACILITY DISTRICT', 'GP SQUARE FOOTAGE',
    # #          'VIOL VIOLATION STATUS', 'VIOL VIOLATION CODE', 'DTA SCORE'], axis=1)
    #
    # # Only include violation status 2
    # # data = data[data["Violation Status"] == 2]
    #
    # # Create CSV
    # # data.to_csv("data/CleanData2.csv", index=False)
    #
    # # Load data
    # # clean_data = pd.read_csv("data/CleanData2.csv")
    #
    # # Sort data
    # data = data.sort_values(["FACILITY ID", "INSPECTION ID", "DTA ACTIVITY DATE", "VIOL VIOLATION CODE"])
    #
    # # Rename columns
    # data = data.rename(columns={'FACILITY ID': 'Restaurant ID', 'FACILITY DISTRICT': 'District',
    #                             'GP SQUARE FOOTAGE': 'Square Footage', 'VIOL VIOLATION STATUS': 'Violation Status',
    #                             'VIOL VIOLATION CODE': 'Violation Code', 'DTA SCORE': 'Demerit Total',
    #                             'INSPECTION ID': 'Inspection ID', 'GRADE': 'Grade', 'DTA SERVICE': 'DTA Service',
    #                             'DTA ACTIVITY DATE': 'Date', 'GP IRREGULAR HOURS': 'Irregular Hours',
    #                             'GP RISK CATEGORY': 'Risk Category', 'VIOL REPEAT VIOLATION': 'Repeat Violation'})
    #
    # # Create CSV
    # data.to_csv("data/AllData.csv", index=False)
    #
    # # Sort data
    # data = data.sort_values(["Restaurant ID", "Inspection ID", "Date", "Violation Status", "Violation Code"])
    #
    # # Drop certain features
    # data = data.drop(
    #         ["FACILITY NAME", "FACILITY ADDRESS", "FACILITY CITY", "FACILITY STATE", "FACILITY ZIP", "GP RECORD ID",
    #          "GP FACILITY ID", "GP PROGRAM IDENTIFIER", "DTA RESULT CODE", "VIOL VIOLATION ID",
    #          "VIOL VIOLATION DESCRIPTION_x", "VIOLATION NUMBER ON INSPECTION REPORT", "VIOL VIOLATION DESCRIPTION_y"],
    #         axis=1)
    #
    # # Rearrange columns
    # data = data[
    #     ["Restaurant ID", "District", "Risk Category", "Square Footage", "GP PE", "Irregular Hours", "Inspection ID",
    #      "Date", "DTA Service", "Violation Status", "Violation Code", "Demerit Value", "Violation Type",
    #      "Imminent Health Hazard", "Repeat Violation", "Demerit Total", "Grade"]]
    #
    # # Create CSV
    # data.to_csv("data/CleanData.csv", index=False)

    # Load data
    data = pd.read_csv("data/CleanData.csv")