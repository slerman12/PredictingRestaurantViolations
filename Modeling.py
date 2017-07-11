import pandas as pd
from sklearn import preprocessing

# Main method
if __name__ == "__main__":
    # Las Vegas 3 Year Inspections
    # all_restaurant_data = pd.read_csv("data/Las Vegas 3 Year Inspections.csv", encoding="ISO-8859-1")

    # Violation codes cheat sheat data
    # cheat_sheet = pd.read_csv("data/Cheat Sheet.csv")

    # New data set to hold all provided information regarding restaurant violations
    # data = all_restaurant_data.merge(cheat_sheet, on="VIOL VIOLATION CODE")

    # Convert column to date time format
    # data["DTA ACTIVITY DATE"] = pd.to_datetime(data["DTA ACTIVITY DATE"])

    # Sort by DTA Activity Date
    # data = data.sort_values(by="DTA ACTIVITY DATE")

    # data = data.groupby("GP FACILITY ID").agg(
    #         {"DTA ACTIVITY DATE": lambda x: (x.diff() > pd.Timedelta(days=15)).cumsum()})

    # For each unique restaurant
    # for restaurant in data["GP FACILITY ID"].unique():
    #     # If the difference between two inspection dates is greater than 15 days, mark as unique inspection
    #     data.loc[data["GP FACILITY ID"] == restaurant, "INSPECTION ID"] = restaurant + ", " + (
    #         data.loc[data["GP FACILITY ID"] == restaurant, "DTA ACTIVITY DATE"].diff() > pd.Timedelta(
    #             days=15)).cumsum().astype(str)

    # print(data.head())

    # data.sort_values(by="GP FACILITY ID").to_csv("test3.csv", index=False)

    # Load merged data
    # data = pd.read_csv("data/Merged Data.csv")

    # Null inspections catch
    # data.loc[pd.isnull(data["INSPECTION ID"]), "INSPECTION ID"] = "NaN"

    # Numerical encode unique inspections
    # data["INSPECTION ID"] = preprocessing.LabelEncoder().fit_transform(data["INSPECTION ID"])

    # Sort data
    # data = data.sort_values(["GP FACILITY ID", "INSPECTION ID", "VIOL VIOLATION STATUS", "VIOL VIOLATION CODE"])

    # Make CSV
    # data.to_csv("data/AllData.csv", index=False)

    # Load data
    # data = pd.read_csv("data/AllData.csv", encoding="ISO-8859-1")

    # Add Grade column
    # data.loc[data["DTA RESULT CODE"] == 31, "GRADE"] = "A"
    # data.loc[data["DTA RESULT CODE"] == 35, "GRADE"] = "B"
    # data.loc[data["DTA RESULT CODE"] == 30, "GRADE"] = "C"
    # data.loc[(data["DTA RESULT CODE"] == 86) | (data["DTA RESULT CODE"] == 87), "GRADE"] = "Closure"

    # Make CSV
    # data.to_csv("data/AllData.csv", index=False)

    # Load data
    # data = pd.read_csv("data/AllData.csv", dtype={"GP RISK CATEGORY": str})

    # Fix risk categories
    # data.loc[data["GP RISK CATEGORY"] == '1-Jan', "GP RISK CATEGORY"] = 0
    # data.loc[data["GP RISK CATEGORY"] == '2-Jan', "GP RISK CATEGORY"] = 1
    # data.loc[data["GP RISK CATEGORY"] == '3-Jan', "GP RISK CATEGORY"] = 2
    # data.loc[data["GP RISK CATEGORY"] == '2', "GP RISK CATEGORY"] = 3
    # data.loc[data["GP RISK CATEGORY"] == '3', "GP RISK CATEGORY"] = 4
    # data.loc[data["GP RISK CATEGORY"] == '1-Apr', "GP RISK CATEGORY"] = 5
    # data.loc[data["GP RISK CATEGORY"] == '2-Apr', "GP RISK CATEGORY"] = 6
    # data.loc[data["GP RISK CATEGORY"] == '3-Apr', "GP RISK CATEGORY"] = 7

    # Create CSV
    # data.to_csv("data/AllData.csv", index=False)

    # Load data
    # data = pd.read_csv("data/CleanData.csv")

    # Initialize imminent health hazard feature
    # data["Imminent Health Hazard"] = 0

    # Set imminent health hazard feature
    # data.loc[data["VIOLATION NUMBER ON INSPECTION REPORT"] == "Imminent Health Hazard", "Imminent Health Hazard"] = 1

    # Create CSV
    # data.to_csv("data/CleanData.csv", index=False)

    # Load data
    # data = pd.read_csv("data/CleanData.csv")

    # Numerically encode repeat violations
    # data.loc[data["Repeat Violation"] == "Y", "Repeat Violation"] = 1
    # data.loc[data["Repeat Violation"] == "N", "Repeat Violation"] = 0

    # Numerically encode irregular Hours
    # data.loc[data["Irregular Hours"] == "Y", "Irregular Hours"] = 1
    # data.loc[data["Irregular Hours"] == "N", "Irregular Hours"] = 0

    # Create CSV
    # data.to_csv("data/CleanData.csv", index=False)

    # Load data
    data = pd.read_csv("data/CleanData.csv")


