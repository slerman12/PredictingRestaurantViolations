import pandas as pd

# Main method
if __name__ == "__main__":
    all_restaurant_data = pd.read_csv("data/Las Vegas 3 Year Inspections.csv", encoding="ISO-8859-1")
    print(all_restaurant_data.head(n=5))

