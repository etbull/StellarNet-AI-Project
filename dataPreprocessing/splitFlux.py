"""
This program splits the amature data into testing and training
"""
import pandas as pd

def main():
    path = r"backend\\data\\amature.csv"
    df = pd.read_csv(path)

    # Splitting data
    training = df.iloc[:30]
    testing = df.iloc[30:]
        
    # Saving Data
    training.to_csv(r"backend\\data\\amaTrain.csv", index=False)
    testing.to_csv(r"backend\\data\\amaTest.csv", index=False)
    print('Data saved')

if __name__ == "__main__":
    main()