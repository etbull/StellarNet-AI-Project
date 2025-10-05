"""
This program downloads all the required light flux data using the lightkurve API
- It does this by first determining the list of Kepler IDs (KepID) that it needs
- It then requests this from the API 
- It runs 8 downloads in parallel to speed up the download process
"""
import lightkurve as lk
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

# This function gets all required kepIDs 
def getKepIDs():
    path = r'backend\\data\\koi.csv'
    df = pd.read_csv(path, comment="#")
    ids = df["kepid"]
    print(ids)
    return ids

# This function actually downloads the light flux data
def download_one(kepid):
    try:
        search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler")
        return search.download_all()
    except Exception as e:
        print(f"Failed {kepid}: {e}")
        return None

def main():
    print('Program started...')
    ids = getKepIDs() # Returns list of kepIDs
    print('KepIDs stored...')
    ids = ids[:8]  # DEBUGGING

    # Downloading all id's
    for i in ids:
        print(f'Current ID = {i}')
        startTime = time.perf_counter()
        download_one(i)
        endTime = time.perf_counter()
        print(f'Time to download {i}: {round(endTime-startTime,2)}')

if __name__ == "__main__":
    main()