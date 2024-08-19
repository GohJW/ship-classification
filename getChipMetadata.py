# Used to calculate and append 'waterbody percentage' column to the Dataframe before re-saving the csv. Do this only after getData.py has finished collecting all entries in
# an AISdata csv file. Run this script from command line interface to make use of multithreading.
import os, sys
import pandas as pd
import argparse
import ee
import concurrent.futures
from pathlib import Path
sys.path.append('.')
from getData import getBoundingBox

# Authentication and connection to google earth engine needed
try:
    ee.Initialize(project='ee-sal-project')
except:
    ee.Authenticate(force=True,  auth_mode='notebook')
    ee.Initialize(project='ee-sal-project')

def getChipMetadata(filename, imagesize, terrainmap):
    """Calculates and appends the percentage waterbody of the image based on the chipsize specified. Also appends the 'date' and 'chipname' columns for ease of use in subsequent
    model trainings. Uses terrain data from ESA/WorldCover/v100 in EarthEngineAPI.

    Args:
        filename (string): the csv file to get metadata
        imagesize (int): the size of the chip to calculate waterbody percentage
        terrainmap (EarthEngineObject): the EarthEngineObject containing the terrain map
    """
    print(f'Scanning {filename}')
    file = pd.read_csv(filename, index_col= [0])
    
    if 'waterbody percentage' not in list(file.columns): 
        print(f'Calculating Water Percentages for {filename}')          
        for index in file.index:
            # print(index)
            # print('calculating water body percentage for index: ', index)
            
            #find water body percentage
            region = getBoundingBox(file.loc[index, 'LON'], file.loc[index, 'LAT'], imagesize, file.loc[index, 'crs'], file.loc[index, 'image resolution'])
            terrain = terrainmap.clip(region)
            
            terrainarea = terrain.eq([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]).reduceRegion( #select all terrain types
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=100,
                maxPixels=1e9
            )
            totalarea = sum(ee.Number(terrainarea).getInfo().values())
            
            watercover = terrain.eq(80) #select only Permanent water bodies
            watercover = watercover.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=100,
                maxPixels=1e9
            )
            waterarea = sum(ee.Number(watercover).getInfo().values())
            file.loc[index, 'waterbody percentage'] = 100 if totalarea == 0 else waterarea/totalarea*100
            
    if 'date' not in list(file.columns):
        date = os.path.basename(filename).split('.')[0]
        file['date'] = date
        chipname = date + '-' + file.index.astype(str) + '.tif'
        file['chipname'] = chipname
    pd.DataFrame.to_csv(file, filename)
 
   
if __name__ == '__main__':
        terrainmap = ee.ImageCollection('ESA/WorldCover/v100').first().select(0)
        parser = argparse.ArgumentParser(description ="Calculates water body percentage given the size of the chip for indices with a SAR image found for all csvs in a folder")
        parser.add_argument('--folder', type = str, help = 'the path to the folder of csvs')
        parser.add_argument('--size', type = int, default=700, help = 'the size of the chip to check water body percentage')
        parser.add_argument('--threads', type= int, help= 'number of concurrent threads to run')
        args = parser.parse_args()
        csvfolder = Path(args.folder)
        files = [filename for filename in csvfolder.glob('**/*.csv') if os.path.basename(filename) != 'missing_MMSI.csv']
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers= args.threads) as executor:   
                for filename in files:
                    executor.submit(getChipMetadata, filename, args.size, terrainmap)
        except KeyboardInterrupt:
            executor.shutdown(wait=False)
            sys.exit(1)

