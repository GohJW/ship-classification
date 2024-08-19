# Crops chips based on the given criteria. Use this only after running getChipMetadata.py on the csv file. Edit the glob string in '__main__' if needed and run this script from the
# command line interface to make use of multithreading. Note may stop occasionally even though not all chips have been cropped, use this together with collectMissingChip.py to filter
# actual chips available for model training.
import ee
import pyproj
import os
import pandas as pd
import argparse
from math import isnan  
import concurrent.futures
from pathlib import Path
from getData import getBoundingBox
from scipy.spatial.distance import cdist


# Authentication and connection to google earth engine needed
try:
    ee.Initialize(project='ee-sal-project')
except:
    ee.Authenticate(force=True,  auth_mode='notebook')
    ee.Initialize(project='ee-sal-project')

def process_chipname(chipname):
    """Adds additional suffix to 'image name' column of Dataframe for model to retrieve correct image. Original images are single channel, while dual images contain '_dualchannel'

    Args:
        chipname (string): the name of the chip

    Returns:
        string: the name of the chip with the appended suffix
    """
    parts = chipname.split('.')
    return str(parts[0] + '_dualchannel.tif')

def convertToUTM(lon, lat, crs):
    lonlat_utm = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy= True)
    utmx, utmy = lonlat_utm.transform(lon, lat)
    return utmx, utmy

def getNearestShip(df):
    """Collects all ship entries from each full unique SAR image and calculate their pairwise distances, saving the smallest pairwise distance in the 'nearest ship' column. If no 
    other ship is found within the full SAR image, pairwise distance is returned as 'inf'. Will skip the Dataframe if it already contains a 'nearest ship' column.

    Args:
        df (pd.Dataframe): Dataframe containing entries

    Returns:
        pd.Dataframe: the Dataframe with the appended 'nearest ship' columns calculated.
    """
    if 'nearest ship' not in list(df.columns):
        for imagename in df['image name'].unique():
            df_imagename = df[df['image name'] == imagename]
            if len(df_imagename) == 1:
                df.loc[df_imagename.index, 'nearest ship'] = float('inf')
                continue    
            else:
                crs = df_imagename.loc[df_imagename.index[0], 'crs']
                df_imagename['x'], df_imagename['y'] = convertToUTM(df_imagename['LON'].values, df_imagename['LAT'].values, crs)
                distances = cdist(df_imagename[['x', 'y']], df_imagename[['x', 'y']])
                # print(distances)
                distances[range(len(distances)), range(len(distances))] = float('inf')
                df_imagename['nearest ship'] = distances.min(axis = 1)
                df.loc[df_imagename.index, 'nearest ship'] = df_imagename['nearest ship']
    return df
    
def cropChip(filename, percentage, timediff, size):
    """Filters the csv for the specific percentage waterbody and time difference, before cropping the image from the EarthEngineAPI and saving it to googledrive.

    Args:
        filename (string): the name of the csv file to crop chips of
        percentage (int): minimum waterbody percentage
        timediff (int): maximum time difference between AIS ship time and SAR image time
        size (int): width/height of chip to crop
    """
    print(f"cropping chips for {filename}")
    file = pd.read_csv(filename)
    file_water = file[file['waterbody percentage'] >= percentage]
    file_water_timediff = file_water[file_water['timediff'] <= timediff]
    file_water_timediff = getNearestShip(file_water_timediff)
    
    #for dual channel
    file_water_timediff['chipname'] = file_water_timediff['chipname'].apply(process_chipname)
    
    for index in file_water_timediff.index:
        nearestShip = file_water_timediff.loc[index, 'nearest ship']
        if nearestShip <= size:
            continue
        # print(f"cropping {file_water_timediff.loc[index, 'chipname']}")
        image = ee.Image('COPERNICUS/S1_GRD/' + file_water_timediff.loc[index,'image name']).select(['VV', 'VH']) #select channels
        task = ee.batch.Export.image.toDrive(
            image = image,
            description= f"{file_water_timediff.loc[index, 'chipname'].split('.')[0]}",
            folder= 'dso_internship',
            region= getBoundingBox(file_water_timediff.loc[index, 'LON'],file_water_timediff.loc[index, 'LAT'], size, file_water_timediff.loc[index, 'crs'], file_water_timediff.loc[index, 'image resolution'])
            )
        task.start()
    print(f'{len(list(file_water_timediff.index))} images cropped')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ="Crops a chip of specified size for each ship in the csv that matches the waterbody percentage and time difference criteria")
    parser.add_argument('--folder', type=str, help = "The path to the folder of csvs")
    parser.add_argument('--percentage', type= float, default=100 , help='percentage waterbody criteria')
    parser.add_argument('--timediff', type= int, default=1000,  help='the maximum time difference criteria')
    parser.add_argument('--size', type= int, default=700, help='size of the chip to crop in metres')
    parser.add_argument('--threads', type= int, help='number of concurrent threads to run')
    parser.add_argument('--glob_str', type= str, default='**/*.csv', help='number of concurrent threads to run')

    args = parser.parse_args()
    # getDataset(args.filename, args.filename.split('.')[0])
    csvfolder = Path(args.folder)
    files = [filename for filename in csvfolder.glob(args.glob_str) if os.path.basename(filename) != 'missing_MMSI.csv']
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers= args.threads) as executor:   
            for filename in files:
                executor.submit(cropChip, filename, args.percentage, args.timediff, args.size)
    except KeyboardInterrupt:
            executor.shutdown(wait=False)