# Filters through a csv file containing AISdata of ships, and looks for the ship location within the SAR image dataset. If an image is found containing the location of the ship,
# saves the AISdata of that particular ship into a new csv file for collection of metadata.
from pathlib import Path
import ee
import pyproj
import os
import pandas as pd
import time
import datetime
from  dateutil import parser
import csv
import argparse
import concurrent.futures
from multiprocessing import Pool
from itertools import repeat
from pathlib import Path
from ee import Geometry, ImageCollection
# Authentication and connection to google earth engine needed
try:
    ee.Initialize(project='ee-sal-project')
except:
    ee.Authenticate(force=True,  auth_mode='notebook')
    ee.Initialize(project='ee-sal-project')

def timedelta_to_hours_minutes(timedelta_obj):
    # Get total seconds
    total_seconds = timedelta_obj.total_seconds()

    # Calculate hours and minutes
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return hours, minutes

def getBoundingBox(lon, lat, size, crs, resolution_meters):
    lonlat_utm = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy= True)
    utm_lonlat = pyproj.Transformer.from_crs(crs,'EPSG:4326', always_xy= True)
    # bl_heading = 225
    # tr_heading = 45
    # print(lon, lat)
    utmx, utmy = lonlat_utm.transform(lon, lat)
    # print(utmx, utmy)
    distance = size/resolution_meters
    utmx_bl = utmx - distance*resolution_meters
    utmy_bl = utmy - distance*resolution_meters
    
    utmx_tr = utmx + distance*resolution_meters
    utmy_tr = utmy + distance*resolution_meters
    
    lon_bl, lat_bl = utm_lonlat.transform(utmx_bl, utmy_bl)
    lon_tr, lat_tr = utm_lonlat.transform(utmx_tr, utmy_tr)
    # print(lon_bl, lat_bl, lon_tr, lat_tr)
    return Geometry.BBox(lon_bl, lat_bl, lon_tr, lat_tr)
    
    
def getDataset(filename, outputfolder, workers):
    """Filters through the AIS csv file, discarding entries for ships with a speed over ground 'SOG' >=0.1. Then filters by each unique ship 'MMSI', taking the first entry and
    searching for a SAR image at that latitude and longitude location. If an image exists, calculate the timediff between that SAR image and all entries of the ship with that
    unique 'MMSI'. Subsequently sorts entries of that MMSI by shortest timediff, checking again if the latitude and longitude of the shortest timediff entry is contained within
    the boundaries of the SAR image, before saving the entry into a new csv file for metadata processing. 
    This is all done to minimise the time difference and distance the ship
    moves between the time of the SAR image taken and the time of AIS data reported by the ship to ensure maximum chance of the ship appearing within the image chip.
    Args:
        filename (string): the path to the AIS csv file
        outputfolder (string): the output folder to store the csv files
    """
    print(f'scanning {filename}')
    outputfolder = Path(outputfolder)
    outputfolder.mkdir(exist_ok=True, parents=True)

    date = datetime.date.fromisoformat(os.path.basename(filename).split('.', 1)[0].split('_', 1)[1].replace('_', '-'))

    if (outputfolder / (str(date)+'.csv')).exists():
        return

    file = pd.read_csv(filename)
    print(f'load {filename}')
    count = 0
    print(f'date {filename}')

    file_SOG = file.loc[file['SOG'] < 0.1].copy()
    del file
    # file.query(file['SOG'] < 0.1, inplace=True)
    # file_SOG = file

    print(f'query {filename}')
    file_SOG['BaseDateTime'] = pd.to_datetime(file_SOG['BaseDateTime'])
    print(f'loop {filename}')


    file_SOG_MMSI = [file_SOG[file_SOG['MMSI'] == mmsi] for mmsi in file_SOG['MMSI'].unique()]
    start_time = time.time()    

    pool = Pool(workers)
    results = pool.starmap(process_mmsi, zip(file_SOG_MMSI, repeat(date), range(len(file_SOG_MMSI)), repeat(len(file_SOG_MMSI))))

    # print(results)

    valid_results = [result for result in results if isinstance(result, pd.DataFrame)]
    valid_results = pd.concat(valid_results)

    missing_mmsi = [result for result in results if isinstance(result, str)]
    missing_mmsi = pd.DataFrame({'MMSI': missing_mmsi})


    pd.DataFrame.to_csv(valid_results, outputfolder / (str(date)+'.csv'),  header=True, index= False)

    if len(missing_mmsi) > 0:
        pd.DataFrame.to_csv(missing_mmsi, outputfolder / 'missing_MMSI.csv', header=False, index= False)


    # processes = []
    # try:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers= args.threads) as executor:   
    #         for mmsi in file_SOG['MMSI'].unique():
    #             file_SOG_MMSI = file_SOG[file_SOG['MMSI'] == mmsi].copy()
    #             processes.append(executor.submit(getDataset, file_SOG_MMSI, date))
    # except KeyboardInterrupt:
    #         executor.shutdown(wait=False)

        
    print(f'Total images found: {len(valid_results)} of {len(file_SOG["MMSI"].unique())}')        
    end_time = time.time()
    print((end_time - start_time) / 60)


def process_mmsi(file_SOG_MMSI, date, index, max_index):
    index = file_SOG_MMSI.index[0] #get the index of the first row
    lat = file_SOG_MMSI.loc[index, 'LAT']
    lon = file_SOG_MMSI.loc[index, 'LON']
    try:
        imagecollection = (ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate(str(date), str(date + datetime.timedelta(days= 1)) )
                    .filterBounds(Geometry.Point(lon, lat))
                    )
        metadata = imagecollection.getInfo()
        size = imagecollection.size().getInfo()
        if(size != 0):
            #calc time diff
            imagename = metadata['features'][0]['properties']['system:index']
            splitname = imagename.split('_')
            # print(splitname[4])
            starttime = datetime.datetime.strptime(splitname[4], "%Y%m%dT%H%M%S") #starttime of the SAR image
            file_SOG_MMSI['timediff'] = abs((file_SOG_MMSI['BaseDateTime'] - starttime)).dt.total_seconds() #get timediff for all entries with matching MMSI
            file_SOG_MMSI = file_SOG_MMSI.sort_values('timediff').reset_index()  #sort by timediff asc and reset index
            imageboundaries = imagecollection.first().select(0).geometry()
            
            if not imageboundaries.contains(ee.Geometry.Point(file_SOG_MMSI.loc[0, 'LON'], file_SOG_MMSI.loc[0, 'LAT'])).getInfo(): 
            #check if lat lon of smallest timediff row is within image, if not discard
                return None
            print(f'{str(date)}: found image for MMSI: {file_SOG_MMSI.at[0,"MMSI"]}, {index}/{max_index}')
            #store data
            file_SOG_MMSI.loc[0, "image name"] = imagename
            file_SOG_MMSI.loc[0, "image polarisation"] = str(metadata['features'][0]['properties']['transmitterReceiverPolarisation'])
            file_SOG_MMSI.loc[0, "image resolution"] = metadata['features'][0]['properties']['resolution_meters']
            projection = imagecollection.first().select(0).projection().getInfo()
            file_SOG_MMSI.loc[0, "crs"] = projection['crs']
            #append smallest timediff row to new csv file
            return file_SOG_MMSI.iloc[0:1,:].copy()
        
    except Exception as e:
        print(e)
        return file_SOG_MMSI.at[0,'MMSI']

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description ="Searches rows of csv file and looks for valid SAR images within Earth Engine API for a given time range")
    parser.add_argument('--folder', type=str, help = "The path to the folder of csvs")
    parser.add_argument('--glob_str', type= str, default='AIS*.csv', help='glob to filter csv. Eg AIS_2021_01*.csv')
    parser.add_argument('--threads', type= int, help='number of concurrent threads to run')
    parser.add_argument('--output', type=str, help= 'The output folder')
    args = parser.parse_args()
    # print(args.filename.split('.')[0])
    # getDataset(args.filename, args.filename.split('.')[0])
    csvfolder = Path(args.folder)
    files = sorted([filename for filename in csvfolder.glob(args.glob_str)])
    output_folders = [Path(args.output) / Path(filename).stem for filename in files]

    # pool = Pool(args.threads)
    # pool.starmap(getDataset, zip(files, output_folders))

    [getDataset(filepath, output_folder, args.threads) for filepath, output_folder in zip(files, output_folders)]

    # try:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers= args.threads) as executor:   
    #         for filename in files:
    #             # print(str(filename).split('.')[0])
    #             executor.submit(getDataset, filename, os.path.join(args.output, os.path.basename(str(filename)).split('.')[0]))
    # except KeyboardInterrupt:
    #         executor.shutdown(wait=False)