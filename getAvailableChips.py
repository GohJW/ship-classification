# Compiles all csv files in the given folder, filters out missing chips not in the image folder and returns a new csv to run the classification model on. Since cropping
# may sometimes stop unexpectedly, this helps to filter out images that were not accounted for so that the classification model does not try to retrieve an image that does
# not exist in the image folder. Run this to generate a new csv to pass to shipclassification.py for model training. 
# Uncomment cropMissingChips to retry the cropping of missing chips.

import pandas as pd
import os
from pathlib import Path
from cropChip import getNearestShip
import rasterio
import numpy as np
import ee
import argparse
from cropChip import process_chipname
# Authentication and connection to google earth engine needed
try:
    ee.Initialize(project='ee-sal-project')
except:
    ee.Authenticate(force=True,  auth_mode='notebook')
    ee.Initialize(project='ee-sal-project')

def filterAvailableChips(compiledcsv, missingcsv, chipfolder, availablecsv):
    """Filters chips available for ship classification based on the current images available in the image folder and returns a csv containing chips to run the model classification

    Args:
        compiledcsv (string): the csv containing the full list of entries from all csvs
        missingcsv (string): the csv containing entries with missing images
        chipfolder (string): the image folder to check for chip images
        availablecsv (string): the output csv containing available chips
    """
    compiled = pd.read_csv(compiledcsv)
    missing = pd.read_csv(missingcsv)
    
    #for dual channel
    compiled['chipname'] = compiled['chipname'].apply(process_chipname)
    
    available = compiled[~compiled['chipname'].isin(missing['chipname'])]
    for chip in available['chipname']:
        src = rasterio.open(os.path.join(chipfolder,chip))
        image = src.read(1)
        if np.isnan(image).all():
            index = available[available['chipname'] == chip].index
            print(index)
            available = available.drop(index)
            
    available.to_csv(availablecsv, index = False)

def findMissingChips(compiledcsv, chipfolder, missingcsv):
    """Finds all missing chips. From the csv file, each entry has a chipname that will be looked for in the image folder. If it does not exist, the image is missing and
    the entry cannot be used to train the model.

    Args:
        compiledcsv (string): the csv containing the full list of entries from all csvs
        chipfolder (string): the image folder to check for chip iamges
        missingcsv (string): the output csv containing missing chips
    """
    df = pd.read_csv(compiledcsv)
    chipfolder = Path(chipfolder)
    allchips = [os.path.basename(chip) for chip in chipfolder.glob('**/*.tif')] 
    
    #for dual channel
    df['chipname'] = df['chipname'].apply(process_chipname)
    
    missing_df = df[~df['chipname'].isin(allchips)]
    missing_df.to_csv(missingcsv, index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compiles available and missing chips into seperate csv files')
    parser.add_argument('--csvfolder', type=str, help='folder containing processed csv files')
    parser.add_argument('--imagefolder', type=str, help = 'folder containing image chips')
    parser.add_argument('--outputfolder', type=str, help = 'outputfolder for output csvs')
    parser.add_argument('--percentage', type=int, help='percentage waterbody criteria')
    parser.add_argument('--timediff', type=int, help='time difference criteria')
    parser.add_argument('--size', type=int, help='chip size')
    args = parser.parse_args()
    if not os.path.exists(args.outputfolder):
        os.mkdir(args.outputfolder)
    compiledpath = os.path.join(args.outputfolder, 'compiled.csv')
    missingpath = os.path.join(args.outputfolder, 'missingchips.csv')
    availablepath = os.path.join(args.outputfolder, 'availablechips.csv')

    
    folder = Path(args.csvfolder)
    chipfolder = args.imagefolder
    files = [filename for filename in folder.glob('**/*.csv') if os.path.basename(filename) != 'missing_MMSI.csv']
    completeddfs = []
    for file in files:
        df = pd.read_csv(file)
        df = df[df['waterbody percentage'] >= args.percentage]
        df = df[df['timediff'] <= args.timediff]
        df = getNearestShip(df)
        df = df[df['nearest ship'] > args.size]
        completeddfs.append(df)
    bigdf = pd.concat(completeddfs)
   
    pd.DataFrame.to_csv(bigdf,compiledpath, index=False)
    findMissingChips(compiledpath, chipfolder, missingpath)
    filterAvailableChips(compiledpath, missingpath, chipfolder, availablepath)
