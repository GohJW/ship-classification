import pandas as pd
import ee
from getData import getBoundingBox
import argparse

def cropMissingChips(missingcsv):
    """Crops chips from the missing chips csv. Does not guarantee that all chips within will be cropped. Some chip images are unable to be cropped due to EarthEngineAPI with
    image boundary issues, searching for boundaries of image may return true, but the actual location contains no pixels.

    Args:
        missingcsv (string): the csv containing entries with missing images
    """
    file = pd.read_csv(missingcsv)
    for index in file.index:
            print(f"cropping {file.loc[index, 'chipname']}")
            image = ee.Image('COPERNICUS/S1_GRD/' + file.loc[index,'image name']).select(['VV', 'VH'])
            task = ee.batch.Export.image.toDrive(
                image = image,
                description= f"{file.loc[index, 'chipname'].split('.')[0]}",
                folder= 'dso_internship',
                region= getBoundingBox(file.loc[index, 'LON'],file.loc[index, 'LAT'], 700, file.loc[index, 'crs'], file.loc[index, 'image resolution'])
                )
            task.start()
    print(f'{len(list(file.index))} images cropped')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retry cropping of missing chips from the missing chips csv')
    parser.add_argument('--csv', type=str, help= 'path of csv containing missing chips')
    args = parser.parse_args()
    cropMissingChips(args.csv) #uncomment to retry cropping of missing chips