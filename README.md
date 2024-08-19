# Data Collection, Cropping and Classification with Earth Engine
This program uses Automatic Identification System (AIS) data in conjunction with Google Earth Engine API to collect ship metadata and images for model classification.
## Data Collection
Data collection is done with `getData.py` and `getChipMetadata.py`.
### getData
`python getData.py --folder [folder path containing csv files] --output [output folder path] --threads [number of concurrent threads to run]`\
getData looks for all csv files within the folder path matching `AIS*.csv`. For each csv file, we filter for SOG < 0.1, before looping through each unique MMSI, taking the first
entry and looking for a SAR image at the latitude and longitude location. If a SAR image is found, we calculate the time difference between the image time taken all entries with
that unique MMSI. If the entry with the shortest time difference is within the image boundaries, we save the entry in a seperate csv file with the image name, polarisations, resolution and crs. If the MMSI returns any error, it is saved in `missing_MMSI.csv`.
> NOTE: The SOG criteria can be modified or removed from the script. This script uses the Sentinel-1 GRD C-band dataset. It can also be modified to work with other datasets, but ensure the band names and image property names match. See `https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD#description`.
### getChipMetadata
`python getChipMetadata.py --folder [folder path containing csv files of filtered entries] --size [size of chip to calculate waterbody percentage] --threads [number of concurrent threads
to run]`\
getChipdata use the ESA world cover map to calculate the percentage waterbody of each entry based on the specified size. It also appends the date and image chip name for ease of use when performing cropping and model training. This additional data is appended to the csv file itself.
## Cropping
With image collection done, we can start cropping our image chips.
> NOTE: Cropping should be done after running `getChipMetadata.py`.

`python cropChip.py --folder [folder path containing processed csv files] --percentage [percentage waterbody criteria] --timediff [time difference criteria] --size [crop size]
--threads [number of concurrent threads to run]`\
cropChip filters for entries that match the required waterbody percentage and time difference criteria. It then calculates the pairwise distance between all ships within the same
SAR image, and discard ships with their shortest pairwise distance less than the crop size, before cropping and saving the chip to Google Drive.
> NOTE: This script currently set to work with dual channel images. This can be changed within the script by editing the cropChip function to select the desired bands. Remove the `_dualchannel` suffix using `process_chipname` function.

## Classification
With the cropped chips downloaded, we can perfomr model classification. This is done with `getAvailableChips.py` and `ship_classification.py`
### getAvailableChips
`python getAvailableChips.py --csvfolder [folder containing processed csvs] --imagefolder [folder containing downloaded chip images] --outputfolder [output folder for filtered
available and missing chips csvs] --percentage [waterbody percentage criteria] --timediff [time difference criteria] --size [crop size]`

As the Earth Engine cropping of images may run into unexpected cropping errors, some images within the csv files may not have been cropped. We use `getAvailableChips.py` to 
check the image folder for chip names of successfully cropped and downloaded chips. The processed csv files are combined into one big csv `compiled.csv`. With `findMissingChips` function, we compare the chipname column of the compiled csv with the images within our saved images folder, saving all missing chip images in `missingchips.csv`.
> NOTE: Similar to `cropChip.py`, this script is also currently set to work with the `_dualchannel` suffix, and can be modified using the `process_chipname` function.

> NOTE: Use the same waterbody percentage, time difference and image size as the original values to process and crop the images.

With `filterAvailableChips` function, we take the images that are present within our saved images folder and save them in `availablechips.csv` to perform our image classification and model training with `ship_classification.py`.
### ship_classification
The data in `availablechips.csv` is split into train and test sets, and saved as `train.csv` and `test.csv` for ease of use when performing gradcam analysis.\
Under `__main__` is a bunch of parameters that can be used to finetune and modify our model parameters. Every 10 epoch runs, we run a validation test and log our findings within tensorboard. Use `tensorboard --logdir=[log directory]` to access the tensorboard logs. If the performance of the model is better than its previous validation, the confusion matrix and model parameters are saved, overwriting the previous best if any. 

> NOTE: The model's classification labels vary with the size of the dataset. Currently, it is set to have a minimum criteria of 100 examples BEFORE train test split in order to consider that VesselType as a label.

> NOTE: The name of the matrix model and log is saved based on the current model configurations and may overwrite previous iterations.

> NOTE: Certain parameters, such as t_max_value, lr_patience and lr_reduction_factor are dependent on the scheduler used and would not influence the model if a different or no scheduler is used.

## Other functions
### gradcam
the `gradcam.py` script within the `traintestvalidation` folder can be used visualise what the model focuses on when performing its label prediction. Load the desired model by changing the model and model checkpoint path, image folder, output folder and train and test csv files.
> NOTE: The dataset used should match the dataset the model is trained on as the classification labels of the model are dependant on the dataset size.
### cropMissingChip
`python cropMissingChip.py --csv [missing chips csv path]`
`cropMissingChip.py` can be used to retry cropping of missing chips from the `missingchips.csv` file generated from `getAvailableChips.py`. Once the new images have been downloaded, rerun `getAvailableChips.py` to get the newly filtered `availablechips.csv` for model training.
### dataset examples
`11kdatasetdualchannel` and `12kdatasetdualchannel` contains examples of the full data collection, image cropping and model training process. Both datasets use a waterbody percentage criteria of 100%, time difference of 1000 seconds and crop size of 700m. the 12k dataset includes all data from `datasets/completed` while the 11k dataset excludes data from June. \
`python getAvailableChips.py --csvfolder datasets/completed --imagefolder dso_internship --outputfolder 12kdatasetdualchannel --percentage 100 --timediff 1000 --size 700`

