# Use this to quickly filter and remove single channel images when downloading from Google Drive
from pathlib import Path
import os
# Define the directory containing the files
directory = 'dso_internship'

# Define the string that should be present in the files
target_string = 'dualchannel'

# List all files in the directory
folder = Path(directory)
imagenames = [str(os.path.basename(image)) for image in folder.glob("*dual*")]
print(imagenames, len(imagenames))
for file in folder.iterdir():
    if str(os.path.basename(file)) not in imagenames:
        print(str(os.path.basename(file)))
        os.remove(file)