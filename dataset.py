# import os
# import logging
# logger1 = logging.getLogger('logger1')
# logger1.setLevel(logging.INFO)
# file_handler1 = logging.FileHandler('dataset.txt')
# file_handler1.setFormatter(logging.Formatter('%(message)s'))
# logger1.addHandler(file_handler1)
# logger1.info(f"No of netCDF4 files scraped from the web: {len(os.listdir('Satellite Imagery'))}")
# logger1.info(f"No of netCDF4 files that corresponds to number of images are:{len(os.listdir('Filtered_satellite_imagery'))}")
# print("****")
import os
import logging

# Configure logger1
logger1 = logging.getLogger('logger1')
logger1.setLevel(logging.INFO)
file_handler1 = logging.FileHandler('dataset.log')
file_handler1.setFormatter(logging.Formatter('%(message)s'))
logger1.addHandler(file_handler1)

# Log the number of netCDF4 files scraped from the web
num_scraped_files = len(os.listdir('Satellite Imagery'))
logger1.info(f"No of netCDF4 files scraped from the web: {num_scraped_files}")

# Log the number of netCDF4 files that correspond to the number of images
num_filtered_files = len(os.listdir('Filtered_satellite_imagery'))
logger1.info(f"No of netCDF4 files that correspond to number of images: {num_filtered_files}")

print("****")
