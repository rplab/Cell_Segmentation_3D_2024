# %% [markdown]
# # Segmentation and Region Properties Extraction for Enteric Neurons and Macrophages
#
# **Author(s):** Piyush Amitabh
#
# **Description:**
# This script processes downsampled images to segment Enteric Neurons and macrophages, and extracts their region properties. The extracted properties are saved to CSV files. The script is designed to be OS-agnostic and should work on both Windows and Linux as long as the original path is provided correctly.
#
# **Created:** March 31, 2022
#
# **Updated:**
# - **Dec 06, 2022:** Batch processes downsampled images and saves their region properties. Updated regionprops list for skimage-v0.19.2.
# - **Feb 15, 2023 (v9):** Made code OS-agnostic.
# - **Mar 23, 2023:** Extended for macrophages.
# - **Mar 23, 2023:** Extended for ENS.
#
# **License:**
# GNU GPL v3.0
#
# **Usage Instructions:**
# 1. Ensure that images are sorted into different directories by channel (BF/GFP/RFP). Use the `sort_img_by_channels.py` script to sort images before running this script.
# 2. Run the script and provide the main directory containing all images to be segmented when prompted.
# 3. The script will segment the images and save the region properties to CSV files in the respective directories.
#
# **Dependencies:**
# - numpy
# - pandas
# - matplotlib
# - scipy
# - skimage
# - tifffile
# - os
#
# **Functions:**
# - `get_info_table_ens(img_path)`: Reads images and returns segmented object labels and region properties table for ENS.
# - `filter_info_table_ens(labels, info_table)`: Filters the region properties table for ENS based on object properties.
# - `get_info_table_macs(img_path)`: Reads images and returns segmented object labels and region properties table for macrophages.
# - `filter_info_table_macs(labels, info_table)`: Filters the region properties table for macrophages based on object properties.
# - `find_surface_area(filt_labels, info_table_filt)`: Calculates the surface area of objects and adds it to the region properties table.
# - `segment_save_props_ds(img_path, save_path, csv_name)`: Segments images, extracts region properties, and saves them to CSV files.
#
# **Contact:**
# For any questions or issues, please contact Piyush Amitabh.
#
# ---
# %% [markdown]
# ---

import os

import numpy as np
import pandas as pd
import skimage
import tifffile as tiff
from scipy import ndimage as ndi

# %% [markdown]
# The pixel spacing in this dataset is 1µm in the z (leading!) axis, and  0.1625µm in the x and y axes.
n = 4  # downscaling factor used in x and y
zd, xd, yd = 1, 0.1625, 0.1625  # zeroth dimension is z
orig_spacing = np.array([zd, xd, yd])  # change to the actual pixel spacing from the microscope
new_spacing = np.array([zd, xd * n, yd * n])  # downscale x&y by n

# %%
# list of regionprops updated for skimage-v0.19.2


def get_info_table_ens(img_path):
    """
    Reads images given by img_path, segments enteric neurons and calculates region properties of segmented objects.
    Returns the labelled segmented image and region properties.

    Parameters:
    img_path (str): Path to the image file.

    Returns:
    tuple: A tuple containing:
        - labels (ndarray): Labeled image array where each object is assigned a unique integer.
        - info_table (pd.DataFrame): DataFrame containing region properties of the segmented objects.
    """
    stack_full = tiff.imread(img_path)
    if len(stack_full.shape) != 3:  # return None if not a zstack
        return (None, None)
    print("Reading: " + img_path)

    denoised = ndi.median_filter(stack_full, size=3)
    mean = np.mean(stack_full)  # tried median, very close values doesn't matter
    std = np.std(stack_full)
    thresh_val = mean + 10 * std
    simple_thresh = denoised > thresh_val  # based on image values

    labels = skimage.measure.label(simple_thresh)  # labels is a bool image
    info_table = pd.DataFrame(
        skimage.measure.regionprops_table(
            labels,
            intensity_image=stack_full,
            properties=[
                "label",
                "centroid",
                "centroid_weighted",
                "area",
                "equivalent_diameter_area",
                "intensity_mean",
            ],  # , #'slice','moments_normalized', 'coords'
        )
    ).set_index("label")
    return (labels, info_table)


# %%
def filter_info_table_ens(labels, info_table):
    """
    Filters the region properties table obtained by get_info_table_ens according to volume limits defined below.

    Parameters:
    labels (ndarray): Labeled image array where each object is assigned a unique integer.
    info_table (pd.DataFrame): DataFrame containing region properties of the segmented objects.

    Returns:
    tuple: A tuple containing:
        - filt_labels (ndarray): Labeled image array after filtering.
        - info_table_filt (pd.DataFrame): Filtered DataFrame containing region properties of the segmented objects.
    """
    voxel_width_min = (10**3) / 6  # 1000/2 -> 26.5/2 um^3 #as main cell body of ens can be smaller
    voxel_width_max = 10**5  # for ens can be around 10**4 so take an order of magnitude bigger

    info_table_filt = info_table[
        np.logical_and(info_table.area > voxel_width_min, info_table.area < voxel_width_max)
    ].copy()

    bad_label = list(set.difference(set(info_table.index), set(info_table_filt.index)))
    filt_labels = labels.copy()

    for i in bad_label:
        filt_labels[np.where(labels == i)] = 0

    return (filt_labels, info_table_filt)


def get_info_table_macs(img_path):
    """
    Reads images given by img_path, segments macrophages and calculates region properties of segmented objects.
    Returns the labelled segmented image and region properties.

    Parameters:
    img_path (str): Path to the image file.

    Returns:
    tuple: A tuple containing:
        - labels (ndarray): Labeled image array where each object is assigned a unique integer.
        - info_table (pd.DataFrame): DataFrame containing region properties of the segmented objects.
    """
    stack_full = tiff.imread(img_path)
    if len(stack_full.shape) != 3:  # return None if not a zstack
        return (None, None)
    print("Reading: " + img_path)

    denoised = ndi.median_filter(stack_full, size=3)

    mean = np.mean(stack_full)  # try median
    std = np.std(stack_full)
    thresh_val = mean + 10 * std
    simple_thresh = denoised > thresh_val  # manual value

    dilated_high_thresh = ndi.binary_dilation(simple_thresh, iterations=5)
    skeleton = skimage.morphology.skeletonize(dilated_high_thresh)
    connected_regions = np.logical_or(simple_thresh, skeleton)
    labels = skimage.measure.label(connected_regions)

    info_table = pd.DataFrame(
        skimage.measure.regionprops_table(
            labels,
            intensity_image=stack_full,
            properties=[
                "label",
                "centroid",
                "centroid_weighted",
                "area",
                "equivalent_diameter_area",
                "intensity_mean",
            ],  # , #'slice','moments_normalized', 'coords'
        )
    ).set_index("label")
    return (labels, info_table)


# %%
def filter_info_table_macs(labels, info_table):
    """
    Filters the region properties table obtained by get_info_table_macs according to volume limits defined below.

    Parameters:
    labels (ndarray): Labeled image array where each object is assigned a unique integer.
    info_table (pd.DataFrame): DataFrame containing region properties of the segmented objects.

    Returns:
    tuple: A tuple containing:
        - filt_labels (ndarray): Labeled image array after filtering.
        - info_table_filt (pd.DataFrame): Filtered DataFrame containing region properties of the segmented objects.
    """
    voxel_width_min = (10**3) / 2  # 1000/2 -> 26.5/2 um^3
    voxel_width_max = 10**5

    info_table_filt = info_table[
        np.logical_and(info_table.area > voxel_width_min, info_table.area < voxel_width_max)
    ].copy()

    bad_label = list(set.difference(set(info_table.index), set(info_table_filt.index)))
    filt_labels = labels.copy()

    for i in bad_label:
        filt_labels[np.where(labels == i)] = 0

    return (filt_labels, info_table_filt)


# %%
def find_surface_area(filt_labels, info_table_filt):
    """
    Uses marching cubes to find the surface area of the objects in info_table_filt.
    Adds the computed user_surface_area and sphericity to the info_table_filt.

    Parameters:
    filt_labels (ndarray): Labeled image array after filtering.
    info_table_filt (pd.DataFrame): Filtered DataFrame containing region properties of the segmented objects.

    Returns:
    None
    """
    list_surface_area = []

    for selected_cell in range(len(info_table_filt.index)):
        regionprops = skimage.measure.regionprops(filt_labels.astype("int"))
        volume = (filt_labels == regionprops[selected_cell].label).transpose(1, 2, 0)
        verts, faces, _, values = skimage.measure.marching_cubes(volume, level=0, spacing=(1.0, 1.0, 1.0))
        surface_area_pixels = skimage.measure.mesh_surface_area(verts, faces)
        list_surface_area.append(surface_area_pixels)

    info_table_filt["user_surface_area"] = list_surface_area
    info_table_filt["sphericity"] = (
        np.pi * np.square(info_table_filt["equivalent_diameter_area"]) / info_table_filt["user_surface_area"]
    )


# %%
def segment_save_props_ds(img_path, save_path, csv_name):
    """
    Reads downsampled images given by 'img_path', segments the images, extracts region properties, and saves them in 'csv_name' at 'save_path'.

    Parameters:
    img_path (str): Path to the image file.
    save_path (str): Directory where the CSV file will be saved.
    csv_name (str): Name of the CSV file to save the region properties.

    Returns:
    None
    """
    if "rfp" in img_path.casefold():
        labels, info_table = get_info_table_macs(img_path)
        if np.all(labels is None):  # return for non-zstacks
            return
        filt_labels, info_table_filt = filter_info_table_macs(labels, info_table)
    elif "gfp" in img_path.casefold():
        labels, info_table = get_info_table_ens(img_path)
        if np.all(labels is None):  # return for non-zstacks
            return
        filt_labels, info_table_filt = filter_info_table_ens(labels, info_table)
    else:
        print("ERROR: img location name must have gfp or rfp keywords")
        exit()

    find_surface_area(filt_labels, info_table_filt)

    if not os.path.exists(save_path):  # check if the save_path for region_props exist
        print("Save path doesn't exist")
        os.makedirs(save_path)
        print(f"{save_path} created..")
    else:
        print("save path exists..")
    info_table_filt.to_csv(os.path.join(save_path, csv_name))
    print("Successfully saved: " + csv_name)


# %% [markdown]
# Save region props for all the time points
main_dir = input("Enter the Main directory containing ALL images to be segmented: ")

sub_dirs = ["GFP", "RFP"]  # as we can only segment in fluorescent channels

print("Images need to be sorted in different directories by channel(BF/GFP/RFP).")
print("Run the sort_img_by_channels.py script before running this")

flag = input("Did you sort images by channel? (y/n)")
if flag.casefold().startswith("y"):
    print("ok, starting segmentation")
else:
    print("Okay, bye!")
    exit()

# now do os walk then send all images to the segment function to starting segmentation
for root, subfolders, filenames in os.walk(main_dir):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        filename_list = filename.split(".")
        og_name = filename_list[0]  # first of list=name
        ext = filename_list[-1]  # last of list=extension

        if ext == "tif" or ext == "tiff":  # only if tiff file
            for sub in sub_dirs:
                if sub.casefold() in og_name.casefold():  # find the image channel
                    save_path = os.path.join(root, sub.casefold() + "_region_props")
                    segment_save_props_ds(img_path=filepath, save_path=save_path, csv_name=og_name + "_info.csv")
