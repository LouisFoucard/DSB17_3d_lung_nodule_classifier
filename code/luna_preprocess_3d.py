import os
import sys
from glob import glob

import SimpleITK as sitk
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.ndimage
from keras import backend as K
from tqdm import tqdm

K.set_image_dim_ordering('tf')


# Some helper functions

def matrix2int16(matrix):
    '''
    matrix must be a numpy array NXN
    Returns uint16 version
    '''
    m_min = np.min(matrix)
    m_max = np.max(matrix)
#     print(m_min, m_max)
    matrix = matrix - m_min
    return np.array(np.rint(matrix / float(m_max - m_min) * 65535.0), dtype=np.uint16)


def make_mask(center, diam, z, width, height, spacing, origin):
    '''
    Center : centers of circles px -- list of coordinates x,y,z
    diam : diameters of circles px -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)
    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam and abs(center[2]-z)<diam/2.5:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return mask.astype(np.uint8)


#####################
#

# Load the scans in given folder path
class Dicom_ersatz:
    def __init__(self, spacing, pixel_array, direction, mask_array=None):
        if mask_array is None:
            mask_array = []
        self.SliceThickness = spacing[2]
        self.pixel_array = pixel_array
        self.PixelSpacing = list(spacing[:2])
        self.direction = direction
        self.mask_array = mask_array


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image < -1548] = -1000
    image2 = np.zeros(image.shape, dtype=np.uint16)
    for n, im in enumerate(image):
        im = matrix2int16(im)
        image2[n, ...] = im

    return image2


def get_masks(scans):
    masks = np.stack([s.mask_array if s.direction[0] > 0 else s.mask_array.transpose() for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    masks = masks.astype(np.int8)

    return masks


def resample(image, scan, new_spacing=None, crop=(0, 0)):
    # Determine current pixel spacing

    if new_spacing is None:
        new_spacing = [1, 1, 1]
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    #     print(resize_factor)
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    #     image = image[...,crop[0]:-crop[0], crop[1]:-crop[1]]
    return image, new_spacing


def load_scan(path, df_node):
    mini_df = df_node[df_node["file"] == path]  # get all nodules associate with file

    nodes = []
    for index, row in mini_df.iterrows():
        #         print(row)
        node_x = row["coordX"]
        node_y = row["coordY"]
        node_z = row["coordZ"]
        diam = row["diameter_mm"]
        nodes += [(node_x, node_y, node_z, diam)]

    # print(path)
    itk_img = sitk.ReadImage(path)
    slices_npy = sitk.GetArrayFromImage(itk_img)
    num_z, height, width = slices_npy.shape
    spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
    direction = np.array(itk_img.GetDirection())

    origin = np.array(itk_img.GetOrigin())
    slices = []

    for iz, s in enumerate(slices_npy):

        mask = np.zeros([height, width], dtype=np.uint8)

        for node in nodes:
            node2 = direction[0] * node[0], direction[0] * node[1], node[2], node[3]
            origin2 = direction[0] * origin[0], direction[0] * origin[1], origin[2]
            center = np.array([node2[0], node2[1], node2[2]])
            if direction[0] < 0:
                center[0], center[1] = center[1], center[0]

            diam = node2[3]
            mask += make_mask(center, diam, iz * spacing[2] + origin2[2], width, height, spacing, origin2)

        slices += [Dicom_ersatz(spacing, s, direction, mask)]

    tumor_coords = []
    for node in nodes:
        node2 = direction[0] * node[0], direction[0] * node[1], node[2], node[3]
        origin2 = direction[0] * origin[0], direction[0] * origin[1], origin[2]
        center = ((np.array([node2[0], node2[1], node2[2]]) - np.array(origin2)) / spacing).astype('int16')
        center[0], center[1] = center[1], center[0]
        diam = int(node[3] / spacing[0])
        tumor_coords += [(center, diam)]

    return slices, tumor_coords


def main(n_folders):
    for n_folder in n_folders:
        # Getting list of image files
        luna_path = "../data/luna/"

        luna_subset_path = luna_path + "subset{}/".format(n_folder)
        output_path = "../data/luna/preprocessed_3d/"
        file_list = glob(luna_subset_path + "*.mhd")
        print(len(file_list))

        # Helper function to get rows in data frame associated
        # with each file
        def get_filename(case):
            for f in file_list:
                if case in f:
                    return f

        # The locations of the nodes
        df_node = pd.read_csv(luna_path + "annotations.csv")
        df_node["file"] = df_node["seriesuid"].apply(get_filename)
        df_node = df_node.dropna()
        pad = 32

        for patient_file in tqdm(file_list):
            try:
                first_patient, tumor_info = load_scan(patient_file, df_node)

                first_patient_pixels = get_pixels_hu(first_patient)
                first_patient_masks = get_masks(first_patient)

                pix_resampled, spacing = resample(first_patient_pixels, first_patient,
                                                  [1.25, first_patient[0].PixelSpacing[0],
                                                   first_patient[0].PixelSpacing[1]])
                masks_resampled, spacing = resample(first_patient_masks, first_patient,
                                                    [1.25, first_patient[0].PixelSpacing[0],
                                                     first_patient[0].PixelSpacing[1]])

                ratio = float(first_patient_pixels.shape[0]) / pix_resampled.shape[0]
                #     print(ratio)
                #     print(pix_resampled.shape)
                #     print(first_patient_pixels.shape)

                #     print(tumor_info)
                # get 10 random cubes:
                print('found {} tumors'.format(len(tumor_info)))

                if not tumor_info:
                    x_rand = np.random.randint(50, 420, 10)
                    y_rand = np.random.randint(50, 420, 10)
                    z_rand = np.random.randint(32, pix_resampled.shape[0] - 32, 10)
                    for n_t, (x, y, z) in enumerate(zip(x_rand, y_rand, z_rand)):
                        tumor_cube = pix_resampled[z - pad:z + pad,
                                     x - pad: x + pad,
                                     y - pad: y + pad]
                        mask_cube = masks_resampled[z - pad:z + pad,
                                    x - pad: x + pad,
                                    y - pad: y + pad]

                        #             for im, mask in zip(tumor_cube, mask_cube):

                        #                 plt.imshow(im, cmap=plt.cm.gray)
                        #                 plt.imshow(mask, cmap=plt.cm.cool, alpha=0.2)
                        #                 plt.show()

                        np.save(
                            output_path + 'cube_scans_neg/{}_{}.npy'.format(n_t, os.path.basename(patient_file)[:-4]),
                            tumor_cube)

                else:

                    x_rand = np.random.randint(50, 420, 5)
                    y_rand = np.random.randint(50, 420, 5)
                    z_rand = np.random.randint(50, pix_resampled.shape[0] - 50, 5)
                    for n_t, (x, y, z) in enumerate(zip(x_rand, y_rand, z_rand)):
                        tumor_cube = pix_resampled[z - pad:z + pad,
                                     x - pad: x + pad,
                                     y - pad: y + pad]
                        mask_cube = masks_resampled[z - pad:z + pad,
                                    x - pad: x + pad,
                                    y - pad: y + pad]

                        np.save(
                            output_path + 'cube_scans_neg/{}_{}.npy'.format(n_t, os.path.basename(patient_file)[:-4]),
                            tumor_cube)

                        #             for im, mask in zip(tumor_cube, mask_cube):

                        #                 plt.imshow(im, cmap=plt.cm.gray)
                        #                 plt.imshow(mask, cmap=plt.cm.cool, alpha=0.2)
                        #                 plt.show()

                for n_t, tumor in tqdm(enumerate(tumor_info)):
                    tumor_coords = tumor[0]
                    tumor_coords[2] = int(tumor_coords[2] / ratio)
                    r_offset = np.random.randint(-16, 16)

                    tumor_cube = pix_resampled[tumor_coords[2] + r_offset - pad:tumor_coords[2] + r_offset + pad,
                                 tumor_coords[0] + r_offset - pad: tumor_coords[0] + r_offset + pad,
                                 tumor_coords[1] + r_offset - pad: tumor_coords[1] + r_offset + pad]
                    mask_cube = masks_resampled[tumor_coords[2] + r_offset - pad:tumor_coords[2] + r_offset + pad,
                                tumor_coords[0] + r_offset - pad: tumor_coords[0] + r_offset + pad,
                                tumor_coords[1] + r_offset - pad: tumor_coords[1] + r_offset + pad]

                    np.save(output_path + 'cube_scans_pos/0_{}_{}.npy'.format(n_t, os.path.basename(patient_file)[:-4]),
                            tumor_cube)
                    np.save(output_path + 'cube_masks_pos/0_{}_{}.npy'.format(n_t, os.path.basename(patient_file)[:-4]),
                            mask_cube)

                    r_offset = np.random.randint(-16, 16)

                    tumor_cube = pix_resampled[tumor_coords[2] + r_offset - pad:tumor_coords[2] + r_offset + pad,
                                 tumor_coords[0] + r_offset - pad: tumor_coords[0] + r_offset + pad,
                                 tumor_coords[1] + r_offset - pad: tumor_coords[1] + r_offset + pad]
                    mask_cube = masks_resampled[tumor_coords[2] + r_offset - pad:tumor_coords[2] + r_offset + pad,
                                tumor_coords[0] + r_offset - pad: tumor_coords[0] + r_offset + pad,
                                tumor_coords[1] + r_offset - pad: tumor_coords[1] + r_offset + pad]

                    np.save(output_path + 'cube_scans_pos/1_{}_{}.npy'.format(n_t, os.path.basename(patient_file)[:-4]),
                            tumor_cube)
                    np.save(output_path + 'cube_masks_pos/1_{}_{}.npy'.format(n_t, os.path.basename(patient_file)[:-4]),
                            mask_cube)

            except Exception as e:
                print(patient_file, e)

if __name__ == '__main__':
    args = [int(i) for i in sys.argv[1:]]
    print(args)
    main(args)
