# import multiprocessing as mp
# import time
# import numpy as np
# import tifffile
# from ScanImageTiffReader import ScanImageTiffReader
# from PIL import Image
# import fabio
# import cv2

# size = 1000
# image_names = [f'D:/Downloads/dataset_phantom_rebin122_150um/full/proj_{i:04d}.tiff' for i in range(size)]
# a = np.zeros((size, 512, 512), np.uint16)


# start = time.perf_counter()
# for index, name in enumerate(image_names):
#     image = tifffile.imread(name)
#     a[index, :, :] = image
# print('tifffile:', time.perf_counter()-start, 'sec')

# start = time.perf_counter()
# for index, name in enumerate(image_names):
#     image = ScanImageTiffReader(name).data()
#     a[index, :, :] = image
# print('scan_image:', time.perf_counter()-start, 'sec')

# start = time.perf_counter()
# for index, name in enumerate(image_names):
#     image = fabio.open(name).data
#     a[index, :, :] = image
# print('fabio:', time.perf_counter()-start, 'sec')

# start = time.perf_counter()
# for index, name in enumerate(image_names):
#     image = Image.open(name)
#     a[index, :, :] = image
# print('Pillow:', time.perf_counter()-start, 'sec')

# start = time.perf_counter()
# for index, name in enumerate(image_names):
#     image = cv2.imread(name, 0)
#     a[index, :, :] = image
# print('OpenCV:', time.perf_counter()-start, 'sec')

import numpy as np
import nxtomowriter as ntw

if __name__ == '__main__':  
    in_filename = 'data/IMAT00008300.nxs'

    angles = np.linspace(0, 390, 1125).tolist()
    projection_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/Tomo'

    # Optional
    dark_before_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/dark_before' 
    flat_before_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/flat_before'  
    flat_after_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/flat_after'
    dark_after_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/dark_after'

    ntw.save_tomo_to_nexus(in_filename, angles, projection_path, dark_before=dark_before_path,
                           flat_before=flat_before_path, flat_after=flat_after_path,
                           dark_after=dark_after_path)