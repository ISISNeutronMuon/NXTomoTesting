from contextlib import suppress
import datetime
import os
import shutil
import h5py
import numpy as np
import tifffile


def add_nxtomo_entry(filename, image_names, image_keys, angles):
    """Adds nxtomo entry to the given nexus file 

    :param filename: path to nexus file 
    :type filename: str
    :param image_names: list of images paths
    :type image_names: List[str]    
    :param image_keys: list of image keys 
    :type image_keys: List[int]
    :param angles: list of rotation angles 
    :type angles: List[float]
    """

    time = datetime.datetime.now().isoformat()
    
    with h5py.File(filename, 'a') as nxs_file:
        main_entry = None
        for _, item in nxs_file.items():
            nx_class = item.attrs.get('NX_class')
            with suppress(AttributeError):
                nx_class = nx_class.decode('utf-8')
            if nx_class == 'NXentry':
                main_entry = item
                break
        
        if main_entry is None:
            main_entry = nxs_file.create_group('entry')
            main_entry.attrs['NX_class'] = u'NXentry'
        
        nxs_file.attrs['HDF5_Version'] = h5py.version.version
        nxs_file.attrs['file_name'] = filename 
        nxs_file.attrs['file_time'] = time
        entry = main_entry.create_group('tomo_entry')
        entry.attrs['NX_class'] = u'NXsubentry'
        
        entry['definition'] = u'NXtomo'
        entry['title'] = main_entry.get('title', u'')
        entry['start_time'] = main_entry.get('start_time', time)
        entry['start_time'].attrs['units'] = u'ISO8601'
        entry['end_time'] = main_entry.get('end_time', time)
        entry['end_time'].attrs['units'] = u'ISO8601'

        instrument = entry.create_group('instrument')
        instrument.attrs['NX_class'] = u'NXinstrument'

        source = instrument.create_group('source')
        source.attrs['NX_class'] = u'NXsource'
        source['type'] = 'Spallation Neutron Source'
        source['name'] = 'ISIS'
        source['probe'] = 'neutron'
        
        detector = instrument.create_group('detector')
        detector.attrs['NX_class'] = u'NXdetector'
        detector['image_key'] = np.array(image_keys, dtype=np.uint8)
        image = tifffile.imread(image_names[0])
        shape = (len(image_names), *image.shape)
        dset = nxs_file.create_dataset(f'{detector.name}/data', shape=shape, dtype=image.dtype)
        dset[0, :, :] = image
        for index, name in enumerate(image_names[1:]):
            image = tifffile.imread(name)
            dset[index + 1, :, :] = image
        
        sample = entry.create_group('sample')
        sample.attrs['NX_class'] = u'NXsample'
        sample['name'] =  main_entry.get('sample/name', u'')
        sample['rotation_angle'] = np.array(angles, dtype=np.float32)
        sample['rotation_angle'].attrs['units'] = 'degrees'
        sample['rotation_angle'].attrs['axis'] = 1

        # Create the LINKS         
        data = entry.create_group('data')
        data.attrs['NX_class'] = u'NXdata'
        data['data'] = entry['instrument/detector/data']
        data['rotation_angle'] = entry['sample/rotation_angle']
        data['image_key'] = entry['instrument/detector/image_key']

   
def get_tiffs(image_dir):
    """Gets list of tiff images in the given directory

    :param image_dir: directory of images 
    :type image_dir: str
    :return: List of absolute paths of images
    :rtype: List[str]
    """
    images = []

    with os.scandir(image_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            
            name = entry.name.lower()
            
            if not(name.endswith('.tiff') or name.endswith('.tif')):
                continue 

            images.append(entry.path)

    return sorted(images)


def prepare_images(rot_angles, projections, dark_before='', flat_before='', half_circle='',  
                   flat_after='', dark_after=''):
    """Gets image data from given directories and place them in the appropriate order 

    :param rot_angles: list of rotation angles for projection images 
    :type rot_angles: List[float]
    :param projections: directory of projection images 
    :type projections: str
    :param dark_before: directory of dark before images 
    :type dark_before: str    
    :param flat_before: directory of flat before images 
    :type flat_before: str
    :param half_circle: directory of 180 degree images 
    :type half_circle: str
    :param flat_after: directory of flat after images 
    :type flat_after: str
    :param dark_after: directory of dark after images 
    :type dark_after: str
    :return: List of absolute paths of images, keys and and angle
    :rtype: Tuple[List[str], List[int], List[float]]
    """
    if not projections:
        raise ValueError('A path to the projection images must be provided!')
    
    start_angle, stop_angle = rot_angles[0], rot_angles[-1]
    paths = [dark_before, flat_before, projections, half_circle, flat_after, dark_after]  
    keys = [2, 1, 0, 0, 1, 2]

    image_names = []
    image_keys = []
    angles = []
    for index, value in enumerate(paths):
        if not value:
            continue

        images = get_tiffs(value)
        if not images:
            raise FileNotFoundError(f'No tiffs were found in the directory: {value}.')    

        key = keys[index]
        size = len(images)
        image_names.extend(images)
        image_keys.extend([key] * size)
        
        if index < 2:  # before images
            angles.extend([start_angle] * size)
        elif index == 2:  # projection
            if size != len(rot_angles):
                raise ValueError(f'The number of projection images {size} does not match the number of angles {len(rot_angles)}.')
            angles.extend(rot_angles)
        elif index == 3:  # 180 image
            angles.extend([180.0] * size)
        else:  # after images
            angles.extend([stop_angle] * size)

    return image_names, image_keys, angles


def save_tomo_to_nexus(filename, rot_angles, projections, dark_before='', flat_before='', 
                       half_circle='', flat_after='', dark_after='', make_copy=True):
    """Saves tomography data to a given nexus file using the NXtomo standard

    :param filename: path of input nexus file 
    :type filename: str
    :param rot_angles: list of rotation angles for projection images 
    :type rot_angles: List[float]
    :param projections: directory of projection images 
    :type projections: str
    :param dark_before: directory of dark before images 
    :type dark_before: str    
    :param flat_before: directory of flat before images 
    :type flat_before: str
    :param half_circle: directory of 180 degree images 
    :type half_circle: str
    :param flat_after: directory of flat after images 
    :type flat_after: str
    :param dark_after: directory of dark after images 
    :type dark_after: str
    :param make_copy: indicates tomo entry should be added to copy of the nexus file 
    :type make_copy: bool
    """
    image_names, image_keys, angles = prepare_images(rot_angles, projections, dark_before, flat_before, half_circle,  
                                                     flat_after, dark_after)
    out_filename = filename 
    if make_copy:
        tmp = os.path.splitext(filename)
        out_filename = f'{tmp[0]}_with_tomo{tmp[1]}'
        shutil.copyfile(filename, out_filename)
        
    add_nxtomo_entry(out_filename, image_names, image_keys, angles)



