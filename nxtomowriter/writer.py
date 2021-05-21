from contextlib import suppress
import datetime
import os
from os.path import sameopenfile
import shutil
import h5py
import numpy as np
import tifffile
from tqdm import tqdm


def add_nxtomo_entry(filename, image_names, image_keys, angles, translations=None):
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
        
        image_count = len(image_names)
        with tqdm(total=image_count, bar_format='{l_bar}{bar:60}{r_bar}{bar:-10b}') as progress_bar:
            detector['image_key'] = np.array(image_keys, dtype=np.uint8)
            image = tifffile.imread(image_names[0])
            shape = (image_count, *image.shape)
            dset = nxs_file.create_dataset(f'{detector.name}/data', shape=shape, dtype=image.dtype)
            dset[0, :, :] = image
            progress_bar.update(1)
            for index, name in enumerate(image_names[1:]):
                image = tifffile.imread(name)
                dset[index + 1, :, :] = image
                progress_bar.update(1)
        
        sample = entry.create_group('sample')
        sample.attrs['NX_class'] = u'NXsample'
        sample['name'] =  main_entry.get('sample/name', u'')
        sample['rotation_angle'] = np.array(angles, dtype=np.float32)
        sample['rotation_angle'].attrs['units'] = 'degrees'
        sample['rotation_angle'].attrs['axis'] = 1

        if translations is not None:
            sample['x_translation'] = translations[:, 0].astype(np.float32)
            sample['x_translation'].attrs['units'] = 'mm'
            sample['y_translation'] = translations[:, 1].astype(np.float32)
            sample['y_translation'].attrs['units'] = 'mm'
            sample['z_translation'] = translations[:, 2].astype(np.float32)
            sample['z_translation'].attrs['units'] = 'mm'

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
        
        if index == 2:  # projection
            if size != len(rot_angles):
                raise ValueError(f'The number of projection images {size} does not match the number of angles {len(rot_angles)}.')
            angles.extend(rot_angles)
        elif index == 3:  # 180 image
            angles.extend([180.0] * size)
        else:  # # before and after images
            angles.extend([0.0] * size)


    return image_names, image_keys, angles


def save_tomo_to_nexus(filename, rot_angles, projections, dark_before='', flat_before='', half_circle='', 
                       flat_after='', dark_after='', open_beam_position=(), sample_position=(), make_copy=True):
    """Saves tomography data to a given nexus file using the NXtomo standard. If the nexus file exist the 
    NXtomo entry (tomo_entry) will be appended as a subentry of the first NXentry in the file. If the file 
    does not exist or the file has no NXentry, the file will be created if needed and a new entry will be 
    created then the NXtomo entry will be appended to it (/entry/tomo_entry). If make_copy is True (default), 
    the NXtomo entry will be appended to a copy of the nexus file instead. The copied file will be in the same 
    directory with the suffix "_with_tomo" appended to the original filename.   

    :param filename: path of nexus file 
    :type filename: str
    :param rot_angles: list of rotation angles for projection images (in degrees)
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
    :param open_beam_position: X, Y, Z, positioner (in mm) values when imaging open beam 
    :type open_beam_position: [float, float, float]
    :param open_beam_position: X, Y, Z, positioner values (in mm)  when imaging sample 
    :type open_beam_position: [float, float, float]
    :param make_copy: indicates tomo entry should be added to copy of the nexus file 
    :type make_copy: bool
    """
    image_names, image_keys, angles = prepare_images(rot_angles, projections, dark_before, flat_before, half_circle,  
                                                     flat_after, dark_after)

    if not sample_position and not open_beam_position:
        translations = None
    elif sample_position and open_beam_position:
        translations = np.tile(open_beam_position, (len(image_names), 1))
        translations[np.argwhere(np.equal(image_keys, 0)), :] = sample_position
    elif sample_position:
        translations = np.tile(sample_position, (len(image_names), 1))
    else:
        raise ValueError(f'The sample position is also required when setting open beam position.')
    
    out_filename = filename 
     
    if os.path.isfile(filename) and make_copy:
        tmp = os.path.splitext(filename)
        out_filename = f'{tmp[0]}_with_tomo{tmp[1]}'
        shutil.copyfile(filename, out_filename)
        
    add_nxtomo_entry(out_filename, image_names, image_keys, angles, translations)
