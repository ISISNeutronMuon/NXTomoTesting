import datetime
import os
import shutil
import h5py
import numpy as np
import tifffile


def add_nxtomo_entry(filename, image_names, image_keys, angles):
    time = datetime.datetime.now().isoformat()
    
    with h5py.File(filename, 'r+') as nxs_file:
        main_entry = None
        for _, item in nxs_file.items():
            if item.attrs.get('NX_class').decode('utf-8') == 'NXentry':
                main_entry = item
        
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
            # start = time.perf_counter() 
            dset[index + 1, :, :] = image
            # print(time.perf_counter() - start, 'sec')
            # print(index)
        
        sample = entry.create_group('sample')
        sample.attrs['NX_class'] = u'NXsample'
        sample['name'] = main_entry.get('sample/title', u'fred')
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
    images = []

    with os.scandir(image_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            
            name = entry.name.lower()
            
            if not(name.endswith('.tiff') or name.endswith('.tif')):
                continue 

            images.append(entry.path)

    return images


def prepare_images(projections, dark_before='', flat_before='', half_circle='',  
                   flat_after='', dark_after=''):
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
        angles.extend([0.0] * size)

    return image_names, image_keys, angles


def save_tomo_to_nexus(in_filename, out_filename, projections, dark_before='', flat_before='', half_circle='',  
                       flat_after='', dark_after=''):
    
    image_names, image_keys, angles = prepare_images(projections, dark_before, flat_before, half_circle,  
                                                     flat_after, dark_after)
    
    shutil.copyfile(in_filename, out_filename)
    add_nxtomo_entry(os.path.abspath(out_filename), image_names, image_keys, angles)


if __name__ == '__main__':  
    in_filename = 'IMAT00008300.nxs'
    out_filename = f'mod_{in_filename}'

    projection_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/Tomo'
    dark_before_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/dark_before' 
    flat_before_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/flat_before'  
    flat_after_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/flat_after'
    dark_after_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/dark_after'

    save_tomo_to_nexus(in_filename, out_filename, projection_path, dark_before=dark_before_path,
                       flat_before=flat_before_path, flat_after=flat_after_path,
                       dark_after=dark_after_path)
