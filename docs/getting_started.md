Introduction
============
NXtomoWriter provides a function to convert tomography data stored as TIFF image files to a Nexus compliant file format 
(using [NXtomo](https://manual.nexusformat.org/classes/applications/NXtomo.html))


- [Installation](#Installation)
- [Usage](#Usage)
- [Explanation](#Explanation)
- [Reference](#Reference)

Installation
------------
A Python wheel of the beta can be downloaded from here. The wheel can be nistalled using pip

    pip install nxtomowriter-0.1.0b0-py3-none-any.whl

Usage
-----
The example below writes the white beam flower dataset into the given nexus file. The example file can be download from here 

    import numpy as np
    import nxtomowriter as ntw

    if __name__ == '__main__':     
            filename = 'IMAT00010675.nxs'

            projection_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/Tomo'
            angles = np.linspace(0, 390, 1143).tolist()
            

            dark_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/dark_before' 
            flat_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_before'  
            flat_after_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_after'
            dark_after_path = ''
            half_circle_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/180deg'
            
            open_beam_position = (370, -207, -180)  
            projection_position  = (468, -7, -180)  

            ntw.save_tomo_to_nexus(filename, angles, projection_path, dark_before=dark_before_path,
                                   flat_before=flat_before_path, flat_after=flat_after_path,
                                   half_circle=half_circle_path, open_beam_position=open_beam_position, 
                                   projection_position=projection_position)

Explanation
-----------
if *filename* is an existing nexus file, the NXtomo entry will be appended to the existing NXentry (**/raw_data_1/tomo_entry**) 
otherwise a new entry will be created (**/entry/tomo_entry**). The NXtomo entry will be appended to a copy of the nexus file if 
*make_copy* is True (which is the default). If a tomo_entry already exist in the file, an exception will be raised. 

    filename = 'IMAT00010675.nxs'
    make_copy = True

The path of the projection images and the rotation angles in degrees are the only required data

    projection_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/Tomo'
    angles = np.linspace(0, 390, 1143).tolist()

The paths for the dark and flat images are optional. *half_circle_path* is the path for the 180 image folder.

    dark_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/dark_before' 
    flat_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_before'  
    flat_after_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_after'
    dark_after_path = ''
    half_circle_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/180deg'

Optionally, the translation values for the open beam frames and the projection frames can be specified as a tuple of X Y Z values.

    open_beam_position = (370, -207, -180)          
    projection_position  = (468, -7, -180) 

Also rotation axis can be specified using 0 for x-axis and 1 for y-axis. 

    rotation_axis = 1

Calling the *save_tomo_to_nexus* function below will append the tomography data to a copy of the input file (IMAT00010675_with_tomo.nxs). 
The data is viewable using the ImageJ loader for HDF5 from PSI and works in SAVU.

    ntw.save_tomo_to_nexus(filename, angles, 
                           projection_path, 
                           dark_before=dark_before_path,
                           flat_before=flat_before_path, 
                           flat_after=flat_after_path,
                           half_circle=half_circle_path, 
                           rotation_axis=rotation_axis,
                           open_beam_position=open_beam_position, 
                           projection_position=projection_position,
                           make_copy=make_copy)


Reference
---------
**nxtomowriter.save_tomo_to_nexus(filename, rot_angles, projections, dark_before='', flat_before='', half_circle='', 
                                flat_after='', dark_after='', rotation_axis = 1, open_beam_position=(), projection_position=(), 
                                make_copy=True):**

**Parameter**:


* **filename** (str): path of nexus file 
* **rot_angles** (List[float]): list of rotation angles for projection images (in degrees)
* **projections** (str): directory of projection images 

Keyword Arguments:

* **dark_before** (str, Optional): directory of dark before images   
* **flat_before** (str, Optional): directory of flat before images 
* **half_circle** (str, Optional): directory of 180 degree images 
* **flat_after** (str, Optional): directory of flat after images 
* **dark_after** (str, Optional): directory of dark after images 
* **rotation_axis** (int, Optional): axis of rotation (i.e. 0 for x-axis, 1 for y-axis), default is 1
* **open_beam_position** ([float, float, float], Optional): X, Y, Z, positioner (in mm) values when imaging open beam 
* **projection_position** ([float, float, float], Optional): X, Y, Z, positioner values (in mm)  when imaging sample 
* **make_copy** (bool, Optional): indicates tomo entry should be added to copy of the nexus file, default is True 
    