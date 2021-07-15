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
A Python wheel of the beta can be installed using pip. Open the command line and type

    pip install nxtomowriter-0.2.0b0-py3-none-any.whl

Usage
-----
The example below writes the white beam flower dataset into a copy of the given nexus file.

    import nxtomowriter as ntw

    if __name__ == '__main__':     
            filename = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/IMAT00010675.nxs'

            projection_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/Tomo'
            angles = (0.0, 359.9584)
            

            dark_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/dark_before' 
            flat_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_before'  
            flat_after_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_after'
            dark_after_path = ''
            half_circle_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/180deg'
            
            open_beam_position = (370, -207, -180)  
            projection_position  = (468, -7, -180)  

            out_filename = ntw.save_tomo_to_nexus(in_filename, angles, projection_path, dark_before=dark_before_path,
                                                 flat_before=flat_before_path, flat_after=flat_after_path,
                                                 dark_after=dark_after_path, open_beam_position=open_beam_position, 
                                                 projection_position=projection_position)

            print(f'Data was saved successfully to {out_filename}')

If the example script is written into a file called "example.py", the script can be run from the command line

    python example.py

Explanation
-----------
if *filename* is an existing nexus file, the NXtomo entry will be appended to the existing NXentry (**/raw_data_1/tomo_entry**) 
otherwise a new file will be created with entry (**/entry/tomo_entry**). The NXtomo entry will be appended to a copy of the nexus file if 
*make_copy* is True (which is the default). If a tomo_entry already exist in the file, an exception will be raised. 

    filename = 'IMAT00010675.nxs'
    make_copy = True

The path of the projection images and the rotation angles in degrees are the only required data. Rotation angle 
can either be a path to a file containing the angles or a start and stop angle in a tuple which will be used to 
generate n equally spaced angles (n is the number of projection images) 

    projection_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/Tomo'
    angles = (0.0, 359.9584)

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

    out_filename = ntw.save_tomo_to_nexus(filename, angles, 
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
**out_filename = nxtomowriter.save_tomo_to_nexus(filename, rot_angles, projections, dark_before='', flat_before='', half_circle='', 
                                                 flat_after='', dark_after='', rotation_axis = 1, open_beam_position=(), projection_position=(), 
                                                 make_copy=True):**

**Parameter**:

* **filename** (str): path of nexus file 
* **rot_angles** (Union[str, Tuple[float, float]]): path of file with rotation angles or tuple of start and stop angles for projection images (in degrees)
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

Returns

* **out_filename** (str): path of written (output) nexus file 