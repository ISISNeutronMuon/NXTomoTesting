import nxtomowriter as ntw

if __name__ == '__main__':  
        # Example Usage: save_tomo_to_nexus
        #  
        # if "filename" is an existing nexus file, the NXtomo entry will be appended to 
        # the existing NXentry (/raw_data_1/tomo_entry) otherwise a new file will be created with entry 
        # (/entry/tomo_entry). The NXtomo entry will be appended to a copy of the existing nexus file if
        # make_copy is True (which is the default). If a tomo_entry already exist in the file, an 
        # exception will be raised.  
        filename = 'IMAT00010675.nxs'
        make_copy = True

        # The path of the projection images and the rotation angles in degrees are the only required data
        # Rotation angle can either be a path to a file containing the angles or a start and stop angle in 
        # a tuple which will be used to generate n equally spaced angles (n is the number of projection images) 
        projection_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/Tomo'
        angles = (0, 359.9584)
        
        # Optional keywords
        dark_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/dark_before' 
        flat_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_before'  
        flat_after_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_after'
        dark_after_path = ''
        half_circle_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/180deg'

        # X Y Z positioning system values for open beam images
        open_beam_position = (370, -207, -180)  
        
        # X Y Z positioning system values for projection images 
        projection_position  = (468, -7, -180)

        # rotation axis, 0 indicates x-axis and 1 indicates y-axis (default value is 1)
        rotation_axis = 1

        # Calling the function below will append the tomography data to a copy of the input file 
        # (IMAT00010675_with_tomo.nxs). The data is viewable using the ImageJ loader for HDF5 
        # from PSI and works in SAVU.
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
