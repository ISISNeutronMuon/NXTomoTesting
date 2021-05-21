import numpy as np
import nxtomowriter as ntw

if __name__ == '__main__':  
        # if "in_filename" is an existing nexus file, the NXtomo entry will be appended to 
        # the existing NXentry (/raw_data_1/tomo_entry) otherwise a new entry will be created
        # (/entry/tomo_entry). The NXtomo entry will be appended to a copy of the nexus file if
        # make_copy is True (which is the default).   
        in_filename = 'IMAT00010675.nxs'

        # The path of the projection images and the rotation angles in degrees are the only required data
        projection_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/Tomo'
        angles = np.linspace(0, 390, 1143).tolist()
        
        # Optional keywords
        dark_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/dark_before' 
        flat_before_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_before'  
        flat_after_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/flat_after'
        dark_after_path = ''
        half_circle_path = '//ISIS/Shares/IMAT/ExampleData/Flower_WhiteBeam/180deg'
        
        # X Y Z positioning system values for open beam images
        open_beam_position = (370, -207, -180)  
        
        # X Y Z positioning system values for projection images 
        sample_position  = (468, -7, -180)  

        # Calling the function below will append the tomography data to a copy of the input file 
        # (IMAT00010675_with_tomo.nxs). The data is viewable using the ImageJ loader for HDF5 
        # from PSI and works in SAVU.
        ntw.save_tomo_to_nexus(in_filename, angles, projection_path, dark_before=dark_before_path,
                               flat_before=flat_before_path, flat_after=flat_after_path,
                               half_circle=half_circle_path, open_beam_position=open_beam_position, 
                               sample_position=sample_position)