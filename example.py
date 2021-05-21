import numpy as np
import nxtomowriter as ntw

if __name__ == '__main__':  
    in_filename = 'demo.nxs' #'data/IMAT00008300.nxs'

    angles = np.linspace(0, 390, 1125).tolist()
    projection_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/Tomo'

    # Optional
    dark_before_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/dark_before' 
    flat_before_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/flat_before'  
    flat_after_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/flat_after'
    dark_after_path = 'D:/Downloads/dataset_phantom_rebin122_150um/modified/dark_after'

    open_beam_position = (370, -207, -180)  # x y z positioner values
    sample_position  = (468, -7, -180)  # x y z positioner values 

    ntw.save_tomo_to_nexus(in_filename, angles, projection_path, dark_before=dark_before_path,
                           flat_before=flat_before_path, flat_after=flat_after_path,
                           dark_after=dark_after_path, open_beam_position=open_beam_position, 
                           sample_position=sample_position)
