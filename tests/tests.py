import os
import shutil
import tempfile
import unittest
import unittest.mock as mock
import h5py
import numpy as np
import tifffile
from nxtomowriter import writer 


class TestWriter(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.file_descriptors = []

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    @mock.patch('nxtomowriter.writer.tqdm', autospec=True)
    def testAddNxtomoEntry(self, tqdm):
        filename = os.path.join(self.test_dir, 'output.nxs')
        image_names = []
        image_keys = [2, 2, 2, 0, 0, 0, 0, 1, 1, 1]
        angles = [0, 0, 0, 0, 45, 90, 180, 180, 180, 180]

        rints = np.random.randint(0, np.iinfo(np.uint16).max, size=(10, 128, 128)).astype(np.uint16)
        for i in range(rints.shape[0]):
            image_names.append(os.path.join(self.test_dir, f'image_{i}.tif'))
            tifffile.imwrite(image_names[-1], rints[i])

        writer.add_nxtomo_entry(filename, image_names, image_keys, angles)

        with h5py.File(filename, 'r') as new_file:
            np.testing.assert_array_equal(rints, new_file['/entry/tomo_entry/data/data'])
            np.testing.assert_array_equal(image_keys, new_file['/entry/tomo_entry/data/image_key'])
            np.testing.assert_array_almost_equal(angles, new_file['/entry/tomo_entry/data/rotation_angle'])
            self.assertEqual(new_file['/entry/tomo_entry/data/rotation_angle'].attrs['axis'], 1)
            self.checkStringEqual(new_file['/entry/tomo_entry/sample/name'][()], b'')
            self.checkStringEqual(new_file['/entry/tomo_entry/definition'][()], b'NXtomo')
            self.checkStringEqual(new_file['/entry/tomo_entry/title'][()], b'')

        filename = os.path.join(self.test_dir, 'output.nxs')
       
        with h5py.File(filename, 'w') as new_file:
            main_entry = new_file.create_group('entry')
            main_entry.attrs['NX_class'] = 'NXentry'
            main_entry['title'] = 'Title'
            main_entry['sample/name'] = 'Sample'
        
        translations = np.random.randint(-1000, 1000, size=(10, 3)).astype(np.float32)
        writer.add_nxtomo_entry(filename, image_names, image_keys, angles, translations, rotation_axis=0)

        with h5py.File(filename, 'r') as new_file:
            np.testing.assert_array_equal(rints, new_file['/entry/tomo_entry/data/data'])
            np.testing.assert_array_equal(image_keys, new_file['/entry/tomo_entry/data/image_key'])
            np.testing.assert_array_almost_equal(angles, new_file['/entry/tomo_entry/data/rotation_angle'])
            self.assertEqual(new_file['/entry/tomo_entry/data/rotation_angle'].attrs['axis'], 0)
            np.testing.assert_array_almost_equal(translations[:, 0], new_file['/entry/tomo_entry/sample/x_translation'])
            np.testing.assert_array_almost_equal(translations[:, 1], new_file['/entry/tomo_entry/sample/y_translation'])
            np.testing.assert_array_almost_equal(translations[:, 2], new_file['/entry/tomo_entry/sample/z_translation'])
            self.checkStringEqual(new_file['/entry/tomo_entry/sample/name'][()], b'Sample')
            self.checkStringEqual(new_file['/entry/tomo_entry/definition'][()], b'NXtomo')
            self.checkStringEqual(new_file['/entry/tomo_entry/title'][()], b'Title')

    def testExtractAngles(self):
        filename = os.path.join(self.test_dir, 'angles.txt')
        angles = (0, 45, 90, 135, 180)
       
        np.savetxt(filename, angles)
        np.testing.assert_array_almost_equal(writer.extract_angles(filename), angles)

        log_data = """Sun Feb 10 00:22:04 2019   Projection:  0  angle: 0.0   Monitor 3 before:  4577907   Monitor 3 after:  4720271
                      Sun Feb 10 00:22:37 2019   Projection:  1  angle: 45.0   Monitor 3 before:  4729337   Monitor 3 after:  4871319
                      Sun Feb 10 00:23:10 2019   Projection:  2  angle: 90.0   Monitor 3 before:  4879923   Monitor 3 after:  5022689
                      Sun Feb 10 00:23:43 2019   Projection:  3  angle: 135.0   Monitor 3 before:  5031423   Monitor 3 after:  5172216
                      Sun Feb 10 00:24:16 2019   Projection:  4  angle: 180.0   Monitor 3 before:  5180904   Monitor 3 after:  5322691"""
        with open(filename, 'w') as logfile:
            logfile.write(log_data)
        
        np.testing.assert_array_almost_equal(writer.extract_angles(filename), angles)

        log_data = """Sun Feb 10 00:22:04 2019,Projection:0,angle:0.0,Monitor 3 before:4577907,Monitor 3 after:4720271
                      Sun Feb 10 00:22:37 2019,Projection:1,angle:45.0,Monitor 3 before:4729337,Monitor 3 after:4871319
                      Sun Feb 10 00:23:10 2019,Projection:2,angle:90.0,Monitor 3 before:4879923,Monitor 3 after:5022689
                      Sun Feb 10 00:23:43 2019,Projection:3,angle:135.0,Monitor 3 before:5031423,Monitor 3 after:5172216
                      Sun Feb 10 00:24:16 2019,Projection:4,angle:180.0,Monitor 3 before:5180904,Monitor 3 after:5322691"""
                      
        with open(filename, 'w') as logfile:
            logfile.write(log_data)
        
        np.testing.assert_array_almost_equal(writer.extract_angles(filename), angles)

    def testPrepareImages(self):
        self.assertRaises(ValueError, writer.prepare_images, [], '')

        proj_dir = tempfile.mkdtemp(dir=self.test_dir)
        
        proj_paths = self.createFakeImages(proj_dir, 2)
        names, keys, angles = writer.prepare_images([0, 170], proj_dir)
        self.assertEqual(names, proj_paths)
        self.assertEqual(keys, [0, 0])
        self.assertEqual(angles, [0, 170])

        self.assertRaises(ValueError, writer.prepare_images, [0, 85, 170], proj_dir)

        dark_before_dir = tempfile.mkdtemp(dir=self.test_dir)
        self.assertRaises(FileNotFoundError, writer.prepare_images, [0, 170], proj_dir, dark_before_dir)
        
        db_paths = self.createFakeImages(dark_before_dir, 2)
        names, keys, angles = writer.prepare_images([0, 170], proj_dir, dark_before_dir)
        self.assertEqual(names, [*db_paths, *proj_paths])
        self.assertEqual(keys, [2, 2, 0, 0])
        self.assertEqual(angles, [0, 0, 0, 170])

        half_circle_dir = tempfile.mkdtemp(dir=self.test_dir)
        hc_paths = self.createFakeImages(half_circle_dir)
        names, keys, angles = writer.prepare_images([0, 170], proj_dir, dark_before_dir, half_circle=half_circle_dir)
        self.assertEqual(names, [*db_paths, *proj_paths, *hc_paths])
        self.assertEqual(keys, [2, 2, 0, 0, 0])
        self.assertEqual(angles, [0, 0, 0, 170, 180])
        
        flat_after_dir = tempfile.mkdtemp(dir=self.test_dir)
        fa_paths = self.createFakeImages(flat_after_dir, 2)
        names, keys, angles = writer.prepare_images([0, 170], proj_dir, dark_before_dir, half_circle=half_circle_dir, 
                                                    flat_after=flat_after_dir)
        self.assertEqual(names, [*db_paths, *proj_paths, *hc_paths, *fa_paths])
        self.assertEqual(keys, [2, 2, 0, 0, 0, 1, 1])
        self.assertEqual(angles, [0, 0, 0, 170, 180, 0, 0])

        flat_before_dir = tempfile.mkdtemp(dir=self.test_dir)
        fb_paths = self.createFakeImages(flat_before_dir, 3)
        dark_after_dir = tempfile.mkdtemp(dir=self.test_dir)
        da_paths = self.createFakeImages(dark_after_dir, 3)
        rot_angles = os.path.join(self.test_dir, 'angles.txt')
        np.savetxt(rot_angles, (0, 170))
        names, keys, angles = writer.prepare_images(rot_angles, proj_dir, dark_before_dir, flat_before_dir, half_circle_dir, 
                                                    flat_after_dir, dark_after_dir)
        self.assertEqual(names, [*db_paths, *fb_paths, *proj_paths, *hc_paths, *fa_paths, *da_paths])
        self.assertEqual(keys, [2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2])
        self.assertEqual(angles, [0, 0, 0, 0, 0, 0, 170, 180, 0, 0, 0, 0, 0])
        
        np.savetxt(rot_angles, (0, 45, 90, 135, 180))
        self.assertRaises(ValueError, writer.prepare_images, rot_angles, proj_dir)

    def testGetTiffs(self):
        self.assertEqual(writer.get_tiffs(self.test_dir), [])

        tempfile.mkdtemp(dir=self.test_dir)  # add empty folder to dir

        try:
            fd, _ = tempfile.mkstemp(dir=self.test_dir, suffix='.txt') # add text file to dir
        finally:
            os.close(fd)
        
        self.file_descriptors.append(fd)
        self.assertEqual(writer.get_tiffs(self.test_dir), [])

        paths = self.createFakeImages(self.test_dir, 20)
        self.assertEqual(writer.get_tiffs(self.test_dir), paths)

    @mock.patch('nxtomowriter.writer.add_nxtomo_entry', autospec=True)
    def testSaveTomoToNexus(self, add_func):
        proj_dir = tempfile.mkdtemp(dir=self.test_dir)
        _ = self.createFakeImages(proj_dir, 5)

        filename = os.path.join(self.test_dir, 'random.nxs')
        rot_angles = (0.0, 180.0)
        writer.save_tomo_to_nexus(filename, rot_angles, proj_dir)
        self.assertEqual(add_func.call_args[0][0], filename)
        add_func.assert_called()
        self.assertIsNone(add_func.call_args[0][4])
        
        try:
            fd, filename = tempfile.mkstemp(dir=self.test_dir, suffix='.nxs')
        finally:
            os.close(fd)

        copy_name = f'{filename[:-4]}_with_tomo.nxs'
        self.assertRaises(ValueError, writer.save_tomo_to_nexus, filename, rot_angles, proj_dir, 
                          make_copy=False, open_beam_position=(10, 10, 10))

        flat_after_dir = tempfile.mkdtemp(dir=self.test_dir)
        _ = self.createFakeImages(flat_after_dir, 2)

        add_func.reset_mock()
        out_filename = writer.save_tomo_to_nexus(filename, rot_angles, proj_dir, flat_after=flat_after_dir, 
                                                 make_copy=False, projection_position=(10, 10, 10))
        
        trans = np.tile([10, 10, 10], (7, 1))
        add_func.assert_called()
        np.testing.assert_array_almost_equal(add_func.call_args[0][4], trans)
        self.assertEqual(add_func.call_args[0][5], 1)
        self.assertEqual(filename, out_filename)
        self.assertFalse(os.path.isfile(copy_name))

        add_func.reset_mock()
        rot_angles = os.path.join(self.test_dir, 'angles.txt')
        np.savetxt(rot_angles, (0, 45, 90, 135, 180))
        out_filename = writer.save_tomo_to_nexus(filename, rot_angles, proj_dir, flat_after=flat_after_dir, 
                                                 rotation_axis=0, make_copy=True, open_beam_position=(12, 11, 9), 
                                                 projection_position=(10, 10, 10))
        
        trans[-2:, :] = [12, 11, 9]
        add_func.assert_called()
        np.testing.assert_array_almost_equal(add_func.call_args[0][4], trans)
        self.assertEqual(add_func.call_args[0][5], 0)
        self.assertEqual(copy_name, out_filename)
        self.assertTrue(os.path.isfile(copy_name))
    
    def testFileSortKey(self):
        list_of_strings = ['home/test_034', 'home/test_031', 'home/test_033', 'home/test_032']
        sorted_list = sorted(list_of_strings, key=writer.filename_sorting_key)
        self.assertListEqual(sorted_list, ['home/test_031', 'home/test_032', 'home/test_033', 'home/test_034'])

        list_of_strings = ['C:/home/recon001', 'C:/home/recon010', 'C:/home/recon008', 'C:/home/recon004']
        sorted_list = sorted(list_of_strings, key=writer.filename_sorting_key)
        self.assertListEqual(sorted_list,
                             ['C:/home/recon001', 'C:/home/recon004', 'C:/home/recon008', 'C:/home/recon010'])
    
    @staticmethod
    def createFakeImages(dir, count=1):
        paths = []
        
        for i in range(count):
            ext = '.TIF' if i % 2 else '.tiff'
            try:
                fd, tp =tempfile.mkstemp(dir=dir)    
            finally:
                os.close(fd)
            new_name = os.path.join(os.path.dirname(tp), f'test_file_{i}{ext}')
            os.rename(tp, new_name)
            paths.append(new_name)
        return paths

    def checkStringEqual(self, first, second):
        self.assertEqual(first, second)

if __name__ == '__main__':
    unittest.main()
