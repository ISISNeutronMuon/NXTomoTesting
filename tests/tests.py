from contextlib import suppress
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
            tifffile.imsave(image_names[-1], rints[i])

        writer.add_nxtomo_entry(filename, image_names, image_keys, angles)

        with h5py.File(filename, 'r') as new_file:
            np.testing.assert_array_equal(rints, new_file['/entry/tomo_entry/data/data'])
            np.testing.assert_array_equal(image_keys, new_file['/entry/tomo_entry/data/image_key'])
            np.testing.assert_array_almost_equal(angles, new_file['/entry/tomo_entry/data/rotation_angle'])
            self.assertEqual(new_file['/entry/tomo_entry/data/rotation_angle'].attrs['axis'], 1)
            self.checkStringEqual(new_file['/entry/tomo_entry/sample/name'][()], '')
            self.checkStringEqual(new_file['/entry/tomo_entry/definition'][()], 'NXtomo')
            self.checkStringEqual(new_file['/entry/tomo_entry/title'][()], '')

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
            self.checkStringEqual(new_file['/entry/tomo_entry/sample/name'][()], 'Sample')
            self.checkStringEqual(new_file['/entry/tomo_entry/definition'][()], 'NXtomo')
            self.checkStringEqual(new_file['/entry/tomo_entry/title'][()], 'Title')

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
        names, keys, angles = writer.prepare_images([0, 170], proj_dir, dark_before_dir, flat_before_dir, half_circle_dir, 
                                                    flat_after_dir, dark_after_dir)
        self.assertEqual(names, [*db_paths, *fb_paths, *proj_paths, *hc_paths, *fa_paths, *da_paths])
        self.assertEqual(keys, [2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2])
        self.assertEqual(angles, [0, 0, 0, 0, 0, 0, 170, 180, 0, 0, 0, 0, 0])

    def testGetTiffs(self):
        self.assertEqual(writer.get_tiffs(self.test_dir), [])

        tempfile.mkdtemp(dir=self.test_dir)  # add empty folder to dir

        try:
            fd, _ = tempfile.mkstemp(dir=self.test_dir, suffix='.txt') # add text file to dir
        finally:
            os.close(fd)
        
        self.file_descriptors.append(fd)
        self.assertEqual(writer.get_tiffs(self.test_dir), [])

        paths = self.createFakeImages(self.test_dir, 2)
        self.assertEqual(writer.get_tiffs(self.test_dir), paths)

    @mock.patch('nxtomowriter.writer.add_nxtomo_entry', autospec=True)
    def testSaveTomoToNexus(self, add_func):
        proj_dir = tempfile.mkdtemp(dir=self.test_dir)
        _ = self.createFakeImages(proj_dir, 5)

        filename = os.path.join(self.test_dir, 'random.nxs')
        rot_angles = [0, 45, 90, 135, 180]
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
        writer.save_tomo_to_nexus(filename, rot_angles, proj_dir, flat_after=flat_after_dir, 
                                  make_copy=False, projection_position=(10, 10, 10))
        
        trans = np.tile([10, 10, 10], (7, 1))
        add_func.assert_called()
        np.testing.assert_array_almost_equal(add_func.call_args[0][4], trans)
        self.assertEqual(add_func.call_args[0][5], 1)
        self.assertFalse(os.path.isfile(copy_name))

        add_func.reset_mock()
        writer.save_tomo_to_nexus(filename, rot_angles, proj_dir, flat_after=flat_after_dir, rotation_axis=0, 
                                  make_copy=True, open_beam_position=(12, 11, 9), projection_position=(10, 10, 10))
        
        trans[-2:, :] = [12, 11, 9]
        add_func.assert_called()
        np.testing.assert_array_almost_equal(add_func.call_args[0][4], trans)
        self.assertEqual(add_func.call_args[0][5], 0)
        self.assertTrue(os.path.isfile(copy_name))

    @staticmethod
    def createFakeImages(dir, count=1):
        paths = []
        
        for i in range(count):
            ext = '.TIF' if i % 2 else '.tiff'
            try:
                fd, tp =tempfile.mkstemp(dir=dir, prefix=f'{i}', suffix=ext)
                paths.append(tp)
            finally:
                os.close(fd)

        return paths

    def checkStringEqual(self, first, second):
        with suppress(AttributeError):
            first = first.decode('utf-8')
        
        with suppress(AttributeError):
            second = second.decode('utf-8')

        self.assertEqual(first, second)

if __name__ == '__main__':
    unittest.main()
