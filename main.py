from gooey import GooeyParser, Gooey
import nxtomowriter as ntw


@Gooey(program_name=f'NxTomo Writer v{ntw.version}', progress_regex=r'^(\d+)%', hide_progress_msg=True)
def main():
    # Required
    parser = GooeyParser(description='An application to writes tomography data to a nexus file')
    required_args = parser.add_argument_group("Required Arguments") 
    required_args.add_argument('filename', type=str, help='path of nexus file.', widget='FileChooser', 
                               gooey_options={'full_width': True, 'message': 'Select Nexus File', 'wildcard': 'Nexus file (*.nxs)|*.nxs'})
    required_args.add_argument('projection_path', type=str, help='directory of projection images', 
                               widget='DirChooser', gooey_options={'full_width': True, 'message': 'Select Folder with Projection Images'})

    stuff = required_args.add_mutually_exclusive_group(required=True, gooey_options={'initial_selection': 0})
    stuff.add_argument('--log_file', metavar='Read rotation angles from file', type=str, default='', 
                       help='path of csv file containing rotation angles (in degrees)', 
                       widget='FileChooser', gooey_options={'full_width': True, 'message': 'Select File with Rotation Angles'})
    stuff.add_argument('--angles', metavar='Enter start and stop angle', type=str, default='0.0, 0.0', 
                       help='first and last angle in equally spaced range', gooey_options={'full_width': True, 
                       'validator': {'test': 'len([float(x) for x in user_input.split(",")]) == 2', 
                                    'message': 'Must be comma seperated start and stop angle'}})

    # Optional   
    optional_args = parser.add_argument_group("Optional Arguments") 
    optional_args.add_argument('--make_copy', action='store_true', default=True, 
                               help='indicates tomo entry should be added to copy of the nexus file')
    optional_args.add_argument('--rotation_axis', choices=['horizontal', 'vertical'], default='vertical', help='axis of rotation')
    optional_args.add_argument('--dark_before_path', type=str, help='directory of dark before images', default='', 
                               widget='DirChooser', gooey_options={'full_width': True, 'message': 'Select Folder with Dark Before Images'})
    optional_args.add_argument('--flat_before_path', type=str, help='directory of flat before images', default='', 
                               widget='DirChooser', gooey_options={'full_width': True, 'message': 'Select Folder with Flat Before Images'})
    optional_args.add_argument('--dark_after_path', type=str, help='directory of dark after images', default='', 
                               widget='DirChooser', gooey_options={'full_width': True, 'message': 'Select Folder with Dark After Images'})
    optional_args.add_argument('--flat_after_path', type=str, help='directory of flat after images', default='', 
                               widget='DirChooser', gooey_options={'full_width': True, 'message': 'Select Folder with Flat After Images'})
    optional_args.add_argument('--half_circle_path', type=str, help='directory of 180 degree image', default='', 
                               widget='DirChooser', gooey_options={'full_width': True, 'message': 'Select Folder with 180 Degree Image'})

    open_beam_position_group = parser.add_argument_group('Open beam position', description='positioner XYZ (in mm) value when imaging open beam', 
                                                          gooey_options={'columns': 3, 'show_border': True})
    open_beam_position_group.add_argument('--open_beam_position_x', type=float, help='positioner X (in mm) value when imaging open beam', 
                                           gooey_options={'show_help':False, 'validator': {'test': 'float(user_input)', 'message': 'Must be a number'}})
    open_beam_position_group.add_argument('--open_beam_position_y', type=float, help='positioner Y (in mm) value when imaging open beam', 
                                           gooey_options={'show_help':False, 'validator': {'test': 'float(user_input)', 'message': 'Must be a number'}})
    open_beam_position_group.add_argument('--open_beam_position_z', type=float, help='positioner Z (in mm) value when imaging open beam', 
                                           gooey_options={'show_help':False, 'validator': {'test': 'float(user_input)', 'message': 'Must be a number'}})

    projection_position_group = parser.add_argument_group('Projection position', description='positioner XYZ (in mm) value when imaging sample', 
                                                          gooey_options={'columns': 3, 'show_border': True})
    projection_position_group.add_argument('--projection_position_x', type=float, help='positioner X (in mm) value when imaging sample', 
                                           gooey_options={'show_help':False, 'validator': {'test': 'float(user_input)', 'message': 'Must be a number'}})
    projection_position_group.add_argument('--projection_position_y', type=float, help='positioner Y (in mm) value when imaging sample', 
                                           gooey_options={'show_help':False, 'validator': {'test': 'float(user_input)', 'message': 'Must be a number'}})
    projection_position_group.add_argument('--projection_position_z', type=float, help='positioner Z (in mm) value when imaging sample', 
                                           gooey_options={'show_help':False, 'validator': {'test': 'float(user_input)', 'message': 'Must be a number'}})
    
    args = parser.parse_args() 
    angles = ''
    if args.log_file:
        angles = args.log_file
    else:
        angles = tuple([float(x) for x in args.angles.split(",")])
    

    rotation_axis = 1 if args.rotation_axis=='vertical' else 0
    
    if args.open_beam_position_x is None or args.open_beam_position_y is None or args.open_beam_position_z is None:
        open_beam_position = ()
    else:
        open_beam_position = (args.open_beam_position_x, args.open_beam_position_y , args.open_beam_position_z)
                                          
    if args.projection_position_x is None or args.projection_position_y is None or args.projection_position_z is None:
        projection_position = ()
    else:
        projection_position = (args.projection_position_x, args.projection_position_y , args.projection_position_z)
    
    ntw.use_gui(True)
    print('Writing data to file ...')
    out_filename = ntw.save_tomo_to_nexus(args.filename, angles, args.projection_path, dark_before=args.dark_before_path,
                                          flat_before=args.flat_before_path, flat_after=args.flat_after_path,
                                          dark_after=args.dark_after_path, open_beam_position=open_beam_position, 
                                          projection_position=projection_position, rotation_axis=rotation_axis, 
                                          make_copy=args.make_copy)

    print(f'Data was saved successfully to {out_filename}')


if __name__ == '__main__':
    main()