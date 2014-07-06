# reads model details (parameters, size, weights)
function model = load_model( 	model_name, ...
										filepath   		= 'C:\Users\jjsm\Documents\MNIST NN project\'  )


   	filepath_into_model_folder = strcat( filepath, model_name, '\' );

	# read weight matrix dimensions
	parameter_file_id = fopen( strcat( filepath_into_model_folder , 'parameters.txt' ), 'r' );

	params = strsplit( fscanf( parameter_file_id, '%s' ), ',' );  # reads a line and splits at commas

	num_input_units 		= str2num( params{ 1 } ); # strsplit returns a cell array hence curly brackets
	num_hidden_units 	= str2num( params{ 2 } );
	num_output_units 	= str2num( params{ 3 } );

	fclose( parameter_file_id );

	# get saved weight data and put it into the model struct
	model = struct( "input_to_hidden_weights", [ ], "hidden_to_class_weights", [ ] );
	
	#  input to hidden weights
	i2h_file_id  = fopen( strcat( filepath_into_model_folder , 'i2h.txt' ), 'r' );
	
	model.input_to_hidden_weights = fscanf( i2h_file_id, '%f', [ num_input_units, num_hidden_units ] );
	fclose( i2h_file_id );
	
	#  output to hidden weights
	h2c_file_id = fopen( strcat( filepath_into_model_folder , 'h2c.txt' ), 'r' );
	model.hidden_to_class_weights = fscanf( h2c_file_id, '%e', [ num_hidden_units, num_output_units] );
	fclose( h2c_file_id );

   
endfunction