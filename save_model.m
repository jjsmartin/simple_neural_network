# writes model details into a new folder that will contain one file for 
# each of: input to hidden weights, hidden to class weights
function save_model( 	model, ...
							model_name, ...
							filepath   		= 'C:\Users\jjsm\Documents\MNIST NN project\'  )


	# create a folder for the model files
	mkdir( filepath, model_name );
	filepath_into_model_folder = strcat( filepath, model_name, '\' );

	# write input to hidden weights
	i2h_file_id  = fopen( strcat( filepath_into_model_folder , 'i2h.txt' ), 'wt' );
	fprintf( i2h_file_id, '%e ', model.input_to_hidden_weights );
	fclose( i2h_file_id );
	
	# write output to hidden weights
	h2c_file = fopen( strcat( filepath_into_model_folder , 'h2c.txt' ), 'wt' );
	fprintf( h2c_file, '%e', model.hidden_to_class_weights );
	fclose( h2c_file );

	# write model parameters to another file
	num_input_units 		= size( model.input_to_hidden_weights, 1 ); # includes bias weight
	num_hidden_units 	= size( model.input_to_hidden_weights, 2 ); 
	num_output_units 	= size( model.hidden_to_class_weights, 2 );

	params = [ num_input_units, num_hidden_units, num_output_units ];
	
	parameter_file = fopen( strcat( filepath_into_model_folder , 'parameters.txt' ), 'wt' );
	fprintf( parameter_file, '%d,', params );
	fclose( parameter_file );

	
endfunction