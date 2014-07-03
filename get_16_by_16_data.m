# Reads 16*16 digit data (originally from Coursera NN course)
function [training_images, training_labels, validation_images, validation_labels, test_images, test_labels]  = get_16_by_16_data(	data_filename = 'data16.mat', ...
																																						filepath		 = 'C:\Users\jjsm\Documents\MNIST NN project\' )																
	# Load images and labels (note that this is a struct with everything folded up into it)
	images = load( strcat( filepath, data_filename ) );

	training_images 		= images.data.training.inputs;
	validation_images 	= images.data.validation.inputs;
	test_images 			= images.data.test.inputs;

	training_labels 		= transpose( images.data.training.targets );
	validation_labels 		= transpose( images.data.validation.targets );
	test_labels 			= transpose( images.data.test.targets );	
	
endfunction