# read the Kaggle competition files
function [training_images, training_labels, validation_images, validation_labels, test_images, test_labels]  = ...
	get_Kaggle_data( train_filename = 'kaggle_train.csv', ...
				 test_filename  = 'kaggle_test.csv', ...
				 filepath	     = 'C:\Users\jjsm\Documents\MNIST NN project\' )																
	# inputs should be  <number of input vars> * <number of examples>
	# targets hould be  <number of examples> * < number of classes> 

	labelled_training_data = csvread( strcat( filepath, train_filename ) );
	test_data = csvread( strcat( filepath, test_filename ) );

	original_training_data = transpose( labelled_training_data( :, 2:785 ) ) / 255;
	original_training_labels = labels_to_vectors( transpose( labelled_training_data( :, 1 ) ) );
	
	original_test_data  = transpose( test_data ) /255;


	#######################################################################
	# FOR TEST PURPOSES:
	# divide the original test data into train/test/validation sets with labels

	breakpoint_train_val = 15000;
	breakpoint_val_train = 30000;

	training_images = original_training_data( :, 1:breakpoint_train_val );
	training_labels = original_training_labels( 1:breakpoint_train_val, : );

	validation_images = original_training_data( :, (breakpoint_train_val +1) : breakpoint_val_train );
	validation_labels = original_training_labels( (breakpoint_train_val +1) : breakpoint_val_train, :);

	test_images = original_training_data( :, (breakpoint_val_train+1):end );	 
	test_labels = original_training_labels( (breakpoint_val_train+1):end, :  );	


	#######################################################################

endfunction