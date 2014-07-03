# Reads MNIST data as downloaded from http://yann.lecun.com/exdb/mnist/ and converts from the original binary format
# returns data partitioned into training, validation and test data
# uses functions loadMNISTImages.m and loadMNISTLabels.m from http://deeplearning.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset

function [training_images, training_labels, validation_images, validation_labels, test_images, test_labels] = get_MNIST_data( ...
																													train_images_filename	= 'train-images.idx3-ubyte', ...
																													train_labels_filename 	= 'train-labels.idx1-ubyte', ...
																													filepath				 	= 'C:\Users\jjsm\Documents\MNIST NN project\' ,...
																													training_proportion = 0.6,...
																													validation_proportion = 0.2,...
																													test_proportion = 0.2)																
	
	# NOTE I can't get test image file to work (possibly a Windows 64 bit issue), so I'm creating a test set from the training data

	# Load training (inclding validation) images and labels
	images 	= loadMNISTImages(strcat( filepath, train_images_filename ) );
	labels	= loadMNISTLabels(strcat( filepath, train_labels_filename) );

	#  partition intro training / validation / test sets
	# NOTE that labels are returned as one-hot vectors
	num_images	= size( images, 2 );

	train_start_point 		 = 1;
	train_end_point 		 = floor( num_images * training_proportion );
	validation_start_point = train_end_point+1;
	validation_end_point  = train_end_point + floor( num_images * validation_proportion );
	test_start_point 		 = validation_end_point + 1;
	test_end_point 		 = validation_end_point + floor( num_images * test_proportion ); 
	
	training_range 	= train_start_point : train_end_point;
	validation_range	= validation_start_point : validation_end_point;
	test_range 		= test_start_point : test_end_point;

	training_images = images( :, training_range );
	training_labels   = labels_to_vectors( transpose( labels( training_range ) ) );
	
	validation_images = images( :, validation_range );
	validation_labels 	 = labels_to_vectors( transpose( labels(  validation_range ) ) );
	
	test_images = images( :, test_range );		
	test_labels   = labels_to_vectors( transpose( labels( test_range ) ) );

endfunction
