# takes a n*1 vector of values corresponding to a digit, and displays the corresponding digit image
function show_digit( image_vector )

	# get side length (we assume a square image)
	image_side_length	 = sqrt( size( image_vector, 1 ) ) ;

	# convert the vector to a matrix, and show it
	image_matrix = reshape(	image_vector, image_side_length, image_side_length );
	imshow (image_matrix);
	
endfunction