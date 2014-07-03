# takes a model (= weights) and input data, displays a confusion matrix of actual and predicted classification
function cm = confusion_matrix( targets, results, num_classes=10 )	

	# set up a matrix of the right dimensionsrows will be targets, columns actual
	cm = zeros( num_classes, num_classes);		

	# the correct number and the predicted number give us a row and column in the matrix. Increment that number
	num_digits = size( targets, 1 );
	for ( digit_idx = 1:num_digits )
		model_digit = results( digit_idx, 1)  + 1; 	# +1 because digit	 0 maps to position 1 etc.
		target_digit = targets( digit_idx, 1)  + 1;
		cm( target_digit, model_digit ) += 1;
		
	end

	# TODO some alternative to using 0 for very top left bit where row/col names overlap
	cm_with_headers = zeros( num_classes+1, num_classes+1);
	cm_with_headers(1,:) =  [0 0 1 2 3 4 5 6 7 8 9];
   	cm_with_headers(:,1) =  [0 0 1 2 3 4 5 6 7 8 9];
   	cm_with_headers( 2:end,2:end ) = cm;
   	cm = cm_with_headers;
   	
endfunction
