# convert label vector back to single digit
function vector_of_single_digits = vectors_to_labels( matrix_of_class_probs )

	num_digits 				= size( matrix_of_class_probs, 1 );
	vector_of_single_digits	= zeros(num_digits, 1);
	
	# indexes start at 1 in Matlab, so mapping is between <position n+1> and <digit n>
	for label_idx = 1:num_digits
		prob_vector								= matrix_of_class_probs( label_idx, : );
		digit_position								= find( prob_vector  == max( prob_vector	 ) , 1) ;  # find max 1 element
		digit 										= digit_position-1;
		vector_of_single_digits( label_idx, 1 )	= digit;	
	end

endfunction
