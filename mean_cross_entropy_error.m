# calculates cross-entropy error (appropriate with softmax) in a numerically stable way
# softmax output for each input is exp(x1) / [ exp(x1) + exp(x2) + ...+ exp(x10) ]
# but we convert everything into logs, for the sake of numerical stability
function mce = mean_cross_entropy_error( output_from_softmax, target_vectors )

	num_classes = size( output_from_softmax, 2 );

	# the softmax outputs are basically log probabilities, so subtracting the total from each one is equivalent to dividing by a normalizer
	log_exp_class_normalizers	= log_sum_exp( output_from_softmax );	
	log_exp_class_prob			= output_from_softmax - repmat( log_exp_class_normalizers, 1, num_classes ); 

	cross_entropy_error = -sum( log_exp_class_prob .* target_vectors, 1 );   

	# return the mean error over all cases
	mce = mean( cross_entropy_error );

endfunction



# computes log( sum( exp( input ), 2 ) ) in a numerically stable way
function results = log_sum_exp( input )

	# max over each input column
  	maximum_for_each_case = max( input, [ ] , 2 );	# seem to need the empty array in position 2 to explicitly specify dimension in position 3	

  	# repeat the maximums to match the original input dimensions
  	num_classes = size( input, 2 );
	maximums_repeated = repmat( maximum_for_each_case, 1, num_classes );	
	results = log( sum( exp( input - maximums_repeated  ), 2 )  ) + maximum_for_each_case;

end
