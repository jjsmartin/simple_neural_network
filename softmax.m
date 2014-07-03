# softmax is exponential for each input, divided by the sum of exponentials, but we're dealing in terms of logs to keep things numerically stable
function class_probs = softmax( input )

	num_classes = size( input, 2 );

	# get class normalizer and repeat it so that it matches the input dimensions
	log_class_normalizer = log_sum_exp( input );			    
	log_class_normalizer_matching_input_dimensions 	= repmat( log_class_normalizer, 1, num_classes );	

	# equivalent to division by the normalizer, since we're dealing with logs
	log_class_probs	= input - log_class_normalizer_matching_input_dimensions;		

	# we calculated log( sum( exp( input ) ) ), for sake of numerical stability, but we want to return sum( exp( input ) )  		
	class_probs = exp( log_class_probs ); 												

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
