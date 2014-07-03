# convert label column vector to matrix of row vectors in "one-hot" form e.g. "3" becomes "0 0 0 1 0 0 0 0 0 0"
# NOTE: indexes start at 1 in Matlab, so mapping is between <digit n> and <position n+1>
function class_probs = labels_to_vectors( vector_of_digits, num_classes = 10)

	num_labels	= size( vector_of_digits, 2 );
	class_probs	= zeros(  num_labels, num_classes );

	for label_idx = 1:num_labels
		class_probs( label_idx , vector_of_digits( label_idx ) + 1  ) = 1;
	end
	
endfunction
