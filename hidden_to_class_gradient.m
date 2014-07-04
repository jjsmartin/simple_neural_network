# calculates gradient for hidden to class weight change
#  x --(w_ih)--> g --(logistic)--> h --(w_hc)--> z --(softmax)--> probs --(error function)--> E

function gradient = hidden_to_class_gradient( model, forward_pass_results, targets, weight_decay_coef )

	# results of differentiation
	# ("z" is the *input* to the softmax, "whc" is the weights between hidden and class (aka softmax) layer)
	dE_by_dz		= forward_pass_results.output_from_softmax - targets;
	dz_by_dwhc	= forward_pass_results.output_from_hidden_units;   
	
	N = size( targets, 1 );

	# return gradient with weight decay term
	gradient 	= 1./N * transpose( dz_by_dwhc ) * dE_by_dz + ( weight_decay_coef * model.hidden_to_class_weights ); 
	
endfunction



