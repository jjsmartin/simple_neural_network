# calculates gradient for input to hidden weight change
# TODO momentum, weight decay etc.

#  x --(w_ih)--> g --(logistic)--> h --(w_hc)--> z --(softmax)--> probs --(error function)--> E

function gradient = input_to_hidden_gradient( model, forward_pass_results, inputs, targets, weight_decay_coef )

	# results of differentiation
	# "z" is the *input* to the softmax, "h" is the output from the hidden layer, "g" is the input to the hidden layer, "wih" is the weights from the input to hidden layer
	dE_by_dz		= forward_pass_results.output_from_softmax - targets;		# v. small
	dz_by_dh		= model.hidden_to_class_weights;							# reasonably sized
	dh_by_dg		= forward_pass_results.output_from_hidden_units .* ( 1 - forward_pass_results.output_from_hidden_units );				# zero or very very small
	dg_by_dwih	= inputs;															# reasonably sized
		
	N = size( targets, 1 );

	# return gradient with weight decay term
	gradient 	= 1./N * ( dg_by_dwih *  ( ( dE_by_dz )  * transpose( dz_by_dh )  .* dh_by_dg ) ) + ( weight_decay_coef * model.input_to_hidden_weights ); 

# ret.input_to_hid = 1/N * ( transpose(transpose(c-t) * w_hy)  .* h.*(1-h) * transpose(x))  + (wd_coefficient * w_ih);   % using transpose to preserve chain order


endfunction



# Input to hidden gradient
#function gradient = input_to_hidden_gradient( model, forward_pass_results, inputs, targets )
#
#	N = size( targets, 1 );
#
#	c = forward_pass_results.output_from_softmax';
#	t = targets';
#	h = forward_pass_results.output_from_hidden_units';
#	x = inputs;
#	w_hy = model.hidden_to_class_weights';
#	
#	gradient = 1./N * ( transpose(transpose(c-t) * w_hy)  .* h.*(1-h) * transpose(x));  
#	gradient = transpose( gradient );
## ret.input_to_hid = 1/N * ( transpose(transpose(c-t) * w_hy)  .* h.*(1-h) * transpose(x));   % using transpose to preserve chain order
#
#endfunction
