# carries out a forward pass over the network, and returns results from each step
# note that we specify whether it's a training, validation or test pass because of different behaviour when we do dropout
function results = forward_pass( training_input, model, dropout_proportion, pass_type )
	
	results.input_to_hidden_units 		= transpose( training_input ) * model.input_to_hidden_weights; 	
	results.output_from_hidden_units	= logistic( results.input_to_hidden_units );

	# optionally do dropout (=remove each hidden unit with a certain probability)
	if dropout_proportion > 0.0
		# if we're doing dropout on a training pass, set a proportion of the hidden units to zero (i.e. remove them from the model)
		if strcmp( pass_type, "training" )
			results.output_from_hidden_units = dropout ( results.output_from_hidden_units, dropout_proportion );
		#  if it's a test or a validation forward pass, just multiply the outgoing weights of *all* the hidden units by dropout_proportion	
		elseif strcmp( pass_type, "validation" ) || strcmp( pass_type, "test" )
			results.output_from_hidden_units = results.output_from_hidden_units * ( 1- dropout_proportion );
		end
	end

	results.input_to_softmax			= results.output_from_hidden_units * model.hidden_to_class_weights;	# result is <number of examples> * <number of classes>
	results.output_from_softmax 		= softmax( results.input_to_softmax );										

endfunction


# dropout removes (=sets to zero) a certain proportion of hidden units
function retained_unit_activations = dropout( unit_activations, dropout_proportion )

		num_cases 					= size( unit_activations, 1 );
		num_units 					= size( unit_activations, 2);
		retained_units	 			= randi( 2, 1, num_units ) - 1;   # subtract 1 to get 0-1 values
		retained_units_repeated 	= repmat( retained_units, num_cases, 1 );	# match dimensions of hidden units
		retained_unit_activations	= unit_activations .* retained_units_repeated;

endfunction