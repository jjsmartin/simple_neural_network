# displays an image of the input weights for each hidden unit
function inspect_hidden_layer( model )

	weights = model.input_to_hidden_weights;
	
	num_hidden_units = size( weights, 2 );

	 # note that we skip over the bias weight (=the first value in each column here).
	for hidden_unit_num = 1:num_hidden_units 
		show_digit( weights( 2:end, hidden_unit_num ) );
		pause( 0.1 )
	end
	
endfunction