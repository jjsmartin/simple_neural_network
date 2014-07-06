# take a model, some test input and corresponding test labels. Return predicted digits
function predictions = make_predictions( model, test_input, dropout_proportion, bias= true )


	if bias
		test_input	= add_bias( test_input );
	end
	
	forward_pass_results = forward_pass( test_input, model, dropout_proportion, "test" ) ;
	predictions 			 =  vectors_to_labels( forward_pass_results.output_from_softmax );
	
endfunction