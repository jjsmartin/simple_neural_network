# adds bias to data - i.e. fixed 
function data_with_bias = add_bias( data )

	# bias values are a new first column, consisting entirely of 1s
	bias_value_row	= ones( 1, size( data, 2 )  ); 
	data_with_bias	= cat( 1, bias_value_row, data );
	
endfunction