# applies logistic function to each input
function output = logistic( input )

	output = 1 ./ ( 1 + exp( -input ) );

endfunction
