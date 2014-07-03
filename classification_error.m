# takes targets and actual digit classifications, returns percentage accurately classified
function error_rate = classification_error( targets, results )

	score = 0;
	num_targets = size( targets, 1 );
	
   	for digit_idx = 1:num_targets
		if ( targets( digit_idx, 1 ) ) == ( results( digit_idx, 1 ) ) 
			score += 1;
		end
   	end
   	error_rate = (1 - (score ./ num_targets) ) * 100;
   	
endfunction
