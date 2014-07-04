# calculates weight decay error i.e. sum of squared weights times a specified coefficient
# Option for L1 weight decay (=sum of absolute values of weights) or L2 (=sum of squared weights)
#	L1 tends to push some weights to zero with others being large
#	L2 tends to push all weights to smaller values
function wd_error = weight_decay_error( model, weight_decay_coef, weight_decay_type )

	
	if strcmp( weight_decay_type, "L1")
		sum_abs_weights = sum( abs( model.input_to_hidden_weights ) (:)  ) + sum( abs( model.hidden_to_class_weights ) (:)  );
		wd_error 			  = sum_abs_weights * weight_decay_coef;
	
	elseif strcmp( weight_decay_type, "L2")
		sum_of_squared_weights	= sum( ( model.input_to_hidden_weights .^2) (:) ) + sum(  (model.hidden_to_class_weights .^2 )(:) );
	  	wd_error 					   	= sum_of_squared_weights * weight_decay_coef;
	  	
  	end

  	
endfunction