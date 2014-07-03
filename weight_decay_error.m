# calculates weight decay error i.e. sum of squared weights times a specified coefficient
function wd_error = weight_decay_error( model, weight_decay_coef )

	sum_of_squared_weights	= sum( ( model.input_to_hidden_weights .^2) (:) ) + sum(  (model.hidden_to_class_weights .^2 )(:) );
  	wd_error 					   	= sum_of_squared_weights * weight_decay_coef;
  	
endfunction