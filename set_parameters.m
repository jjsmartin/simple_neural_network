function parameters = set_parameters()

	parameters = [ "mini-batch",		# learning_mode 	  		
				   50,  					# mini_batch_size  
				   0.001,			     # weight_decay_coef
				  "L1"				     # weight_decay_type	
				   0.9,				     # momentum 				
				   0.01,				# learning_rate 
				   100,					# num_hidden_units 
				   10, 					# num_classes 	
				   1000,				# max_steps 
				   100,				     # validation_frequency
				   0.5				     # dropout_proportion	
				   true,		 	     	# bias
				   false	 ];			     # early_stopping 	

	

endfunction