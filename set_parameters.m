function parameters = set_parameters()

	parameters.learning_mode = "mini-batch";
	parameters.mini_batch_size = 50;
	parameters.weight_decay_coef = 0.001;
	parameters.weight_decay_type = "L1";
	parameters.momentum = 0.9;
	parameters.learning_rate = 0.01;
	parameters.num_hidden_units = 100;
	parameters.num_classes = 10;
	parameters.max_steps = 1000;
	parameters.validation_frequency = 100;
	parameters.dropout_proportion = 0.5;
	parameters.bias = true;
	parameters.early_stopping = false;
	
endfunction



# 134 minutes, 3.5% error on MNIST data
#	learning_mode 	  		= "mini-batch"		
#	mini_batch_size   		= 50	
#	weight_decay_coef		= 0.0005
#	weight_decay_type		= "L1"
#	momentum 		  		= 0.9					
#	learning_rate 		  		= 0.01	
#	num_hidden_units 		= 300
#	num_classes 	  		= 10			
#	max_steps 		 		= 100000
#	validation_frequency		= 20
#	bias 				  		= true
#	early_stopping 	 		= false	
