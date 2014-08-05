# get data and run nn with various parameters
function run_nn()

	# load data
	[a,b,c,d,e,f] = get_16_by_16_data();	

	# set parameters
	parameters.learning_mode = "mini-batch";
	parameters.mini_batch_size = 50;
	parameters.weight_decay_coef = 0.001;
	parameters.weight_decay_type = "L1";
	parameters.momentum = 0.9;
	parameters.learning_rate = 0.01;
	parameters.num_hidden_units = 100;
	parameters.num_classes = 10;
	parameters.max_steps = 10000;
	parameters.validation_frequency = 100;
	parameters.dropout_proportion = 0.5;
	parameters.bias = true;
	parameters.early_stopping = false;


	learning_rates = [0.1, 0.01, 0.001, 0.0001 ];

	# create models with different parameter values, then write results to file
	for i = 1:4
		parameters.learning_rate = learning_rates( i );
		[model, error, runtime] = nn(a,b,c,d,e,f,parameters);
		save( "-append", "results.txt", "error","runtime", "parameters" );
	end;
	
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
