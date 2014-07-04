# Feed-forward neural network with one hidden layer and backprop


# [a,b,c,d,e,f] = get_16_by_16_data();	
# model = nn(a,b,c,d,e,f);

# [g,h,i,j,k,l] = get_MNIST_data();  
# model = nn(g,h,i,j,k,l);



# TODO online learning (maybe)
# TODO dropout / regularization generally (as well as weight decay, early-stopping, which I have)
# TODO try this syntax to allow defaults:	model = nn(a,b,c,d, ~, ~, ~, ~, ~, ~, ~, ~, ~ );
# TODO L1, L2 weight decay?
# TODO other things as listed http://yann.lecun.com/exdb/mnist/


########################################################################################################



function model = nn( training_input, training_targets, validation_input, validation_targets, test_input, test_targets )

	# FOR BOTH ORIGINAL MNIST AND 16*16 DIGITS:
	# inputs are <number of input vars> * <number of examples>
	# targets are <number of examples> * < number of classes> 

	# parameters
	learning_mode 	  		= "mini-batch"		
	mini_batch_size   		= 200	
	weight_decay_coef		= 0.02
	weight_decay_type		= "L2"
	momentum 		  		= 0.9					
	learning_rate 		  		= 0.01			# NOTE large batch size should mean smaller learning rate. 
	num_hidden_units 		= 50
	num_classes 	  		= 10			
	max_steps 		 		= 1000
	validation_frequency		= 10
	bias 				  		= true
	early_stopping 	 		= true	

	# optionally add bias to the inputs
	if bias
		training_input		= add_bias( training_input );
		validation_input	= add_bias( validation_input );
		test_input			= add_bias( test_input );
	end

	# convenient to establish these values here (note we do this *after* adding the bias term)
	num_input_units		= size( training_input, 1 );
	num_training_cases	= size( training_input, 2 );

	# initialize weights
	model	= initialize_weights( num_input_units, num_hidden_units,  num_classes );

	# initialize vectors for recording errors
	training_error_record = [ ];
	validation_error_record = [ ];

	# initialize previous weight changes to zero (we use these with the momentum term)
	prev_i2h_update  = zeros( num_input_units, num_hidden_units );
	prev_h2c_update = zeros( num_hidden_units, num_classes );

	# we'll report how long the learning process took to finish
	start = clock;

	# main loop
	for t = 1:max_steps

		## SELECT BATCH TRAINING CASES
		if strcmp( learning_mode, "mini-batch" )
		
			# select a random subset of the data (without replacement), of the specified mini-batch size
			random_case_indices = randperm( num_training_cases )(1:mini_batch_size);
			training_input_batch   	= training_input( : , random_case_indices  );
			training_targets_batch	= training_targets( random_case_indices, : );

		elseif strcmp( learning_mode, "batch" )

			# use all training examples as the "batch"
			training_input_batch   	= training_input;
			training_targets_batch	= training_targets;
		
		end


		## EARLY STOPPING (if enabled)
		if ( early_stopping ) && ( mod( t, validation_frequency ) == 0 ) 

			# (the validation forward pass and error calc are wrapped up in a single function, since we don't need intermediate results)
			# note that we pad out the validation error record with multiple copies of the error, so it has the same number of values as training error
			val_error 					= validation( model, validation_input, validation_targets, weight_decay_coef, weight_decay_type );
			validation_error_record 	= [ validation_error_record, repmat( val_error, 1, validation_frequency )	 ];

			# if the validation error has increased since last time, stop. Also revert the model to the last model before early stopping
			# (note short circuit operator "&&", so we don't try to check for t=0)
			if ( length( validation_error_record ) >= 2 ) && ( validation_error_record( end )  > validation_error_record( end-1 ) )
				model = model_before_early_stopping;
				break
			else
				model_before_early_stopping = model;
			end	
				
		end

		## FORWARD PASS
		training_forward_pass_results =  forward_pass( training_input_batch, model );
		
		## ERROR CALCULATION
		training_error = mean_cross_entropy_error( training_forward_pass_results.output_from_softmax, training_targets_batch )...
						+ weight_decay_error( model, weight_decay_coef, weight_decay_type );
		training_error_record	= [ training_error_record, training_error ];

		## WEIGHT UPDATES
		h2c_gradient			= hidden_to_class_gradient( model, training_forward_pass_results, training_targets_batch, weight_decay_coef );
		i2h_gradient			= input_to_hidden_gradient( model, training_forward_pass_results, training_input_batch, training_targets_batch, weight_decay_coef );
		
		i2h_weight_update	= ( learning_rate * i2h_gradient ) + ( momentum * prev_i2h_update );
		h2c_weight_update	= ( learning_rate * h2c_gradient ) + ( momentum * prev_h2c_update ); 		
		
		model.input_to_hidden_weights -= ( i2h_weight_update);			
		model.hidden_to_class_weights -= ( h2c_weight_update );

		# record the weight updates (used for momentum in the next iteration)
		prev_i2h_update  = i2h_weight_update;
		prev_h2c_update = h2c_weight_update; 


	end
	
	# remove time elapsed
	minutes_elapsed = etime(clock, start)/60.

	# Use the learned model to make predictions for test data
	predictions = make_predictions( model, test_input );

	# display some results including classification error (rather than cross-entropy error, since classification is 
	# what we are ultimately interested in)
	printf( "After %d epochs:\n", t )
	classification_error_rate = classification_error( vectors_to_labels( test_targets ), predictions );
	confusion_matrix( vectors_to_labels( test_targets ), predictions )
	plot_error( training_error_record, validation_error_record )
	printf( "Error rate: %f %% \n\n\n", classification_error_rate  )  ## % is escape character
	
endfunction


# calculate the validation error for the current model
function val_error = validation( model, validation_input, validation_targets, weight_decay_coef, weight_decay_type )

	validation_forward_pass_results	= forward_pass( validation_input, model );
	val_error 	= mean_cross_entropy_error( validation_forward_pass_results.output_from_softmax, validation_targets ) ...
				   + weight_decay_error( model, weight_decay_coef, weight_decay_type ); 

endfunction


# take a model, some test input and corresponding test labels. Return predicted digits
function predictions = make_predictions( model, test_input )

	forward_pass_results = forward_pass( test_input, model) ;
	predictions 			 =  vectors_to_labels( forward_pass_results.output_from_softmax );
	
endfunction





# plot the training and validation errors at each timestep
function plot_error( training_error_record, validation_error_record )

	hold on
	plot( training_error_record, Linespec='-r' );   
	plot( validation_error_record, Linespec='-b' );
	legend ( 'training_error_record', 'validation_error_record' );
	hold off
	
endfunction





# carries out a forward pass over the network, and returns results from each step
function results = forward_pass( training_input, model )

	results.input_to_hidden_units 		= transpose( training_input ) * model.input_to_hidden_weights; 			# result is <number of examples> * <number of hidden units>
	results.output_from_hidden_units	= logistic( results.input_to_hidden_units );
	results.input_to_softmax			= results.output_from_hidden_units * model.hidden_to_class_weights;	# result is <number of examples> * <number of classes>
	results.output_from_softmax 		= softmax( results.input_to_softmax );										

endfunction



# initialize both layers of weights
function model = initialize_weights( num_input_units, num_hidden_units, num_output_units )

	# set all weights randomly in the range -1 to 1
	model.input_to_hidden_weights =  rand( num_input_units, num_hidden_units ) * 2 - 1 ; # range -1 to 1
	model.hidden_to_class_weights = rand( num_hidden_units, num_output_units ) * 2 - 1;

endfunction













