# writes predictions in format for Kaggle
function write_Kaggle_file( predictions, ...
								filename = 'kaggle_results', ...
								filepath   = 'C:\Users\jjsm\Documents\MNIST NN project\' )													)

	csvwrite( strcat( filepath, filename), predictions, row=2800, col=1);

	# manually add "ImageId" and "label" headers

endfunction