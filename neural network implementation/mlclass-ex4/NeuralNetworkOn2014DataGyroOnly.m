%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 3;  % x,y,z  and xyz moving standard deviationfrom accelerometer
hidden_layer_size = 2;   % 2 hidden units
num_labels = 2;          % 2 labels, true or false 0 or 1
                          

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains accelerometer readings from a smartphone.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

datapoints=loadData();
X = datapoints(:, [ 4, 5 ,6]); y =datapoints(:, 13);
y(y==0)=2; %because of indexing. Octave indexes start from 1;
m = size(X, 1);

validationSet= csvread('C:\Users\John\Dropbox\kool2\kool\01 Semester VIII\thesis\human-reading-activity-recognition-on-smartphones\data analysis\implementation of learning algorithms\neural network\mlclass-ex4\validationSet.csv');
validationData=validationSet(:, [ 4, 5 ,6]); validationLabels =validationSet(:, 13);
validationLabels(validationLabels==0)=2; %because of indexing. Octave indexes start from 1;

testSet= csvread('C:\Users\John\Dropbox\kool2\kool\01 Semester VIII\thesis\human-reading-activity-recognition-on-smartphones\data analysis\implementation of learning algorithms\neural network\mlclass-ex4\testSet.csv');
testData=testSet(:, [ 4, 5 ,6]); testLabels =testSet(:, 13);
testLabels(testLabels==0)=2; %because of indexing. Octave indexes start from 1;







%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies reading position. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)
%to find average we train for 10 times
for i = 1:10
	fprintf('\nInitializing Neural Network Parameters ...\n')
	
	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
	
	% Unroll parameters
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
	
	
	
	
	
	
	
	%% =================== Part 8: Training NN ===================
	%  You have now implemented all the code necessary to train a neural 
	%  network. To train your neural network, we will now use "fmincg", which
	%  is a function which works similarly to "fminunc". Recall that these
	%  advanced optimizers are able to train our cost functions efficiently as
	%  long as we provide them with the gradient computations.
	%
	fprintf('\nTraining Neural Network... \n')
	
	%  After you have completed the assignment, change the MaxIter to a larger
	%  value to see how more training helps.
	options = optimset('MaxIter', 30);
	
	%  You should also try different values of lambda
	lambda = 4;
	
	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X, y, lambda);
	
	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
	
	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));
	
	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));
	
	fprintf('Program paused. Press enter to continue.\n');
	pause;
	
	
	
	
	%% ================= Part 10: Implement Predict =================
	%  After training the neural network, we would like to use it to predict
	%  the labels. You will now implement the "predict" function to use the
	%  neural network to predict the labels of the training set. This lets
	%  you compute the training set accuracy.
	
	validationPred = predict(Theta1, Theta2, validationData);
	
	
		testPred = predict(Theta1, Theta2, testData);
		
		precisionrecall=[testPred,testLabels];
		testpos=sum(precisionrecall(:,1)==1);
		conpos=sum(precisionrecall(:,2)==1);
		truepositive=sum((precisionrecall(:,1)==1)&(precisionrecall(:,2)==1));
		precision=truepositive/testpos;
		precisions(i)=precision;
		
		testneg=sum(precisionrecall(:,1)==2);
		conneg=sum(precisionrecall(:,2)==2);
		trueneg=sum((precisionrecall(:,1)==2)&(precisionrecall(:,2)==2))
		
		precision=double(truepositive/testpos)
		precisions(i)=precision;
		
		negativePredictiveValue=double(trueneg/testneg)
		negativePredictiveValues(i)=negativePredictiveValue;
		
		sensitivity=double(truepositive/conpos)
		sensitivities(i)=sensitivity;
		
		specificity=double(trueneg/conneg)
		specificities(i)=specificity;
		
		Accuracy=double((truepositive+trueneg)/size(testPred,1));
		Accuracies(i)=Accuracy;
		fprintf('\nTraining Set Accuracy on validationSet: %f\n', mean(double(validationPred == validationLabels)) * 100);
	
	fprintf('\nTraining Set Accuracy on validationSet: %f\n', mean(double(validationPred == validationLabels)) * 100);
	
	testPred = predict(Theta1, Theta2, testData);
	
	fprintf('\nTraining Set Accuracy on testSet: %f\n', mean(double(testPred == testLabels)) * 100);
	

endfor

avgPrecision=mean(precisions)
avgNegativePredictiveValue=mean(negativePredictiveValues)
avgSensitivity=mean(sensitivities)
avgSpecifcity=mean(specificities)
avgAccuracy=mean(Accuracies)*100