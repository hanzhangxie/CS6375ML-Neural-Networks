# CS6375ML-Neural-Networks

1. The dataset we used

	We used Iris Plants Database as our dataset. This dataset contains 3 classes of 50 instances each, where each class refers to a type of iris plant. The predicted attribute is the class of iris plant.

	Number of Instances: 150 (50 in each of three classes)
	
	Number of Attributes: 4 numeric, predictive attributes and the class

	Attribute Information: 
	1. sepal length in cm
                       	   
	2. sepal width in cm
                       
	3. petal length in cm
                       
	4. petal width in cm
                       
	5. class:  -- Iris Setosa
			   -- Iris Versicolour
               -- Iris Virginica

	Missing Attribute Values: None

2. The Packages we used

	-numpy, pandas, sklearn

	-From sklearn, we used shuffle, LabelEncoder, OneHotEncoder

3. What we've done

	We modified the starter code in the following ways:
	1. We added tanh and ReLu activation functions and their derivatives besides sigmoid.
	2. We preprocessed the dataset, including handling null or missing values, standardizing, scaling the attributes, converting categorical attributes to numerical values.
	3. We added a parameter "activation" to forward_pass() method, compute_output_delta() method, compute_hidden_layer2_delta() method, compute_hidden_layer1_delta() method, compute_input_layer_delta() method and predict() method. You can call different activation function and its derivative by transmitting different parameter value for "activation". e.g., activation="sigmoid", activation="tanh", activation="ReLu".
	4. We implemented the predict function for applying the trained model on the test dataset and output the test error from this function.
	5. We added two parameters h1 (number of neurons in the first hidden layer) and h2 (number of neurons in the second hidden layer) to predict() method.
	6. We deleted the parameter X from preprocess() method and in the main function we preprocessed the entire dataset including splitting the train and test.

	We used shuffle() method to random sort the dataset and splitted it into 80:20 to create train and test datasets. This resulted with a training set of 120 instances and a testing set of 30 instances. We used preprocess() method to drop missing values and duplicates, standardize attributes and convert categorical(3 classes) to numerical. For each instance label, we considered it as a combination of three classes [Iris Setosa, Iris Versicolour, Iris Virginica], where the class value equals 1 and the other two equals 0 when this instance is classified into this class. e.g., [1, 0, 0] if the instance class is Iris Setosa, [0, 1, 0] if the instance class is Iris Versicolour and [0, 0, 1] if the instance class is Iris Virginica.

4. Output for different combination of parameters

	We chose h1 (number of neurons in the first hidden layer), h2 (number of neurons in the second hidden layer), the number of iterations and the learning rate as parameters to tune. We set some specific values for some parameters and ranges of values for other parameters. e.g., h1,h2 = {2,3,4,5}, max iterations = {500,800,1000,1500,2000,3000,4000,5000}, learning rate is [0.001, 0.1] and its step is 0.01. We used nested loops to combine these different parameters and also used different activation functions. We output the final results to the csv document. (For the ReLu activation function, the values of parameters are different from the other two functions because its specificity.)

5. Summary of our results

	For different combinations of parameters, three activation functions performed differently. So we discussed the best performance of these three functions respectively and compared the performance of them under the same parameters.

	---Sigmoid function performs best:

	Parameters: h1 = 4, h2 = 3, max_iterations = 4000, learning_rate = 0.071
	
	Results: mse = 0.055797

	Under the same parameters, the tanh function's result is 1.999999.

	---Tanh function performs best:

	Parameters: h1 = 5, h2 = 4, max_iterations = 5000, learning_rate = 0.001
	
	Results: mse = 0.07432138

	Under the same parameters, the sigmoid function's result is 0.34596163.

	---ReLu function performs best:

	Parameters: h1 = 8, h2 = 7, max_iterations = 2000, learning_rate = 0.00004
	
	Results: mse = 0.0671056

	Because the output results of ReLu function are very different from the other two under the same parameters, we set different values of them.

	We think that sigmoid activation function performed better than the other two because its error is in (0,1) which is much smaller than others. And the number of neurons sigmoid required is fewer and learning rate is higher, which makes the neural network simpler and faster to get the results.
