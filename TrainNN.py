from fann2 import libfann

connection_rate = .2
learning_rate = 0.7
num_input = 5001
num_hidden = 500
num_output = 5001

desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 1000

ann = libfann.neural_net()
ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file("TrainingData.txt", max_iterations, iterations_between_reports, desired_error)

ann.save("NN.net")
