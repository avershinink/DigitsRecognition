import numpy as np
import _pickle
import gzip
import pprint

class Network(object):
    def __init__(self, sizes, path_to_weights_file = ''):
        """Building a network from the giving map.
        sizes[0]    - inputs count
        sizes[1:-1] - hidden layers
        sizez[-1]   - outputs count

        path_to_weights_file - path to file with calculated weights data
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # biases/activations is array of arrays
        # [layer] -> 
        #            [biases/activations]
        self.np_biases                 = np.array([np.random.random(sizes[x+1]) for x in range(len(sizes)-1)])
        self.np_activations            = np.array(        [np.zeros(sizes[x+1]) for x in range(len(sizes)-1)])
        self.np_deltas                 = np.array(        [np.zeros(sizes[x+1]) for x in range(len(sizes)-1)])
        self.np_prev_learn_deltas      = np.array(        [np.zeros(sizes[x+1]) for x in range(len(sizes)-1)])
        self.np_prev_bias_learn_deltas = np.array(        [np.zeros(sizes[x+1]) for x in range(len(sizes)-1)])
        # set weights as array of arrays of arrays
        # [layer]->
        #        ->[neuron]->
        #                  ->[neuron weights]
        try:
            self.np_weights = np.load(path_to_weights_file)
        except:
            self.np_weights = np.array([np.array([np.random.random(sizes[x]) for i in range(sizes[x+1])]) for x in range(len(sizes)-1)])

    def train_network(self, training_data, epochs, learning_rate, mse, momentum, decay, validation_data, test_data = None):
        epoch = 0
        target_mse = mse
        mse = target_mse + 1
        while epoch < epochs and mse>target_mse:
            epoch += 1
            mse = 0
            for inputs,expected_result in training_data:
                self.feedforward(inputs)
                mse += (expected_result - self.np_activations[-1]) * (expected_result - self.np_activations[-1])
                self.backprop(expected_result)
                self.updateweights(inputs, learning_rate, momentum, decay)
            mse = np.sum(mse) / len(training_data)
            print("Epoch {0}; \t mse = {1}".format(epoch, mse))
        if(validation_data):
            ransw = self.evaluate(validation_data)
            print("validation_data right answers count {0};".format(ransw))

    def feedforward(self, inputs):
        for layer in range(len(self.np_weights)):
            sum = (self.np_weights[layer] @ inputs)
            inputs = np.apply_along_axis(np.tanh,0, sum)
            self.np_activations[layer] = inputs

    def backprop(self, expected_results):
        #tanh deriv
        grd = (1 + self.np_activations[-1]) * (1 - self.np_activations[-1])
        self.np_deltas[-1] = grd * (expected_results - self.np_activations[-1])

        for layer in range(len(self.np_activations)-2, -1, -1):
            #tanh deriv
            grd = (1 + self.np_activations[layer]) * (1 - self.np_activations[layer])
            np_sum = (self.np_deltas[layer + 1] @ self.np_weights[layer + 1])
            self.np_deltas[layer] = np_sum * grd

    def updateweights(self, inputs, learning_rate, momentum, decay):
        layer_index = 0
        np_inputs = np.repeat([inputs],len(self.np_activations[0]),0)
        
        for w,dlt, wpd, bpd in zip(self.np_weights,self.np_deltas,self.np_prev_learn_deltas, self.np_prev_bias_learn_deltas):
            tpw = np.transpose(w)
            learn_delta = dlt * learning_rate * np.transpose(np_inputs)
            tpw += learn_delta
            tpw += momentum * wpd
            tpw -= decay * tpw
            self.np_weights[layer_index] = np.transpose(tpw)
            self.np_prev_learn_deltas[layer_index] = learn_delta

            bias_learn_delta = dlt * learning_rate * 1
            new_bias = self.np_biases[layer_index]
            new_bias += bias_learn_delta
            new_bias += momentum * bpd
            new_bias -= decay * new_bias
            self.np_biases[layer_index] = new_bias
            self.np_prev_bias_learn_deltas[layer_index] = bias_learn_delta

            if(layer_index+1 < len(self.np_activations)):
                np_inputs = np.repeat([self.np_activations[layer_index]],len(self.np_activations[layer_index+1]),0)
            layer_index += 1
        pass

    def evaluate(self, test_data, show_erorrs = False):
        right_answ_count = 0
        for td,rd in test_data:
            self.feedforward(td)
            network_answer = np.argmax(self.np_activations[-1])
            if(rd[network_answer] == 1):
                right_answ_count += 1
            else:
                if(show_erorrs):
                    print('nnet answer {0}\t right answer {1}'.format(network_answer,np.argmax(rd)))
        return right_answ_count



def load_data(output_size):
    """Load the MNIST data in a tuple containing the training data,
    the validation data, and the test data.
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = _pickle.load(f, encoding="latin1")
    f.close()
    #tanh vectorization
    vectorized_test_data = tuple((input, set_answer_vector(np.full(output_size,-1.), answ)) for input, answ in zip(test_data[0], test_data[1]))
    vectorized_training_data = tuple((input, set_answer_vector(np.full(output_size,-1.), answ)) for input, answ in zip(training_data[0], training_data[1]))
    vectorized_validation_data = tuple((input, set_answer_vector(np.full(output_size,-1.), answ)) for input, answ in zip(validation_data[0], validation_data[1]))
    return (vectorized_training_data, vectorized_validation_data, vectorized_test_data)

def set_answer_vector(vector, number):
    """Return a vector with a 1.0 in the 'number'
    position and -1 elsewhere.
    Is used to create a desiared output vector from the neural
    network."""
    vector[number] = 1.0
    return vector

print('Building Network')
networklayers = [784, 30, 10]
net = Network(networklayers,'data/'+str(networklayers)+'.npy')
print('Network Built')

print('Loading MNIST Data')
training_data, validation_data, test_data = load_data(10)
print('MNIST Data Loaded')

print('Learning using Stochastic gradient descent')
print('Network training Start')
net.train_network(training_data, 40, .005, 0.02, 0.00001, 0.000001, validation_data = validation_data)
print('Network training End')

print('Processing testing_data...')
ransw_count = net.evaluate(test_data)
print('Right answers from test_data {0} out of {1}'.format(ransw_count, len(test_data)))
saveYN = input('Save Network Weights? Y/N ')
if(saveYN.lower() == 'y'):
    np.save('data/'+str(net.sizes), net.np_weights)