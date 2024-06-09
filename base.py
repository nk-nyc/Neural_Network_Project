import random
rand=random.Random()
from math import exp

global LEARNING_RATE
LEARNING_RATE = 0.2

# training = True/False whether this is training vs. test data
# n = number of images/labels to get
# returns array of (pixels, label)
def get_data(training=True, n=5000):
  lbl_f = open(f"data/train-labels-idx1-ubyte", "rb")   # MNIST has labels (digits)
  img_f = open(f"data/train-images-idx3-ubyte", "rb")     # and pixel vals separate

  img_f.read(16)   # discard header info
  lbl_f.read(8)    # discard header info
  data_array = []
  for _ in range(n):   # number images requested
    image_pixels = []
    lbl = ord(lbl_f.read(1))  # get label (unicode, one byte)
    for _ in range(784):  # get 784 vals from the image file
      image_pixels.append(ord(img_f.read(1)))
    data_array.append((image_pixels, lbl))
  img_f.close(); lbl_f.close()
  return(data_array)

def test_print(item):
  (image_pixels, label) = item
  print(f"This should look kind of like a {label}")
  for y in range(28):
    for x in range(28):
      if image_pixels[y*28+x] > 0:
        print(" ", end='')
      else:
        print("X", end='')
    print("")

def sigmoid(x):
    if x < -500:
       return 0
    elif x > 500:
       return 1
    return 1/(1+exp(-x))

#returns the derivative of the sigmoid function
def sig_deriv(x):
   return x * (1.0 - x)

# Four layers:
# 1. input layer
# 2. hidden layer 1 of 16 neurons, each with 784 weights and one bias
# 3. hidden layer 1 of 16 neurons
# 4. output layer of 10

class Neuron:
    def __init__(self, random_weights=None):
        # The actual number held in this neuron
        self.value = 0
        self.delta = 0
        self.weights = []
        self.bias = rand.uniform(-1, 1)
        if random_weights:
            for _ in range(random_weights):
                self.weights.append(rand.uniform(-1, 1)) #assigns arbitrary weights between 0 and 2
   
    def __str__(self):
       return f"Neuron:{self.weights}"
   
    # Inputs is a list of Neurons that correspond to the weights
    def forward_propagate(self, inputs):
      self.value = self.bias
      for i in range(len(inputs)):
          self.value += self.weights[i] * inputs[i]
      self.value = sigmoid(self.value)

class Layer:
    def __init__(self, neurons=16, random_weights=1):
        self.neurons = []
        for _ in range(neurons):
            self.neurons.append(Neuron(random_weights))
   
    def __str__(self):
       return f"Layer:{[str(n) for n in self.neurons]}"
    #creates a list of the layer, which has an overall bias and 16 neurons

    def forward_propagate(self, prev_layer):
       inputs = [n.value for n in prev_layer.neurons]
       for neuron in self.neurons:
          neuron.forward_propagate(inputs)

#need to write backpropagation that can plug into layer and neuron values
#cost function
def execute_nn(image_pixels, network):    
    for i in range(len(network)):
      if i == 0: # input layer
        for j in range(len(image_pixels)):
           network[i].neurons[j].value = image_pixels[j]
      else: # another layer
        network[i].forward_propagate(network[i-1])

def mean_squared_error(actual_values, predicted_values):
   return 1/(len(actual_values)) * (sum([(actual_values[n] - predicted_values[n])**2 for n in range(len(actual_values))]))

#stochastic gradient descent:
#randomly initialize values
#set parameters (number of iterations and learning rate)
#SGD loop:
    #Shuffle the training dataset for randomness
    #Iterate over each training example in that order
    #Compute the gradient of the cost function with respect to the model parameters using the current training example
    #Update the model parameters in the direction of the negative gradient, scaled by the LR
    #Evaluate the difference in the cost function between iterations
#When max. iterations are complete or the convergence criteria are met, return those parameters
#https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/
#https://realpython.com/gradient-descent-algorithm-python/
       
def gradient_descent(gradient, start, num_epochs, lr, tol):
        vector = start
        for _ in range(num_epochs):
            diff = -lr * gradient(vector)
            if abs(diff) <= tol:
                break
            vector += diff
        return vector
#start should be able to take an array

#error = (output - expected) * sig_deriv(output) in output layer

def backward_propagate():
  pass

def main():

# print("Loading data...")
  data = get_data()
# print(f"Loaded {len(data)} items")

# execute_for_image(0)
   
  input_layer = Layer(neurons=784,random_weights=0)
  layer1 = Layer(neurons=16,random_weights=784)
  layer2 = Layer(neurons=16, random_weights=16)
  output_layer = Layer(neurons=10, random_weights=16)

  network = [
    input_layer,
    layer1,
    layer2,
    output_layer
  ]
  
  for epoch in range(5001):
    for image_index in range(len(data)): # eventually len(data)
      (orig_pixels, label) = data[image_index]
      pixels = [n/256.0 for n in orig_pixels]
      execute_nn(pixels, network)
      for neuron_index, neuron in enumerate(output_layer.neurons):
        neuron.delta = (neuron.value - (0 if neuron_index != label else 1)) * sig_deriv(neuron.value)
      for layer_index in reversed(range(len(network))):
        if layer_index == 0 or layer_index == len(network) - 1:
           continue
        layer = network[layer_index]
        for neuron_index, neuron in enumerate(layer.neurons):
          neuron.delta = 0
          for i in range(len(network[layer_index + 1].neurons)):
            deriv = sig_deriv(neuron.value)
            future_error = network[layer_index+1].neurons[i].delta
            my_weight = network[layer_index + 1].neurons[i].weights[neuron_index]
            neuron.delta += deriv * future_error * my_weight
          
      for layer in network[1:-1]: 
        for neuron in layer.neurons:
            for i in range(len(neuron.weights)):
              neuron.weights[i] = neuron.weights[i] - LEARNING_RATE * neuron.delta * neuron.value
            neuron.bias = neuron.bias - LEARNING_RATE * neuron.delta
        
    if epoch % 1 == 0:
      total_mse = 0
      for image_index in range(len(data)):
        (orig_pixels, label) = data[image_index]
        pixels = [n/256.0 for n in orig_pixels]
        execute_nn(pixels, network)
        total_mse += mean_squared_error([n.value for n in network[-1].neurons], [0 if n != label else 1 for n in range(10)])
      print(epoch, total_mse)

if __name__ == "__main__":
    main()