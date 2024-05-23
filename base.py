
import random
rand=random.Random()
from math import exp

# training = True/False whether this is training vs. test data
# n = number of images/labels to get
# returns array of (pixels, label)
def get_data(training=True, n=100):
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


print("Loading data...")
data = get_data()
print(f"Loaded {len(data)} items")

# Test one of them
test_print(data[0])

def sigmoid(x):
    if x < -500:
       return 0
    elif x > 500:
       return 1 
    return 1/(1+exp(-x))

# Four layers:
# 1. input layer
# 2. hidden layer 1 of 16 neurons, each with 784 weights and one bias
# 3. hidden layer 1 of 16 neurons
# 4. output layer of 10

class Neuron:
    def __init__(self, random_weights=None):
        # The actual number held in this neuron
        self.value = 0 
        self.weights = []
        self.bias = round(rand.uniform(-1, 1), 4)
        if random_weights:
            for _ in range(random_weights):
                self.weights.append(round (rand.uniform(-1, 1), 4)) #assigns arbitrary weights between 0 and 2
   
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

input_layer = Layer(neurons=784,random_weights=0)
# layer1 = Layer(neurons=16,random_weights=784)
output_layer = Layer(neurons=10, random_weights=16)
layer2 = Layer(neurons=16, random_weights=784)

network = [
  input_layer,
  layer2,
  output_layer
]

#need to write backpropagation that can plug into layer and neuron values
#cost function
def execute_nn(image_pixels):    
    for i in range(len(network)):
      if i == 0: # input layer
        for j in range(len(image_pixels)):
           network[i].neurons[j].value = image_pixels[j]
      else: # another layer
        network[i].forward_propagate(network[i-1])

def mean_squared_error(actual_values, predicted_values):
   return 1/(len(actual_values)) * (sum([(actual_values[n] - predicted_values[n])**2 for n in range(len(actual_values))]))

def cost(values, labels):
   mean_squared_error(values, )

def execute_for_image(image_index):
  (pixels, label) = data[image_index]
  execute_nn(pixels)
  for i in range(10):
    print(f"{i}:{output_layer.neurons[i].value}")
  return(mean_squared_error([n.value for n in output_layer.neurons], [0 if n != label else 1 for n in range(10)]))

execute_for_image(0)