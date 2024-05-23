import random
rand=random.Random()

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
test_print(data[20])

# Four layers:
# 1. input layer
# 2. hidden layer 1 of 16 neurons, each with 784 weights and one bias
# 3. hidden layer 1 of 16 neurons
# 4. output layer of 10

class Neuron:
    def __init__(self, random_weights=None):
        self.weights = []
        if random_weights:
            for _ in range(random_weights):
                self.weights.append(rand.uniform(-2, 2)) #assigns arbitrary weights between 0 and 2
    
    def __str__(self):
       return f"Neuron:{self.weights}"

class Layer:
    def __init__(self, neurons=16, random_weights=4):
        self.neurons = []
        for _ in range(neurons):
            self.neurons.append(Neuron(random_weights))
        self.bias = rand.uniform(-10, 10)
    
    def __str__(self):
       return f"Layer:Bias={self.bias}, {[str(n) for n in self.neurons]}"
    
    def sigmoid(x):
       return 1/(1+(2.718281828459**(-x)))

layer2 = Layer()
print(layer2)

# Returns an array of 10 numbers
def execute_nn(image_pixels):
    # for each neuron in layer 1
    pass