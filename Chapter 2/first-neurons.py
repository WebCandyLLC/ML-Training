# Single Neuron with 3 Inputs

# Input vector/array 
inputs = [1, 2, 3]

# Neuron that accepts the 3 inputs and their associated weight
weights = [0.2, 0.8, -0.5]

# linear bias applied to neuron calculation, between weights and bias, it has the form y = mx + b
bias = 2

#Brute force method of neuron
output = (inputs[0]*weights[0] +
          inputs[1]*weights[1] +
          inputs[2]*weights[2] + bias)

print(output)

# Layer of Neurons - Layer of 3 neurons, with 4 inputs to all 3 neurons in that layer

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

# Brute force method
outputs = [
    #Neuron 1
    inputs[0]*weights1[0] +
    inputs[1]*weights1[1] +
    inputs[2]*weights1[2] +
    inputs[3]*weights1[3] + bias1,
    #Neuron 2
    inputs[0]*weights2[0] +
    inputs[1]*weights2[1] +
    inputs[2]*weights2[2] +
    inputs[3]*weights2[3] + bias2,
    #Neuron 3
    inputs[0]*weights3[0] +
    inputs[1]*weights3[1] +
    inputs[2]*weights3[2] +
    inputs[3]*weights3[3] + bias3
]

print(outputs)

# Non-Brute Force method
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

#Output of current layer
layer_outputs = []
#for each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    #Zeroed output of given neuron
    neuron_output = 0
    #For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        #Multiply this inpuot by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input*weight
    #Add bias
    neuron_output += neuron_bias
    #Put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)

print(layer_outputs)

#Tensors, Arrays, and Vectors

#list
l = [1,5,6,2]
#list of lists
lol = [[1,5,6,2],
       [3,2,1,3]]
#list of lists # lists
lolol = [[[1,5,6,2],
          [3,2,1,3]],
         [[1,5,6,2],
          [3,2,1,3]],
         [[1,5,6,2],
          [3,2,1,3]]]
#Non-homologous List
nonlol = [[4,2,3],
          [5,1]]


#dot_product
a = [1,2,3]
b = [2,3,4]
dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(dot_product)

# Batches of Data

batch = [[1,5,6,2],
         [3,2,1,3],
         [5,2,1,2],
         [6,4,8,4],
         [2,8,5,3],
         [1,1,9,4],
         [6,6,0,4],
         [6,7,6,4]]

import numpy as np
#double brackets turn the list into a matrix containing a single row (i.e. turning a vector into a row vector)
np.array([[1,2,3]])
#or
a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T #tranpose operation with numpy

print(np.dot(a, b)) # returns a single element row vector

#Final Example
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layer_outputs)
