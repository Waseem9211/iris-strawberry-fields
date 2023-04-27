from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf
import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate, Kgate, Sgate, Rgate
import sys


# Two modes required: one for "genuine" transactions and one for "fradulent"
mode_number = 2
# Number of photonic quantum layers
depth = 8

# Fock basis truncation
cutoff = 10
# Number of batches in optimization
reps = 5000

# Label for simulation
simulation_label = 1

# Number of batches to use in the optimization
batch_size = 10

# Random initialization of gate parameters
sdev_photon = 0.1
sdev = 1

# Variable clipping values
disp_clip = 5
sq_clip = 5
kerr_clip = 1

# If loading from checkpoint, previous batch number reached
ckpt_val = 0

# Number of repetitions between each output to TensorBoard
tb_reps = 100
# Number of repetitions between each model save
savr_reps = 1000

model_string = str(simulation_label)

# Target location of output
folder_locator = 'outputs/'

# Locations of TensorBoard and model save outputs
board_string = folder_locator + 'tensorboard/' + model_string + '/'
checkpoint_string = folder_locator + 'models/' + model_string + '/'

data = load_iris()
data_X=data.data[:-50]
data_y=data.target[:-50]
X_train, X_test, y_train, y_test = train_test_split(
     data_X, data_y, test_size=0.50, random_state=42)
data_combined=np.append(X_train, y_train[:,None], axis=1)
data_points=len(data_combined)

# Input neurons
input_neurons = 4
# Output neurons of classical part
output_neurons = 4

# Defining classical network parameters
input_classical_layer = tf.placeholder(tf.float32, shape=[batch_size, input_neurons])

output_layer=input_classical_layer


# Number of beamsplitters in interferometer
bs_in_interferometer = int(1.0 * mode_number * (mode_number - 1) / 2)

with tf.name_scope('variables'):
    bs_variables = tf.Variable(tf.random_normal(shape=[depth, bs_in_interferometer, 2, 2]
                                                , stddev=sdev))
    phase_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number, 2], stddev=sdev))

    sq_magnitude_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                          , stddev=sdev_photon))
    sq_phase_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                      , stddev=sdev))
    disp_magnitude_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                            , stddev=sdev_photon))
    disp_phase_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                        , stddev=sdev))
    kerr_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number], stddev=sdev_photon))


parameters = [bs_variables,phase_variables, sq_magnitude_variables, sq_phase_variables, 				disp_magnitude_variables,disp_phase_variables, kerr_variables]


# Defining input QNN layer, whose parameters are set by the outputs of the classical network
def input_qnn_layer():
    with tf.name_scope('inputlayer'):
        Dgate(output_layer[:, 0], output_layer[:, 1]) \
        | q[0]
        Dgate(output_layer[:, 2], output_layer[:, 3]) \
        | q[1]


# Defining standard QNN layers
def qnn_layer(layer_number):
    with tf.name_scope('layer_{}'.format(layer_number)):
        BSgate(bs_variables[layer_number, 0, 0, 0], bs_variables[layer_number, 0, 0, 1]) \
        | (q[0], q[1])

        for i in range(mode_number):
            Rgate(phase_variables[layer_number, i, 0]) | q[i]

        for i in range(mode_number):
            Sgate(sq_magnitude_variables[layer_number, i],
                  sq_phase_variables[layer_number, i]) | q[i]

        BSgate(bs_variables[layer_number, 0, 1, 0], bs_variables[layer_number, 0, 1, 1]) \
        | (q[0], q[1])

        for i in range(mode_number):
            Rgate(phase_variables[layer_number, i, 1]) | q[i]

        for i in range(mode_number):
            Dgate(disp_magnitude_variables[layer_number, i], disp_phase_variables[layer_number, i]) | q[i]

        for i in range(mode_number):
            Kgate(kerr_variables[layer_number, i]) | q[i]


eng, q = sf.Engine(mode_number)

# construct the circuit
with eng:
    input_qnn_layer()

    for i in range(depth):
        qnn_layer(i)

# run the engine (in batch mode)
state = eng.run("tf", cutoff_dim=cutoff, eval=False, batch_size=batch_size)
# extract the state
ket = state.ket()


# Classifications for whole batch: rows act as data points in the batch and columns
# are the one-hot classifications
classification = tf.placeholder(shape=[batch_size, 2], dtype=tf.int32)

func_to_minimise = 0

# Building up the function to minimize by looping through batch
for i in range(batch_size):
    # Probabilities corresponding to a single photon in either mode
    prob = tf.abs(ket[i, classification[i, 0], classification[i, 1]]) ** 2
    # These probabilities should be optimised to 1
    func_to_minimise += (1.0 / batch_size) * (prob - 1) ** 2

# Defining the cost function
cost_func = func_to_minimise
tf.summary.scalar('Cost', cost_func)



# We choose the Adam optimizer
optimiser = tf.train.AdamOptimizer()
training = optimiser.minimize(cost_func)

# Saver/Loader for outputting model
saver = tf.train.Saver(parameters)

session = tf.Session()
session.run(tf.global_variables_initializer())

# Load previous model if non-zero ckpt_val is specified
if ckpt_val != 0:
    saver.restore(session, checkpoint_string + 'sess.ckpt-' + str(ckpt_val))

# TensorBoard writer
writer = tf.summary.FileWriter(board_string)
merge = tf.summary.merge_all()

counter = ckpt_val

# Tracks optimum value found (set high so first iteration encodes value)
opt_val = 1e20
# Batch number in which optimum value occurs
opt_position = 0
# Flag to detect if new optimum occured in last batch
new_opt = False

while counter <= reps:

    # Shuffles data to create new epoch
    np.random.shuffle(data_combined)

    # Splits data into batches
    split_data = np.split(data_combined, data_points / batch_size)

    for batch in split_data:

        if counter > reps:
            break

        # Input data (provided as principal components)
        data_points_principal_components = batch[:, 1:input_neurons + 1]/10
        # Data classes
        classes = batch[:, -1]

        # Encoding classes into one-hot form
        one_hot_input = np.zeros((batch_size, 2))

        for i in range(batch_size):
            if int(classes[i]) == 0:
                # Encoded such that genuine transactions should be outputted as a photon in the first mode
                one_hot_input[i] = [1, 0]
            else:
                one_hot_input[i] = [0, 1]

        # Output to TensorBoard
        if counter % tb_reps == 0:
            [summary, training_run, func_to_minimise_run] = session.run([merge, training, func_to_minimise],
                                                                        feed_dict={
                                                                            input_classical_layer:
                                                                                data_points_principal_components,
                                                                            classification: one_hot_input})
            writer.add_summary(summary, counter)

        else:
            # Standard run of training
            [training_run, func_to_minimise_run] = session.run([training, func_to_minimise], feed_dict={
                input_classical_layer: data_points_principal_components, classification: one_hot_input})

        # Ensures cost function is well behaved
        if np.isnan(func_to_minimise_run):
            compute_grads = session.run(optimiser.compute_gradients(cost_func),
                                        feed_dict={input_classical_layer: data_points_principal_components,
                                                   classification: one_hot_input})
            if not os.path.exists(checkpoint_string):
                os.makedirs(checkpoint_string)
            # If cost function becomes NaN, output value of gradients for investigation
            np.save(checkpoint_string + 'NaN.npy', compute_grads)
            print('NaNs outputted - leaving at step ' + str(counter))
            raise SystemExit

        # Test to see if new optimum found in current batch
        if func_to_minimise_run < opt_val:
            opt_val = func_to_minimise_run
            opt_position = counter
            new_opt = True

        # Save model every fixed number of batches, provided a new optimum value has occurred
        if (counter % savr_reps == 0) and (i != 0) and new_opt and (not np.isnan(func_to_minimise_run)):
            if not os.path.exists(checkpoint_string):
                os.makedirs(checkpoint_string)
            saver.save(session, checkpoint_string + 'sess.ckpt', global_step=counter)
            # Saves position of optimum and corresponding value of cost function
            np.savetxt(checkpoint_string + 'optimum.txt', [opt_position, opt_val])
        
        counter += 1
    print("loss at {} batch {}".format(counter,func_to_minimise_run))    

