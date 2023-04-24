from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import sys
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
physical_devices = tf.config.experimental.list_physical_devices('CPU')
tf.config.experimental.set_visible_devices(physical_devices[0])



data = load_iris()
data_X=data.data[:-50]
data_y=data.target[:-50]
X_train, X_test, y_train, y_test = train_test_split(
     data_X, data_y, test_size=0.25, random_state=42)

data_combined=np.append(X_train, y_train[:,None], axis=1)
data_points=len(data_combined)

def init_weights(modes, layers, active_sd=0.1, passive_sd=1):
    # Number of interferometer parameters:
    M = int(modes * (modes - 1)) 

    # Create the TensorFlow variables
    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    r1_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s_mag_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    s_phase_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    r2_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    dp_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)

    weights = tf.concat(
        	[int1_weights, r1_weights, s_mag_weights,s_phase_weights, int2_weights, r2_weights, dr_weights, 		dp_weights, k_weights], axis=1)

    weights = tf.Variable(weights)

    return weights

def input_qnn_layer(hid,q):
    with tf.name_scope('inputlayer'):
        ops.Dgate(hid[0],hid[1]) | q[0]
        ops.Dgate(hid[2],hid[3]) | q[1]

# Defining standard QNN layers
def qnn_layer(params,layer_number,q):
    with tf.name_scope('layer_{}'.format(layer_number)):
        N = len(q)
        M = int(modes * (modes - 1)) 
        
        int1 = params[:M]
        r1=params[M:M+N]
        sm = params[M+N:M+2*N]
        sp = params[M+2*N:M+3*N]
        int2 = params[M+3*N:2*M+3*N]
        r2=params[2*M+3*N:2*M+4*N]
        dr = params[2*M+4*N:2*M+5*N]
        dp = params[2*M+5*N:2*M+6*N]
        ker = params[2*M+6*N:2*M+7*N]
        
        theta1=int1[:len(int1)//2]
        phi1=int1[len(int1)//2:]
        
        theta2=int2[:len(int2)//2]
        phi2=int2[len(int2)//2:]
        
        
        
        for k, (q1, q2) in enumerate(combinations(q,2)):
            ops.BSgate(theta1[k], phi1[k]) | (q1, q2)

        for i in range(N):
            ops.Rgate(r1[i]) | q[i]

        for i in range(N):
            ops.Sgate(sm[i],sp[i]) | q[i]

        for k, (q1, q2) in enumerate(combinations(q,2)):
            ops.BSgate(theta2[k], phi2[k]) | (q1, q2)

        for i in range(N):
            ops.Rgate(r2[i]) | q[i]

        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]

        for i in range(N):
            ops.Kgate(ker[i]) | q[i]


modes=2
cutoff_dim=10
batch_size=15
layers=6
hidden_units=4


eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim,"batch_size": batch_size})
qnn = sf.Program(modes)

# initialize QNN weights
weights = init_weights(modes, layers) # our TensorFlow weights
num_params = np.prod(weights.shape) 

sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
sf_params = np.array([qnn.params(*i) for i in sf_params])

hid_params = np.arange(num_params,num_params+hidden_units).reshape(hidden_units,1).astype(np.str)
hid_params = np.array([qnn.params(*i) for i in hid_params])

with qnn.context as q:
    input_qnn_layer(hid_params,q)
    for k in range(layers):
        qnn_layer(sf_params[k],k, q)


def cost(X,y):
     
    mapping_wt = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}
    mapping_hid={p.name: w for p, w in zip(hid_params, tf.transpose(X))} 
    mapping_wt.update(mapping_hid)
    results = eng.run(qnn, args=mapping_wt)
    ket = results.state.ket()
    func_to_minimise = 0

# Building up the function to minimize by looping through batch
    for i in range(batch_size):
        # Probabilities corresponding to a single photon in either mode
        prob = tf.abs(ket[i, y[i, 0], y[i, 1]]) ** 2
        # These probabilities should be optimised to 1
        func_to_minimise += (1.0 / batch_size) * (prob - 1) ** 2
    print(func_to_minimise)    
    print(tf.math.real(results.state.trace()))
    return func_to_minimise     

batch_size=15
optimizer = tf.keras.optimizers.Adam()

loss_ls=[]
for i in range(100):
    split_data = np.split(data_combined, data_points / batch_size)

    for batch in split_data:
        data_points_principal_components = batch[:,:-1]/10
        # Data classes
        classes = batch[:, -1]
        # Encoding classes into one-hot form
        one_hot_input = np.zeros((batch_size, 2),dtype=np.int32)

        for k in range(batch_size):
            if int(classes[k]) == 0:
                # Encoded such that genuine transactions should be outputted as a photon in the first mode
                one_hot_input[k] = [1, 0]
            else:
                one_hot_input[k] = [0, 1]
        with tf.GradientTape() as tape:
            loss=cost(data_points_principal_components , one_hot_input)
        gradients=tape.gradient(loss,weights)
        optimizer.apply_gradients(zip([gradients],[weights]))

        loss_ls.append(loss)
    print("loss at iteration {} is {}".format(i,loss))
