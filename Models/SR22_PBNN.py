# Paper Title: "Stochastic ground motion models to NGA-West2 and NGASub databases using Bayesian neural network"
# Code developed by: Sreenath Vemula, Raghukanth STG

# Email: vsreenath2@gmail.com

# Submitted to: Earthquake Engineering and Structural Dynamics journal

### Initialize networks

# Before which, load SR22_West2_PBNN and SR22_Sub_PBNN files
"""

# Load required libraries which we use
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import Input, models, layers
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Input
import matplotlib.pyplot as plot

tfpl = tfp.layers
tfpd = tfp.distributions

#Compute mean and standard deviation to normalize the inputs and targets
West2_Input_mean = np.array([5.729923, 4.413790, 10.152883, 5.998398, 2.012284, 2.037392])
West2_Input_std = np.array([1.20254254, 0.9451266 , 3.82392659, 0.40372298, 0.97509672, 1.24592341])
West2_Target_mean = np.array([-4.87994126, -0.40750808, -1.5720524 , -4.87129283, -4.86554322,
       -4.83396755, -4.78392878, -4.71804622, -4.64194504, -4.52477363,
       -4.42237238, -4.36390945, -4.17157297, -4.10993212, -4.14920651,
       -4.27740031, -4.42336678, -4.570936  , -4.71155478, -4.78408848,
       -4.85785709, -4.99490487, -5.12360358, -5.3604136 , -5.67051165,
       -6.1146535 , -6.48955253, -6.79227626, -7.27018767])
West2_Target_std = np.array([2.42715374, 2.60588243, 3.11510912, 2.42484311, 2.43051757,
       2.43493243, 2.44403298, 2.45402093, 2.46459099, 2.4795051 ,
       2.48593953, 2.48862045, 2.45871136, 2.4408254 , 2.42881608,
       2.43069125, 2.45164507, 2.47593218, 2.51209384, 2.52887418,
       2.54547056, 2.5741994 , 2.60329123, 2.65998072, 2.73565873,
       2.84012449, 2.91487773, 2.9793393 , 3.08117291])

Sub_Input_mean = np.array([6.68478357, 5.17778629, 52.52107701, 6.03977884, 2.58125892, 3.52600285])
Sub_Input_std = np.array([1.07637091, 0.70259695, 33.92583053, 0.4649875, 0.70112263, 1.04375727])
Sub_Target_mean = np.array([-4.62930168, -0.09396118, -0.87646346, -4.62512346, -4.61123033,
       -4.57654343, -4.51962872, -4.44320954, -4.3557332 , -4.22339103,
       -4.10980331, -4.04542538, -3.87901567, -3.84546682, -3.91578905,
       -4.02926474, -4.15708557, -4.2799341 , -4.40100951, -4.46213172,
       -4.51942888, -4.63836273, -4.74801388, -4.95690246, -5.24014924,
       -5.65133563, -5.99135529, -6.28249356, -6.76874589])
Sub_Target_std = np.array([2.27127476, 2.13369015, 2.53483697, 2.27242527, 2.27640746,
       2.28640393, 2.30366638, 2.32003444, 2.33673787, 2.36636804,
       2.38607502, 2.39422058, 2.38914363, 2.36506948, 2.32110199,
       2.28934885, 2.25629222, 2.23293288, 2.2116741 , 2.20308297,
       2.19585583, 2.1842407 , 2.17238935, 2.15886921, 2.15639852,
       2.15997677, 2.17487696, 2.1878803 , 2.21326018])

# Defining Gaussian prior to all the layers with zero mean and unit standard deviation
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return Sequential([tfpl.DistributionLambda(lambda t: tfpd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))])

# Defining loss function
def NLL(y_true, y_pred):
    return -y_pred.log_prob(y_true)

# We use posterior mean field assumption
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return Sequential([
      tfpl.VariableLayer(2 * n, dtype=dtype),
      tfpl.DistributionLambda(lambda t: tfpd.Independent(
          tfpd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),])

# Initializing the network
West2_PBNN = Sequential([
    Input(shape=(6)), #Input layer
    tfpl.DenseVariational(  #Probabilistic layer
        units=7,
        name="dense_var_1",
        make_prior_fn=prior,
        make_posterior_fn=posterior_mean_field,
        kl_weight=1/203,
       activation='sigmoid'),
    tfpl.DenseVariational(   #Probabilistic layer
        units=7,
        name="dense_var_2",
        make_prior_fn=prior,
        make_posterior_fn=posterior_mean_field,
        kl_weight=1/203,
        activation='sigmoid'),
    Dense(units=tfpl.IndependentNormal.params_size(29),name='output'), #output is mean and standard deviation of targets
    tfpl.DistributionLambda(lambda x: tfpd.Independent(tfpd.Normal(loc = x[..., :29], scale = tf.nn.softplus( x[..., 29:]))),name='lambda')
  ], name='PBNN')

West2_PBNN.compile(keras.optimizers.Adam(learning_rate=0.01),loss=NLL, metrics=keras.metrics.MeanSquaredError())

Sub_PBNN = Sequential([
    Input(shape=(6)),
    tfpl.DenseVariational(
        units=7,
        name="dense_var_1",
        make_prior_fn=prior,
        make_posterior_fn=posterior_mean_field,
        kl_weight=1/136.965625,
       activation='sigmoid'),
    tfpl.DenseVariational(
        units=7,
        name="dense_var_2",
        make_prior_fn=prior,
        make_posterior_fn=posterior_mean_field,
        kl_weight=1/136.965625,
        activation='sigmoid'),
    Dense(units=tfpl.IndependentNormal.params_size(29),name='output'),
    tfpl.DistributionLambda(lambda x: tfpd.Independent(tfpd.Normal(loc = x[..., :29], scale = tf.nn.softplus( x[..., 29:]))),name='lambda')
  ], name='PBNN')

#Adam optimizer, with MSE metric
Sub_PBNN.compile(keras.optimizers.Adam(learning_rate=0.01),loss=NLL, metrics=keras.metrics.MeanSquaredError())

West2_PBNN.load_weights('SR22_West2_PBNN.h5') #Load the pretrained weights to the network
Sub_PBNN.load_weights('SR22_Sub_PBNN.h5')

"""### Inputs"""

Case = int(input("Enter database (1: NGA-West2, 2: NGA-Sub): "))
######## NGA West2 Model Applicability ########
#Mw : 3.3 – 7.9 (R=1), 5.1 – 7.9 (R=2), 5.9 – 7.6 (R=3), and 6.2 – 7 (R=4) 
#Rrup: < 175 km (R=3),  < 400 km (otherwise). Zhyp < 25 km, Vs30: 90 – 2000 m/s. 

######## NGA Sub Model Applicability ########
#Mw : 4.5 – 8 (R=1), 4 – 7.1 (R=3), 5 – 9.1 (R=4), and 5.7 – 8.8 (R=5) 
#Rrup : 30 – 1000 km (R=1), 32 – 670 km (R=3), 8.3 – 929 (R=4), and 13 – 1000 km (R=5).
#Zhyp : 3 – 40 km (interface) and 20 – 180 km (intraslab). Vs30 : 90 – 2200 m/s. 

Mw = float(input("Enter magnitude: "))
Rrup = float(input("Enter rupture distance (km): "))
Zhyp = float(input("Enter rupture depth (km): "))
Vs30 = float(input("Enter shear wave velocity (m/s): "))
print('Fault flag for both the regions')
print('1: Strike-slip, 2: Normal or Normal-oblique, 3: Reverse or Reverse-oblique')
F = int(input("Enter fault flag: "))
if Case == 1:
  print('Region flag For NGA-West2 region')
  print('1. Alaska, California    2: China, Greece, Iran, Italy, Turkey, USSR')
  print('3. Taiwan                4: Japan, New Zealand')
if Case == 2:
  print('Region flag For NGA-Sub region')
  print('1. Alaska, Cascadia      5: Mexico, Central and South America')
  print('3. Taiwan                4: Japan, New Zealand')
R = int(input("Enter region flag: "))
Inputs = np.array([Mw, np.log(Rrup), Zhyp, np.log(Vs30), F, R])
if Case == 1:
  Input_norm = np.expand_dims(np.divide(Inputs - West2_Input_mean,West2_Input_std),1).T
if Case == 2:
  Input_norm = np.expand_dims(np.divide(Inputs - Sub_Input_mean,Sub_Input_std),1).T


"""### Output"""

count = 100 #number of iterations the network is to be run
y_mean = []
y_std = []
for _ in range(count):
  if Case == 1:
    out_dist = West2_PBNN(Input_norm) #network prediction, which is the distribution
    y_mean.append(tf.cast(out_dist.mean(), tf.float64) * West2_Target_std + West2_Target_mean)  #denormalization of the output
    y_std.append(tf.cast(out_dist.stddev(), tf.float64) * West2_Target_std) #denormalization of the deviations
  if Case == 2:
    out_dist = Sub_PBNN(Input_norm)
    y_mean.append(tf.cast(out_dist.mean(), tf.float64) * Sub_Target_std + Sub_Target_mean)
    y_std.append(tf.cast(out_dist.stddev(), tf.float64) * Sub_Target_std)
aleatoric_std = np.mean(y_std,axis=0)     #computing aleatory variability
epistemic_std = np.std(y_mean, axis=0)    #computing epistemic uncertainty
output = np.mean(y_mean, axis=0)  #predicted output of the network
print('Aleatoric')
print(aleatoric_std)
print('Epistemic')
print(epistemic_std)
print('Mean Prediction')
print(np.exp(output))
periods = [0.01,0.02,0.03,0.04,0.05,0.06,0.075,0.09,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1,1.2,1.5,2,2.5,3,4]
fig, ax = plot.subplots(1,1,figsize=(7,5))
lower_bound = np.squeeze(np.exp(output - 1.96*epistemic_std)[:,3:],axis=0)
upper_bound = np.squeeze(np.exp(output + 1.96*epistemic_std)[:,3:],axis=0)
ax.fill_between(periods, lower_bound, upper_bound, color='b', label='Epistemic Uncertainty', alpha=.2)
lower_bound = np.squeeze(np.exp(output - 1.96*aleatoric_std)[:,3:],axis=0)
upper_bound = np.squeeze(np.exp(output + 1.96*aleatoric_std)[:,3:],axis=0)
ax.fill_between(periods, lower_bound, upper_bound, color='c', label='Aleatory Variability', alpha=.2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(periods,tf.math.exp(tf.squeeze(output,axis=0)[3:]), '--', color='k', label='Mean Prediction', linewidth=1.5)
ax.set_xlabel("Period (s)")
ax.set_ylabel("$\mathregular{S_a}$ (g)")
ax.legend()
plot.title('Mean prediction with 95% probability intervals')