# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Bias playground
#
# Place to play around with toy debiasing techniques

# ## Create dataset

# +
# %matplotlib notebook

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.samples_generator import make_classification
from sklearn.model_selection import train_test_split
# -

# ## Produce correlated datasets

# Using sklearn

# generate 2d classification dataset
X, y = make_classification(
    n_samples=5000,
    n_classes=2,
    n_clusters_per_class=2,
    n_informative=2,
    n_redundant=1,
)

# Using numpy

# +
cov = [
    [1, 0., 0.0],
    [0., 1, 0.9],
    [0.0, 0.9, 1],
]
Xa = np.random.multivariate_normal(
    mean=[0, 0, 0],
    cov=cov,
    size=2000,
)
ya = np.zeros(Xa.shape[0])
Xb = np.random.multivariate_normal(
    mean=[0, 2, 5],
    cov=cov,
    size=2000,
)
yb = np.ones(Xa.shape[0])

X = np.concatenate([Xa, Xb])
y = np.concatenate([ya, yb])
# -

y = y.reshape(-1, 1)
D_cont = X[:, -1:]
X = X[:, :2]

X.shape, D_cont.shape, y.shape

# ### Histogram the debiasing data
#
# This is necessary here to allow all the debiasing (KL divergence) to be differetiable in the loss function.
#
# Want to one-hot encode each event into its bin, later in the loss we can contruct the historgram.

# First find the bin edges
num_bins = 60
_, D_edges = np.histogram(D_cont, bins=num_bins)
D_edges.shape

# Then digitize (discretize) the data
D_digi = np.digitize(D_cont, bins=D_edges[:-1])

# One-hot encode
D = np.zeros((*D_cont.shape[:-1], num_bins))
D[np.arange(D_cont.shape[0]), (D_digi-1).reshape(-1)] = 1
D = D.astype('float32')

# ## Train test split

(
    X_train, X_test,
    D_train, D_test,
    D_cont_train, D_cont_test,
    y_train, y_test
) = train_test_split(
    X, D, D_cont, y, test_size=0.1, random_state=42,
)

X_train.shape, D_train.shape, D_cont_train.shape, y_train.shape

# ## Visualise data

# +
# scatter plot, dots colored by class value
df0 = pd.DataFrame(dict(x=X[:200,0], y=X[:200,1], z=D_cont[:200, 0], label=y[:200, 0]))
df1 = pd.DataFrame(dict(x=X[-200:,0], y=X[-200:,1], z=D_cont[-200:, 0], label=y[-200:, 0]))
df = pd.concat([df0, df1])

# colors = {0:'red', 1:'blue', 2:'green'}
colors = {0:'red', 1:'blue'}
# -

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:200, 0], X[:200, 1], D_cont[:200, 0], c='blue')# cmap='Greens');
ax.scatter3D(X[-200:, 0], X[-200:, 1], D_cont[-200:, 0], c='red',)#, cmap='Greens');
# ax.scatter3D(X[:200, 0], X[:200, 1], D_cont[:200, 0], c=y[:200, 0], cmap='viridis')# cmap='Greens');
# ax.scatter3D(X[-200:, 0], X[-200:, 1], D_cont[-200:, 0], c=y[-200:, 0], cmap='viridis')#, cmap='Greens');
ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')
ax.set_zlabel('D')

fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

# ### Plot correlated variable to decorrelate

fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='y', y='z', label=key, color=colors[key])
plt.show()

# ## Do the learn good

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from tensorflow.keras import regularizers, callbacks, optimizers, backend
try:
    from tensorflow.keras import layers, activations
except ModuleNotFoundError:
    print('Did not find')
    from tensorflow.python.keras import layers, activations

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# TF1
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# TF2
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

backend.clear_session()


# ### Create the model

# +
def make_prediction_model(l2_strength=1e-5):
    
    feature_input = layers.Input(shape=(2), name='feat_input')

    l = layers.Dense(8)(feature_input)
    l = layers.LeakyReLU()(l)
    
    l = layers.Dense(8)(l)
    l = layers.LeakyReLU()(l)
    
    output_layer = layers.Dense(1, activation='sigmoid', name='y')(l)

    model = tf.keras.Model([feature_input], output_layer)
#     model.compile(
#         optimizer=optimizers.Adam(1e-3),
#         loss={
#             'y': 'binary_crossentropy',
#         },
#         metrics=['accuracy'],
#     )
#     model.summary()
    return model


pred_model = make_prediction_model()
# gradients = [backend.gradients(pred_model.output, ww) for ww in pred_model.weights]
# values = [ww for ww in pred_model.weights]

#run = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
#os.makedirs('./logs/' + run + '/plugins/profile', exist_ok=True)
# os.makedirs(os.path.join('model_checkpoints/', skim_wg, run), exist_ok=True)
#print(run)
# -

# ### Define loss function

tf.__version__


def loss(model, x_classify, x_div, y, decorr, gamma=0.001):
  
    # Classification
    y_ = model(x_classify)
    #loss_cla = tf.keras.losses.binary_crossentropy(y, y_)
    bce = tf.keras.losses.BinaryCrossentropy()
    loss_cla = bce(y, y_)
    
    # Decorrelation
    y_ = model(x_div)
    div_hist = tf.multiply(y_, decorr)
    div_hist = tf.math.reduce_sum(div_hist, -2)
    decorr_hist = tf.math.reduce_sum(decorr, -2)

    #print(div_hist.shape, decorr_hist.shape)
    
    KL_div = tf.keras.losses.KLDivergence()
    loss_KL = KL_div(div_hist, decorr)
#     print(loss_cla.numpy(), loss_KL.numpy())
    
    loss_KL = gamma * loss_KL
    
    loss_final = loss_cla + loss_KL
    return loss_final, loss_cla, loss_KL


# ### Define gradient function

def grad(model, x_classify, x_div, y, decorr):
    with tf.GradientTape() as tape:
        loss_value, loss_cla, loss_KL = loss(model, x_classify, x_div, y, decorr)
    return loss_value, loss_cla, loss_KL, tape.gradient(loss_value, model.trainable_variables)


# ### Choose an optimiser

optimizer = optimizers.Adam(1e-3)

# ### Run the training

# #### Need a tf Dataset for batch looping

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, D_train, y_train[:, :]))
train_dataset = train_dataset.batch(500)

next(iter(train_dataset))

# +
## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_loss_cla_results = []
train_loss_KL_results = []
train_accuracy_results = []

num_epochs = 10
small_batch_size = 12

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_loss_cla_avg = tf.keras.metrics.Mean()
    epoch_loss_KL_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

    # Outer training loop - using batches of 500
    for feats_big, decorr_big, labels_big in train_dataset:
        
        # Inner training loop - batch size of 12
        for step in range(int(labels_big.shape[0] / small_batch_size)):
            # Fetch the subset of data
            feats_small = feats_big[step * small_batch_size:(step + 1) * small_batch_size]
            labels_small = labels_big[step * small_batch_size:(step + 1) * small_batch_size]
            # Optimize the model
            loss_value, loss_cla, loss_KL, grads = grad(pred_model, feats_small, feats_big, labels_small, decorr_big)
            optimizer.apply_gradients(zip(grads, pred_model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            epoch_loss_cla_avg(loss_cla)  # Add current batch loss
            epoch_loss_KL_avg(loss_KL)  # Add current batch loss
            # Compare predicted label to actual label
            epoch_accuracy(labels_small, pred_model(feats_small))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_loss_cla_results.append(epoch_loss_cla_avg.result())
    train_loss_KL_results.append(epoch_loss_KL_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

#     if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Loss(cla): {:.3f}, Loss(KL): {:.3f}, Accuracy: {:.3%}".format(
        epoch,
        epoch_loss_avg.result(),
        epoch_loss_cla_avg.result(),
        epoch_loss_KL_avg.result(),
        epoch_accuracy.result()))
# -

# ## Plot the bias

threshold = 0.5

preds = pred_model.predict(train_dataset)

# Get the predicted histogram contributions to the decorrelation vars

D_pred = preds * D_train

# Need to compare whether the entries from one class and their predictions are correlated

true_idx = np.argwhere((y_train == 1))[:, 0]

true_idx.shape

# True events
D_true = D_train[true_idx]
decorr_pred_y = preds[true_idx].reshape(-1, 1)

D_true.shape, decorr_pred_y.shape

# True positive events
true_pass_idx = np.argwhere(((y_train == 1) & (preds > threshold)))[:, 0]
D_true_pos = D_train[true_pass_idx]

D_true_pos.shape, true_pass_idx.shape

true_idx, pass_idx

# Convert to histograms

D_hist = np.sum(D_true, -2)
D_pred_hist = np.sum(D_true_pos, -2)

D_hist

# Plot them against each other

plt.figure()
# plt.plot(D_edges[:-1], D_hist, label='True')
# plt.plot(D_edges[:-1], D_pred_hist, label='True Pos')
plt.bar(D_edges[:-1], D_hist, label='True', align='edge', width=0.1)
plt.bar(D_edges[:-1], D_pred_hist, label='True Pos', align='edge', width=-0.1)
plt.legend()

# ### Plot the correlation
#
# Check classifier output vs. decorrelation variable

preds.shape, D_cont_train.shape

np.corrcoef(preds[true_pass_idx, 0], D_cont_train[true_pass_idx, 0])

plt.figure()
plt.scatter(
    preds[true_pass_idx],
    D_cont_train[true_pass_idx],
    c=y_train[true_pass_idx],
    s=1
)
plt.xlabel('Prediction')
plt.ylabel('D')
