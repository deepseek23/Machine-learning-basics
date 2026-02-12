# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
# %%
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv('new.csv')
# %%
df.head()
# %%
X = df.drop(columns=['charges', 'Unnamed: 0'], errors='ignore')
y = np.log1p(df['charges']).astype('float32')

# Keep inputs numeric float32 for Keras
X = X.astype('float32')
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print('train:', X_train.shape, 'val:', X_val.shape, 'test:', X_test.shape)
# %%
normalizer = layers.Normalization()
normalizer.adapt(X_train)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    normalizer,
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.Huber(),
    metrics=[keras.metrics.MeanAbsoluteError(name='mae'), keras.metrics.RootMeanSquaredError(name='rmse')]
)
# %%
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
# %%
# Evaluate on held-out test set
test = model.evaluate(X_test, y_test, verbose=1)
print(dict(zip(model.metrics_names, test)))

# Convert back to original charges scale for interpretability
pred_log = model.predict(X_test, verbose=0).squeeze()
pred = np.expm1(pred_log)
true = np.expm1(y_test)

mae_charges = np.mean(np.abs(pred - true))
rmse_charges = np.sqrt(np.mean((pred - true) ** 2))
print({'mae_charges': float(mae_charges), 'rmse_charges': float(rmse_charges)})