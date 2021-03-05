import tensorflow as tf
from models import DQN

inp = (64, 4)
out = (2)

model = DQN(input_dim=inp,
            output_dim=out)

copy = tf.keras.models.clone_model(model, input_tensors=inp)

print(model.summary())
print(copy.summary())
