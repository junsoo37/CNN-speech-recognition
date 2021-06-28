from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow.compat.v1 as tf
import numpy as np
import melSpectrogram
import cnn_model
import time

t1 = time.time()
tf.logging.set_verbosity(tf.logging.INFO)
# convert wav files into mel spectrogram, assign a label(answer: command_num) to each mel spectrogram
train_data, train_labels, eval_data, eval_labels = melSpectrogram.ReadTrainingSet()

train_data = np.array(train_data)
eval_data = np.array(eval_data)
train_labels = np.array(train_labels)
eval_labels = np.array(eval_labels )
keywords = ["ALEXA", "BIXBY", "GOOGLE", "JINIYA", "KLOVA"]

# Create the Estimator
train_eval_cnn_model = tf.estimator.Estimator(model_fn=cnn_model.cnn_model, model_dir="./weight_bias_dir")

# Set up logging for predictions
#tensors_to_log = {"probabilities": "softmax_tensor"}
#logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000) # print tensors_to_log in 1000 epochs

# Train the model
train_input=tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,batch_size=4,num_epochs=60,shuffle=True)
train_eval_cnn_model.train(input_fn=train_input) 
#train_input=tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,batch_size=4,num_epochs=None,shuffle=True)
#train_eval_cnn_model.train(input_fn=train_input, steps=50000)
#train_eval_cnn_model.train(input_fn=train_input, steps=20000, hooks=[logging_hook])
# Test the optimized model to 20% of data not used for training
eval_input=tf.estimator.inputs.numpy_input_fn(x={"x":eval_data},y=eval_labels,num_epochs=1,shuffle=False)
eval_results = train_eval_cnn_model.evaluate(input_fn=eval_input)

predict_input = tf.estimator.inputs.numpy_input_fn( x={"x": eval_data}, y=None, batch_size=1,   num_epochs=1, shuffle=False)
predict_results = train_eval_cnn_model.predict(input_fn=predict_input)
num = 0 
for pred_dict in predict_results:
      template = ('Prediction: "{}" ({:.1f}%) "{}", for Answer: "{}"')
      command_num_predicted = pred_dict['classes']
      probability = pred_dict['probabilities'][command_num_predicted]
      num = num + 1
      print(template.format(keywords[command_num_predicted], 100* probability, str(num), keywords[eval_labels[num-1]]), end="\t")
      if(command_num_predicted == eval_labels[num-1]):
          print("O")
      else:
          print("XXXXXXXXXX")
t2=time.time()
print(t2-t1)