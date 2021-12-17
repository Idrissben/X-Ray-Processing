# main.py on GCP

import json
import requests
import tensorflow as tf
import numpy as np
import pickle
import os

batch_size = 8
embedding_dim = 256
units = 512
vocab_size = 5000 + 1
max_length = 144

os.mkdir('/tmp/encoderCNN')
os.mkdir('/tmp/decoderRNN')

## https://www.tensorflow.org/tutorials/text/image_captioning
## Code taken and modified from official Tensorflow website

class CNN_Encoder(tf.keras.Model):
    # This encoder passes the image features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    self.units = units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

  def call(self, x, features, hidden):
    # The weights of the Bahdanau attention are embedded in the __init__
    # These weights are used in the call function as an embedded function
    
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
    attention_weights = tf.nn.softmax(self.V(score), axis=1)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (512, 624))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Inputs image path

def load_models(path):
    tokenizer_path = 'link-to-drive-direct-download'
    encoderCNN_checkpoint_path = 'link-to-drive-direct-download'
    encoderCNN_index_path = 'link-to-drive-direct-download'
    encoderCNN_data12_path = 'link-to-drive-direct-download'
    encoderCNN_data02_path = 'link-to-drive-direct-download'
    decoderRNN_checkpoint_path = 'link-to-drive-direct-download'
    decoderRNN_index_path = 'link-to-drive-direct-download'
    decoderRNN_data12_path = 'link-to-drive-direct-download'
    decoderRNN_data02_path = 'link-to-drive-direct-download'
    
    open(path + '/tokenizer.pkl', 'wb').write(requests.get(tokenizer_path).content)

    open(path + '/encoderCNN/checkpoint', 'wb').write(requests.get(encoderCNN_checkpoint_path).content)
    open(path + '/encoderCNN/.index', 'wb').write(requests.get(encoderCNN_index_path).content)
    open(path + '/encoderCNN/.data-00001-of-00002', 'wb').write(requests.get(encoderCNN_data12_path).content)
    open(path + '/encoderCNN/.data-00000-of-00002', 'wb').write(requests.get(encoderCNN_data02_path).content)

    open(path + '/decoderRNN/checkpoint', 'wb').write(requests.get(decoderRNN_checkpoint_path).content)
    open(path + '/decoderRNN/.index', 'wb').write(requests.get(decoderRNN_index_path).content)
    open(path + '/decoderRNN/.data-00001-of-00002', 'wb').write(requests.get(decoderRNN_data12_path).content)
    open(path + '/decoderRNN/.data-00000-of-00002', 'wb').write(requests.get(decoderRNN_data02_path).content)

    #path = Path('/tmp')

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    encoder.load_weights(path + '/encoderCNN/')
    decoder.load_weights(path + '/decoderRNN/')
    
    with open(path + '/tokenizer.pkl', 'rb') as tok:
      tokenizer = pickle.load(tok)

    return encoder, decoder, tokenizer


def predict(image, encoder, decoder, tokenizer):
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights = 'imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    attention_plot = np.zeros((max_length, 252))
    hidden = decoder.reset_state(batch_size = 1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return ' '.join(result[:-1])

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    # attention_plot is used to output the images from the attention as in the Colab Notebook

    return ' '.join(result[:-1])


def handler(request):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return json.dumps({'msg':'No file part'})
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return json.dumps({'msg':'No file name'})
        if file :
            if file.filename.rpartition('.')[2] in ['jpg','JPG','jpeg','JGEG','png','PNG']:
                saved_file_path='/tmp/'+ file.filename
                file.save(saved_file_path)
                path = '/tmp'
                encoder, decoder, tokenizer = load_models(path)
                result = predict(saved_file_path, encoder, decoder, tokenizer)
                return json.dumps({"File Name":file.filename,
                                   "Predicted Caption":result})
            else:
                return json.dumps({"File Name":file.filename,
                                   "Error":"File type {} not supported".format(file.filename.rpartition('.')[2]),
                                   "Suggestion":"Please choose any of jpg,JPG,jpeg,JGEG,png,PNG"})