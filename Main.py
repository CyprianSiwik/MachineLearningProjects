# Cyprian Siwik 12/11/24
# Machine learning intro projects and uses with libraries and models

# This code trains a sentiment analysis model on the IMDB movie reviews dataset
# using a neural network with an embedding layer and an LSTM (Long Short-Term Memory)
# layer. It first loads and preprocesses the data, padding the sequences to a uniform
# length, and then builds a model with an embedding dimension of 100 and a binary output layer.
# After training the model for 20 epochs, it will evaluate the test accuracy,
# plot the training and validation accuracy over epochs,
# and make a sentiment prediction on a sample review.

from tensorflow.keras.datasets import imdb #importing IMDB dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt


#loading in dataset
#num_words is the max limit of top words provided,
# more top words gives more access to the dataset top words.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 2000)

#display the dataset info - in formatted strings
print(f"Training samples: {len(X_train)}") #prints number of training samples
print(f"Testing samples: {len(X_test)}") #pritns number of testing samples
print(f"First review ( as integers ): {X_train[0]}") #prints the first review as list of ints (encoded words)
print(f"Label ( 0 = negative, 1 = positive): {y_train[0]}") #prints the label for the first review - 0=neg, 1=pos

#decode review (map integers back into words)
word_index = imdb.get_word_index() #gets the word index mapping (word -> integer)
reverse_word_index = {value: key for key, value in word_index.items()} #reverse the word index mapping ( integer -> word)
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in X_train[0]]) #decode the first review by converting ints back into words
print(f"Decoded review: {decoded_review}") #print the decoded review

#pad sequences for unifrom input length
maxlen = 500 #maximum review length
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post') #pad training sequences for uniform length
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')   #pad test dequences for uniform length

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

#build model
embedding_dim = 100 #embedding dimensions - vector length of 100
# - if was 1, would basically store words as scalar values with no complex
# way to relate the words to each other.

#initialize the model as sequential (linear stack of layers)
model = Sequential([
    #Embedding layer: converts word indices to dense vectors (embedding_dim = 100)
    #input_dim=10000: size of the vocabulary (top 10,000 words)
    #output_dim=embedding_dim: size of the word vectors (100 dimensions)
    #input_length=maxlen: length of input sequences (max review length)
    Embedding(input_dim=10000, output_dim=embedding_dim, input_length=maxlen),

    # LSTM layer: processes the sequences and captures long-range dependencies
    # 128 units, return_sequences=False (output single vector, not sequence)
    LSTM(128, return_sequences=False),

    # Dense layer: output layer for binary classification (0 or 1 sentiment)
    # Activation function 'sigmoid' outputs a value between 0 and 1
    Dense(1, activation='sigmoid') #binary classification
])

# compile model with adam optimizer, binary cross-entropy loss, and accuracy as the evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#displays the model's architecture - layers, output shapes, number of parameters
model.summary()

#train the model - more epochs may slowly increase accuracy at the cost of computation

#batch size defines how many training samples are included in each batch,
# small batch size - slower processing as more updates are made.
# Large batch size means fewer updates and faster processing time but
# may lack in results.
#validation split is the proportion of training data used as a validation
# set during training. High validation split means more data is used for
# validation , which gives reliable performance estimates but reduces the
# amount of data available for training the model. Low validation split allows
# for more data to be trained but validation results may wane or be noisy.
history = model.fit(X_train, y_train, epochs=6, batch_size=64, validation_split=0.2)

#evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
#test accuracy is a value between 0 and 1, so i.e. 0.76 * 100 == 76%
#:.2f is a format specifier that formats the f-string to present the number
# to 2 decimal places
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy') #plot the training accuracy over epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') #plot the validation accuracy over epochs
plt.xlabel('Epochs') #set the label for the x-axis (epochs)
plt.ylabel('Accuracy') #set the label for the y-axis (accuracy percentage)
plt.legend() #display a legend to differentiate between training and validation accuracy lines
plt.show() #display the plot

#make predictions on a sample review - select a single review from the test set and reshape it for prediction
sample_review = X_test[0].reshape(1,-1) #using a single review
#use the trained model to predict the entiment of the selected review
predicted_sentiment = model.predict(sample_review)
#print the predicted sentiment: 'Positive' if the model output is > 0.5 otherwise 'Negative'
print(f"Predicted Sentiment: {'Positive' if predicted_sentiment > 0.5 else 'Negative'}")

