# Cyprian Siwik 12/11/24
# Machine learning intro projects and uses with libraries and models

from tensorflow.keras.datasets import imdb #importing IMDB dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt


#loading in dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 10000)

#display the dataset info - in formatted strings
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"First review ( as integers ): {X_train[0]}")
print(f"Label ( 0 = negative, 1 = positive): {y_train[0]}")

#decode review (map integers back into words)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in X_train[0]])
print(f"Decoded review: {decoded_review}")

#pad sequences for unifrom input length
maxlen = 500 #maximum review length
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

#build model
embedding_dim = 100 #embedding dimensions

model = Sequential([
    Embedding(input_dim=10000, output_dim=embedding_dim, input_length=maxlen),
    LSTM(128, return_sequences=False),
    Dense(1, activation='sigmoid') #binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#train the model - more epochs may slowly increase accuracy at the cost of computation
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

#evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#make predictions on a sample review
sample_review = X_test[0].reshape(1,-1) #using a single review
predicted_sentiment = model.predict(sample_review)
print(f"Predicted Sentiment: {'Positive' if predicted_sentiment > 0.5 else 'Negative'}")

