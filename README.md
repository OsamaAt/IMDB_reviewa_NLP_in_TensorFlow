🎬 Sentiment Analysis on IMDB Movie Reviews using TensorFlow
📌 Overview

This project performs binary sentiment classification on the IMDB Movie Reviews dataset using a custom Neural Network built with TensorFlow/Keras.
The goal is to train a model that can automatically determine whether a movie review expresses a positive or negative sentiment.

🧠 Dataset Description

The dataset used is IMDB Reviews, which is publicly available via TensorFlow Datasets (TFDS).
It contains 50,000 labeled reviews (25,000 for training and 25,000 for testing):

Split	Number of Reviews	Task
Train	25,000	Used for model training
Test	25,000	Used for evaluation

Each sample consists of:

Text: a movie review in English

Label: 1 for positive, 0 for negative

⚙️ Project Workflow
1. Load and Explore the Dataset
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)


Displays dataset information and samples of training examples.

2. Text Vectorization
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=120
)
vectorize_layer.adapt(train_reviews)


Builds a vocabulary of the most common 10,000 words.

Converts text reviews into integer sequences of equal length (padding applied).

3. Prepare Data Pipelines

The data is shuffled, cached, and batched for efficient GPU training.

Both train and test sets are optimized using:

.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

4. Build and Compile the Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(120,)),
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


Embedding Layer: Converts word indices into dense 16-dimensional vectors.

Dense Layers: Learn the relationships between features.

Output Layer: Uses a sigmoid function for binary classification.

5. Train the Model
model.fit(train_dataset_final, epochs=5, validation_data=test_dataset_final)


The model is trained for 5 epochs with the Adam optimizer and binary crossentropy loss.

6. Visualize Word Embeddings

After training, the model’s embedding weights are saved as:

meta.tsv → contains the words

vecs.tsv → contains the corresponding vector embeddings

These files can be uploaded to TensorFlow Embedding Projector for visualization:
🔗 https://projector.tensorflow.org/

🧾 Example Output
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 120, 16)           160000
 flatten (Flatten)           (None, 1920)              0
 dense (Dense)               (None, 6)                 11526
 dense_1 (Dense)             (None, 1)                 7
=================================================================
Total params: 171,533
Trainable params: 171,533


Example prediction (simplified):

Input review: "This movie was absolutely fantastic, I loved it!"
Predicted sentiment: Positive ✅

🧰 Technologies Used

Python

TensorFlow / Keras

Authur ✍️ : Osama AT
TensorFlow Datasets (TFDS)

NumPy
