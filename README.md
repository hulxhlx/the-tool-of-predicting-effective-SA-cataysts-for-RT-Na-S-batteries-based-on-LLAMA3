# The tool of predicting effective SA cataysts for RT Na-S batteries based on LLAMA3
This project utilizes **ollama3** for converting Excel tables into embeddings, finding the most similar vectors, performing dimensionality reduction, conducting 10-fold cross-validation, and using a retrieval-augmented generation (RAG) system.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
Each script in this repository serves a specific function as part of the workflow. Follow the steps below to use each script.

### Converting Excel Tables to Embeddings
Convert your Excel data into usable embeddings with the following command:

```bash
python transfer_embedding.py
```

This script reads an Excel file and converts the data into embeddings using the ollama3 model.

### Finding the Top 50 Most Similar Vectors

To find the vectors closest to a specified target vector from the generated embeddings, run:

```bash
python findtop50.py
```

This script assesses the similarity between vectors and lists the top 50.

###  Dimensionality Reduction Using t-SNE

Reduce the dimensionality of your data to make it suitable for visualization:

```bash
python tsne.py
```

This will apply the t-SNE algorithm to the embeddings to help visualize the data in lower dimensions.

### 10-Fold Validation

Validate the model performance using 10-fold cross-validation by executing:

```bash
python 10fold.py
```
This script splits the data into ten parts, using each in turn for testing and the rest for training.

### Using the RAG System

Implement the Retrieval-Augmented Generation model on your embeddings:

```bash
python ragexcel.py
```

This script uses the RAG model to enhance the processing capabilities of the language model with the embeddings data.

