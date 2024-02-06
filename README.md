# ProjetosML

Projects of Machine Learning for learning.

## Supervised Learning

### Wines Classifier

- #### Introduction
A supervised learning project to built a classifier for the variety of a wine based on its description. Original: [Wines Classifier](https://www.toptal.com/machine-learning/nlp-tutorial-text-classification). The project intends to compare two approaches for that task: Naive Bayes, a classical Machine Learning algorithm, and Deep Learning, a technique that has gained great notoriety in the last years.

- #### Technologies
  - [Google Colab](https://colab.research.google.com/)
  - [Python](https://www.python.org/)
  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [TensorFlow](https://www.tensorflow.org/?hl=pt-br)
  - [Keras](https://keras.io/)

- #### Results
  - The Naive Bayes model obtained an accuracy of 62.55%.
  - The Deep Learning model was able to reach 76.45%.
  - Thus, we were able to visualize how models like Deep Learning can deliver the same performance or even outcome traditional models.

## Unsupervised

### K-Means Clustering

- #### Introduction
A clustering (unsupervised learning) project with the goal of identifying two clusters of data from schools through the K-Means algorithm. Then, after that, we see how those clusters fit the expected classifications (Private or Public school, given by a column removed beforehand), evaluating the model performance in that sense with metrics such as a confusion matrix and precision. Original: [K-Means Clustering](https://www.kaggle.com/code/karthickaravindan/k-means-clustering-project/notebook).

- #### Technologies
  - [Google Colab](https://colab.research.google.com/)
  - [Python](https://www.python.org/)
  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)

- #### Results
  - The metrics, such as a confusion matrix with a high rate of False Positives (531) and precision of 0.26 in average, accuse that the patterns picked up by the unsupervised model weren't exactly what we hoped for, since the hope was that one cluster represented the Private and the other the Public schools.
  - That allows us to see how unsupervised learning models may pick patterns that aren't that apparent to humans, which can be a good or bad thing, depending on the expectations one have. In general, they're best used when there's not a lot of expectation of it picking up some specific, desired pattern.
