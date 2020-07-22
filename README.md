## Recommendation Systems
 The purpose of this repository is to collect common techniques in recommendation systems such as 

 Every RecSys consists of 3 parts:
 1. Splitting data into significant groups in order to proccess it individually
 2. Use one of the algorithms such as Apriori (association rules), Collaborative filtering, Neural Networks embeddings
 3. A/B test for understanding if your model is working or not

## Repository composition
 * apriori - association rules for understanding what pairs and triplets of products could be bought together. Here also you can fing network chart
 * core - files for core of project. Mostly it is about creating tabular neural network which solves classification problems
    > data_process - measure for estimating a similarity between objects
 * notebooks - notebook examples with implimintation of recsys algorithms, including LightFM implementation. This is not a production ready code. It requires cleaning.
 * examples - notebook examples with implimintation of recsys algorithms. This is not a production ready code. It requires cleaning.
 * data - movie lens dataset (training dataset).
 * pypspark - Tf Idf algorithm wich is used for creating text embeddings. These embeddings are used for finding simmilar objects.
 * scala - creating recomendation system on Scala. The folder is separated into scala-projects because all of them a launched on Hadoop-server separatley from each other.