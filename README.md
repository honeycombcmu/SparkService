# SparkService
Version 2.0

### FEATURES:

1. Start spark via the command line with parameters, the input
parameters are intput/output files path.

2. Use spark to train a machine learning model, and make predictions.

3. Save result as JSON into output path

### Supported machine learning models:
   
* LinearRegression
* NaiveBayes
* RandomForest
* KMeans
* LogisticRegression
* SVM
* Decision Tree

### SETUP INSTRUCTION:

1. Download Spark from [Official Website](http://spark.apache.org/downloads.html) (latest version:1.6.0). Please choose the appropriate package type according to Hadoop version.

2. To build Spark and its example programs, run:
   ``build/mvn``

3. Install python 2.7(should also support Python 3)

4. Login Cluster: 
   ``ssh honeycomb@128.2.7.38`` (password: ask teammates)

5. Copy files into CLuster local host: 
   
   ``scp source_file_name honeycomb@128.2.7.38:/home/honeycomb/SparkTeam``

   e.g:

   ``scp /Users/jacobliu/PySpark.py honeycomb@128.2.7.38:/home/honeycomb/SparkTeam``
   
6. Put files into HDFS:
   
   ``hdfs dfs -put LOCAL_FILE_PATH HDFS_FILE_PATH``
   
   e.g:
   
   ``hdfs dfs -put /home/honeycomb/SparkTeam/sample_multiclass_classification_data_test.txt /user/honeycomb/sparkteam/input``

7. Put PySpark.py and train/test dataset into HDFS and run command line:
   
   ``path/to/spark/spark-submit PySpark.py path/to/training_file path/to/test/file path/to/output``
   
   e.g:
   
   ``/bin/spark-submit /home/honeycomb/SparkTeam/PySpark.py /user/honeycomb/sparkteam/input/sample_multiclass_classification_data.txt /user/honeycomb/sparkteam/input/sample_multiclass_classification_data_test.txt /home/honeycomb/SparkTeam``

### Resources:

1. Deploy Spark: http://spark.apache.org/docs/latest/programming-guide.html

2. Hadoop Version: http://spark.apache.org/docs/latest/hadoop-third-party-distributions.html

3. Python API Docs: https://spark.apache.org/docs/1.5.2/api/python/index.html

4. Machine Learning Library(MLib) Guide: http://spark.apache.org/docs/latest/mllib-guide.html