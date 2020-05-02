# Hadoop, Hive and Pig Tutorial

In the following I have followed the following tutorial on LinkedIn Learning: https://www.linkedin.com/learning/learning-hadoop-2/ . In this repository you may find the work I have done through this course and below are the notes from the tutorial. I have installed Hadoop using the Apache binaries and configured it on windows. The following is a screenshot of http://localhost:9870/ that appears when I run : start-yarn.cmd and hdfs-start.cmd in the command prompt under the \sbin directory of the Hadoop binary files.

<img src="https://github.com/aiday-mar/Images/blob/master/Hadoop_Overview.JPG?raw=true" width = "700">

# Notes

Hadoop uses HDFS (Hadoop Distributed File Systems). HBase is a NoSQL database. The CAP theorem displays three properties that need to be satisfied by databases : Consistency, Availability, Partitioning. These can be understood as the concept of being able to perform transations, be able to access data, perform distributed processing and scale up the database storage.

Hadoop consists of two components and a project : it has an open-source data storage interface HDFS and a processing API called MapReduce. HBase is a library used to display data in a more intuitive way. It has a wide column system meaning that a column can contain different types of data and different amounts of data.

# Hadoop Core Components

Hadoop processes run in separate Java Virtual Machines. The way the framework works is that the client job is sent to the job tracker, which distributes this into portions, each portion sent to a task tracker which further divides the job to task trackers. This part is coded with the MapReduce. The following is a picture depicting the different components of Hadoop.

<img src="https://github.com/aiday-mar/Images/blob/master/Apache_Hadoop_Ecosystem.JPG?raw=true" width = "500">

In the above MapReduce version 2 is also called YARN (Yet Another Resource Navigator) and is built on top of HDFS. HBase is a columnar store. Hive is a query language similar to SQL. Pig is a scripting language used for ETL processes. Mahout is a libary used for Machine Learning and used in Hadoop. Oozie is used for the coordination of jobs. Zookeeper is a system used to coordinate groups of jobs. Sqoop is used to exchange data with other external databases. Flume collects the logs generated by Hadoop during batch processing and display them. While Ambari provisions, manages the hadoop clusters. The cloudera distribution has Hue and HueSDK built on top.

Spark is a processing engine. It processes batches or streams. Spark supports R, SQL, python, scala and java. You can host hadoop on a virtual machine or on docker.

# MapReduce 1.0

MapReduce is a method of programming, designed to solve how to index all the information. Map() run on each node where data lives. HDFS is self healing - will retry on other node that has data. Map outputs set of <key, value> pairs. The reduce part executes the reduce on some set of the data. It aggregates the sets of <key, value> pairs on some nodes and gives a single combined list. There is a shuffle and sort phase too in the map reduce process. When you code you have a static map class, reduce class and a main function which calls the classes above.

A job is a MapReduce instance, a map task runs on each node, a reduce task runs on some nodes, the source of data is HDFS or other. There is a job trackers of which there is one, a task tracker per cluster. 

<img src="https://github.com/aiday-mar/Images/blob/master/Visualizing_Map_Reduce.JPG?raw=true" width = "500">

You can run Hadoop in IDE like Eclipse, or in the command line. Now let's see the code for a Linux environment. To read the above file with Shakespear's play we use :

`hadoop fs -cat <file location> | tails -n50`

The last part of the commmand above selects the last 50 lines. We use a jar file and run it on GDP dataproc. You can submit a job to the cluster in dataproc by specifying the job as a hadoop, specifying where the source is and the output location should be. Then in dataproc you can download the data and see the wordcount. We can visualize the Map Reduce code for the word count example as follows:

<img src="https://github.com/aiday-mar/Images/blob/master/Map_Reduce_Example.JPG?raw=true" width = "500">

The map code is below :

```
publis static class Map extends MapReduceBase
  implements Mapper<LongWritable, Text, Text, IntWritable> {
  private final static IntWritable one = new IntWritable(1); // here this is the key
  private Text word = new Text();
```
