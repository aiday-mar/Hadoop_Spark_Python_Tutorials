# Hadoop, Hive and Pig Tutorial

In the following I have followed the following tutorial on LinkedIn Learning: https://www.linkedin.com/learning/learning-hadoop-2/ . In this repository you may find the work I have done through this course and below are the notes from the tutorial. I have installed Hadoop using the Apache binaries and configured it on windows. The following is a screenshot of http://localhost:9870/ that appears when I run : start-yarn.cmd and hdfs-start.cmd in the command prompt under the \sbin directory of the Hadoop binary files.

<img src="https://github.com/aiday-mar/Images/blob/master/Hadoop_Overview.JPG?raw=true" width = "700">

# Notes

Hadoop uses HDFS (Hadoop Distributed File Systems). HBase is a NoSQL database. The CAP theorem displays three properties that need to be satisfied by databases : Consistency, Availability, Partitioning. These can be understood as the concept of being able to perform transations, be able to access data, perform distributed processing and scale up the database storage.

Hadoop consists of two components and a project : it has an open-source data storage interface HDFS and a processing API called MapReduce. HBase is a library used to display data in a more intuitive way. It has a wide column system meaning that a column can contain different types of data and different amounts of data.

# Hadoop Core Components

Hadoop processes run in separate Java Virtual Machines.
