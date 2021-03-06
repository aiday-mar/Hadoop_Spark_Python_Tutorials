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
public static class Map extends MapReduceBase
  implements Mapper<LongWritable, Text, Text, IntWritable> {
  private final static IntWritable one = new IntWritable(1); // here this is the key
  private Text word = new Text();
  
  public void map(LongWritable key, Text value, OutputCollector<Text, IntWritable> output, Reporter reporter) // here we have the key, value pair as well as in the ouput a list of the words followed by the amount of times the word appears
    throws IOException {
      String line = value.toString();
      StringTokenizer tokenizer = new StringTokenizer(line);
      while (tokenizer.hasMoreTokens()){
        word.set(tokenizer.nextToken());
        output.collect(word,one); // you add to the output the specific word and that it appeared once.
      }
```

The reduce code is below :

```
public static class Reduce extends MapReduceBase
  implements Reducer<Text, IntWritable, Text, IntWritable>{
    public void reduce(Text key, Iterator<IntWritable> values, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException {
      int sum=0;
      while (values.hasNext()) { // where here we have next in the iterator
        sum += values.next().get();
      }
      output.collect(key, new IntWritable(sum)); // here we output the word and the given times it appears in the play
    }
  }
```

Now we need a main class :

```
public static void main(String[] args) throws Exception {
  JobConf conf = new JobConf(WordCount.class);
  conf.setJobName("wordCount");
  conf.setOutputKeyClass(Text.class);
  conf.setOutputValueClass(IntWritable.class);
  conf.setMapperClass(Map.class);
  conf.setCombinerClass(Reduce.class);
  conf.setReducerClass(Reduce.class);
  conf.setInputFormat(TextInputFormat.class);
  FileInputFormat.setInputPaths(conf, new Path(args[0]));
  FileOutputFormat.setOutputPaths(conf, new Path(args[1]));
  JobClient.runJob(conf);
}
```

The key components are the mapper, the reducer, the partitioner, the reporter and the output collector.

# MapReduce 2.0

MapReduce only has batch processing, but was slow. Missing security and availability. YARN adds an abstraction layer between HDFS and Map Reduce. Now it is possible for real-time processing. It supports security scenarios too. Let's code the word dount example.

```
public static class Map extends Mapper
  implements Mapper<LongWritable, Text, Text, IntWritable> {
  private final static IntWritable one = new IntWritable(1); // here this is the key
  private Text word = new Text();
  
  public void map(LongWritable key, Text value, Context context) // Here the context contains some data 
    throws IOException {
      String line = value.toString();
      StringTokenizer tokenizer = new StringTokenizer(line);
      while (tokenizer.hasMoreTokens()){
        word.set(tokenizer.nextToken());
        context.write(word, one);
      }
   }
}
```

The reducer class is :

```
public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable>{
    public void reduce(Text key, Iterator<IntWritable> values, Context context) throws IOException {
      int sum=0;
      while (values.hasNext()) { // where here we have next in the iterator
        sum += values.next().get();
      }
      context.write(key, new IntWritable(sum)); // here we use the context
    }
  }
```
The main changes are in the job runner class as follows :

```
public static void main(String[] args) throws Exception {
  Configuration conf = new Configuration();
  Job job = new Job(conf, "wordcount");
  
  job.setOutputKeyClass(Text.class);
  job.setOutputValueClass(IntWritable.class);
  
  job.setMapperClass(Map.class);
  job.setReducerClass(Reduce.class);
  
  job.setInputFormatClass(TextInputFormat.class);
  job.setOutputFormatClass(TextOutputFormat.class);
  
  FileInputFormat.addInputPath(job, new Path(args[0]));
  FileOutputFormat.setOutputPath(job, new Path(args[1]));
  
  job.waitForCompletion(true);
}
```
In the map class you may add more variables such as the following ones :

```
private boolean caseSensitive = true;
private Set<String> patternsToSkip = new HashSet<String>();
```
You can set up a distributed cache which asks the mapper to place specific files on the map node (such as lookup files, translation files). The corresponding Distributed Cach File is :

```
public void configure(JobConf job) {
  caseSensitive = job.getBoolean("wordcount.case.sensitive", true);
  inputFile = job.get("map.input.file");
  if(job.getBoolean("wordcount.skip.patterns", false)){
    Path[] patternsFile = new Path[0];
    try{
      patternsFiles = DistributedCache.getLocalCacheFiles(job);
    } catch(IOException ioe) {
      System.err.println("Caught exception while getting cached files:" + StringUtils.stringifyException(ioe));
    } for (Path patternsFile : patternsFiles){
      parseSkipFile(patternsFile);
    }
  }
}
```

The utility method, parseSkipFile, is defined below :

```
private void parseSkipFile(Path patternsFile) {
  try{
    BufferedReader fis = new BufferedReader(new FileReader(patternsFile.toString()));
    String pattern = null;
    while ((pattern = fis.readLine()) != null) {
      patternsToSkip.add(pattern);
    }
  } catch (IOException ioe) {
    System.err.println("caught exception while parsing the cached file" + patternsFile + " : " + StringUtils.stringifyException(ioe);
  }
}
```

# Hive

Hive libraries are integrated with HBase, they include the HQL language. It is an SQL-lire query language that produces MapReduce code. Here the same example above is written using Hive :

```
CREATE TABLE wordcount AS
SELECT word, count(1) AS count
FROM (SELECT EXPLODE(SPLIT(LCASE
  (REGEXP_REPLACE
    (line, '[\\p{Punct}, \\p{Cntrl}]', '')),' '))
AS word from myinput) words
GROUP BY word
ORDER BY count DESC, word ASC;
```

# Pig

Pig is an ETL (Extract,Transform,Load) library for Hadoop. There are concepts like fields, tuples, bags and relations. The ETL process can be written as :

```
LOAD <file>
FILTER <set> BY <value>=<number>, JOIN, GROUP BY, FOREACH, GENERATE <values>
DUMP <to screen for testing>
STORE <new file>
```

The word count example using Pig can be written as follows : 

```
lines = LOAD '/user/hadoop/HDFS_File.txt' AS (line:chararray);
words = FOREACH lines GENERATE FLATTEN(TOKENIZE(line)) as word;
grouped = GROUP words by word;
wordcount = FOREACH grouped GENERATE group, COUNT(words);
DUMP wordcount;
```
Above we generate a group and count the number of words and dump the words. Here above the reducer is generated in the `wordcount` and the mapper is generated in the `words` and `grouped`.

# Oozie, Sqoop, ZooKeeper

Oozie is a Workflow scheduler library for Hadoop jobs. Sqoop is a command line utility for transferring data between RDBMS systems and Hadoop. There are connectors for Oracle, SQL Server and other services. This data can be loaded into Hive or HBase. The Sqoop syntax is as so : 

```
sqoop import
  --connect <JDBC connection string>
  --table <tablename>
  --username <username>
  --password <password>
    --hive-import
```

You can optimize Sqoop with direct dump. Zookeeper is a centralized service for Hadoop configuration information. It is used for distributed in-memory computation. 

# Spark

Spark is a framework built on top of Hadoop that runs faster than Hadoop. An example code is the following : 

```
public final class CalcPi{
  public static void main(String[] args) throws Exception {
    SparkSession spark = SparkSession.builder().appName("JavaSparkPi").getOrCreate();
    
    JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
    
    // in the below we are parsing 
    int slices = (args.length == 1) ? Integer.parseInt(args[0]) : 2; //outputing either the one character of the args or having 2 slices
    
    int n= 10000*slices;
    List<Integer> l = new ArrayList<>(n) // new array list containing integers of size n
    for (int i=0; i < n; i ++) {
      l.add(i); // meaning that here we are adding i to the end of the array and we are starting with 0
    }
    
    JavaRDD<Integer> dataSet = jsc.parallelize(l, slices); //running in parallel
    
    // below performing the work to find the digits of Pi
    int count = dataSet.map(integer -> {
      double x = Math.random()*2 - 1;
      double y = Math.random()*2 - 1;
      return (x*x + y*y <= 1) ? 1: 0;
    }).reduce((integer, integer2) -> integer + integer2);
    
    System.out.println("Pi is roughly " + 4.0*count/n);
    
    spark.stop();
  }
}
```
Hadoop now runs more on public cloud data lakes. Some visualization libraries are Ganglia or Tableau.
