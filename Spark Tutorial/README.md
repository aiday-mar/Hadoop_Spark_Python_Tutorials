# Spark

In this repository you may find the work I have done while learning Spark with the following LinkedIn tutorial : https://www.linkedin.com/learning/apache-spark-essential-training/ . Hereby you may find notes I took during the tutorial. 

**Analyzing Data in Spark**

The tutorial is done using the Databricks website. I have created a cluster that is running Spark 2.4.5 and Scala 2.1.1. Then I use the following commands.

To read a dataframe : 

```
df = spark.read.load(path,
                    format='com.databricks.spark.csv', 
                    header='true',
                    inferSchema='true')

display(df)
```

The above infers the schema and you can display the schema using the command *df.printSchema()*. You can use commands like select(), orderby("column name") and distinct() which reflect the commands in standard SQL.

In the below : 

```
display(
  df
    .select(df["Country"], df["Description"],(df["UnitPrice"]*df["Quantity"]).alias("Total"))
    .groupBy("Country", "Description")
    .sum()
    .filter(df["Country"]=="United Kingdom")
    .sort("sum(Total)", ascending=False)
    .limit(10)
)
```
  
We have selected the country and the description column, as well as the total revenu which is given an alias as total. We group by the Country and the Descritption and select the entries where the country is the UK. Then we take the sum of the total for this selection and display the results in a descending order. Furthermore we display the top 10 results. The results are displayed below.

<img src="https://github.com/aiday-mar/Images/blob/master/Spark_Query.JPG" width="600"/>

Then in order to save the table you can simply write :

`df.write.saveAsTable("product_sales_by_country")`

**Using Spark SQL to analyze data**

To create a table using a CSV file and a known schema :

```
CREATE TABLE IF NOT EXISTS online_retail(
  InvoiceNo string,
  StockCode string,
  Description string,
  Quantity int,
  InvoiceDate string,
  UnitPrice double,
  CustomerID int,
  Country string)
  USING CSV
OPTIONS (path "/databricks-datasets/online_retail/data-001/data.csv", header "true");
```

`select * from online_retail limit 100;`

I perform a left join as follows :

```
select CompanyName, IPOYear, Symbol, round(sum(SaleAmount)) as Sales from cogsley_sales
  left join cogsley_clients on CompanyName = Name
  group by CompanyName, IPOYear, Symbol
  order by 1
```

Here above I summed the sale amounts and round up the result, and display this as Sales. I left joined one table onto the second so that we match the company name to the name, and we order the results according to the company name. Using Databricks it is possible to visualize the data as follows :

<img src="https://github.com/aiday-mar/Images/blob/master/Databricks_Graph.JPG" width="400"/>
<img src="https://github.com/aiday-mar/Images/blob/master/Databricks_Map.JPG" width="400"/>

**Machine Learning with MLLib**

Then you can download the data using the `curl` command. After downloading the data it has to be loaded :

```
path = 'file:/databricks/driver/CogsleyServices-SalesData-US.csv'

data = sqlContext.read.format("csv")\
  .option("header", "true")
  .option("inferSchema", "true")\
  .load(path)
 
data.cache()
data = data.dropna()
display(data)
```

Then I select the appropriate columns :

```summary = data.select("OrderMonthYear","SaleAmount").groupBy("OrderMonthYear").sum().orderBy("OrderMonthYear").toDF("OrderMonthYear","SaleAmount")```

And each line of the summary is transformed so that the first entry has dashes removed and changed to emptiness and furthermore the entry is changed into an integer. 

```
results = summary.map(lambda r: (int(r.OrderMonthYear.replace('-','')), r.SaleAmount)).toDF(["OrderMonthYear","SaleAmount"])
```

Then we create the features and the labels, the features will be used to predict the value of the label.

```
from pyspark.mllib.regression import LabeledPoint
 
data = results.select("OrderMonthYear", "SaleAmount")\
  .map(lambda r: LabeledPoint(r[1], [r[0]]))\
  .toDF()
  
display(data)
```

Above both the *OrderMonthYear* and the *SaleAmount* are features. We then build the linear regression model :

```
from pyspark.ml.regression import LinearRegression
 
lr = LinearRegression()
 
modelA = lr.fit(data, {lr.regParam:0.0})
modelB = lr.fit(data, {lr.regParam:100.0})

predictionsA = modelA.transform(data)
predictionsB = modelB.transform(data)

display(predictionsA)
```

Here above we first fit a linear function through the given data above with the right regularization parameter, then we need to apply the transformation once again onto the data to get a prediction. Typically there is a separation between test data and data actually used for the prediction. 

Next we need to evaluate the model and we use the root-mean square parameter for that, which we can find easily.

```
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse")
```

Above we specified the root mean square as the evaluating function. Next we apply this on the actual predictions and find the RMSE value and output it.

```RMSE = evaluator.evaluate(predictionsA)
print("ModelA: Root Mean Squared Error = " + str(RMSE))

RMSE = evaluator.evaluate(predictionsB)
print("ModelB: Root Mean Squared Error = " + str(RMSE))
```

Next we want to display the data and the prediction in a table so we execute the following code :

`cols = ["OrderMonthYear", "SaleAmount", "Prediction"]`

Then we need display the feature as a float, the label and the prediction, for each line of the spark context.

```
tableA = sc.parallelize(\
            predictionsA.map(lambda r: (float(r.features[0]), r.label, r.prediction)).collect()\
         ).toDF(cols)

tableB = sc.parallelize(\
            predictionsB.map(lambda r: (float(r.features[0]), r.label, r.prediction)).collect()\
         ).toDF(cols)

tableA.write.saveAsTable('predictionsA', mode='overwrite')
print "Created predictionsA table"

tableB.write.saveAsTable('predictionsB', mode='overwrite')
print "Created predictionsB table"
```

Then I visualize the data using the following code :

```
%sql 
select 
    a.OrderMonthYear,
    a.SaleAmount,
    a.prediction as ModelA,
    b.prediction as ModelB
from predictionsA a
join predictionsB b on a.OrderMonthYear = b.OrderMonthYear
```

In Databricks it is possible to plot the data as follows :

<img src="https://github.com/aiday-mar/Images/blob/master/Linear_Regression_MLlib_Spark.JPG" width="400"/>

**Spark Streaming**

Micro batching is the operation of achieving real-time data processing by chuking operations into smaller batches. SQL queries can also be formulated through a streaming interface.

```
from pyspark.sql.functions import *

streamingInputDF = (
  spark
    .readStream                       
    .schema(salesSchema)             
    .option("maxFilesPerTrigger", 1)
    .csv(path)
)

streamingCountsDF = (                 
  streamingInputDF
    .select("ProductKey", "SaleAmount")
    .groupBy("ProductKey")
    .sum()
)

streamingCountsDF.isStreaming
```

Create then a streaming table, this time we are writing the data :

```
query = (
  streamingCountsDF
    .writeStream
    .format("memory")      
    .queryName("sales_stream")     
    .outputMode("complete") 
    .start()
)
```
