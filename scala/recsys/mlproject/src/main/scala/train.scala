import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, StringIndexer, IndexToString}
import org.apache.spark.ml.Pipeline

object train extends App {

  override def main(args: Array[String]): Unit = {
    val spark = new SparkSession
      .Builder()
      .appName("dmitry.korniltsev")
      .getOrCreate()

    import spark.implicits._
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    val pathDir = "/user/dmitry.korniltsev/model"

    var df = spark.read.json("hdfs:///labs/laba07")
    df = df
      .withColumn("_tmp", split($"gender_age", "\\:"))
      .withColumn("explode", explode(col("visits")).alias("explode"))
      .select(
        $"uid", $"visits",
        $"_tmp".getItem(0).as("gender"),
        $"_tmp".getItem(1).as("label"),
        $"explode.timestamp". alias("timestamp"),
        $"explode.url".alias("url")
      )
      .withColumn("host", lower(callUDF("parse_url", $"url", lit("HOST"))))
      .withColumn("domain", regexp_replace($"host", "www.", ""))
      .withColumn("domain", regexp_replace($"domain", "[.]", "-"))

    val data = df
      .groupBy("uid", "label")
      .agg(collect_list("domain").alias("features"))

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    val cv = new CountVectorizer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setVocabSize(1000)
      .fit(data)

    val lr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(cv, labelIndexer, lr, labelConverter))

    val model = pipeline.fit(data)

    model.write.overwrite().save(pathDir+"/spark-logistic-regression-model")

  }
}
