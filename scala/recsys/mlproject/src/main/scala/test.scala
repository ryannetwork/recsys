import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.{DataFrame, SparkSession}

object test extends App {

  override def main(args: Array[String]): Unit = {
    val spark = new SparkSession
      .Builder()
      .appName("dmitry.korniltsev")
      .getOrCreate()

    import spark.implicits._
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    val pathDir = "/user/dmitry.korniltsev/model"
    val input_topic = "dmitry_korniltsev"
    val output_topic = "dmitry_korniltsev_lab04b_out"

    val sameModel = PipelineModel.load(pathDir+"/spark-logistic-regression-model")

    val dfInput = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "10.0.1.13:6667")
      .option("subscribe", input_topic)
      .load()

    var df = dfInput.select('value.cast("string"))

    val schema = StructType(Seq(
      StructField("uid", StringType, true),
      StructField("visits", ArrayType(StructType(
        Seq(StructField("timestamp",LongType,true),
          StructField("url", StringType, true))), true), true)
    ))

    df = df.withColumn("jsonData", from_json(col("value"), schema)).select("jsonData.*")

    df = df
      .withColumn("explode", explode(col("visits")).alias("explode"))
      .select(
        $"uid",
        $"explode.url".alias("url")
      )
      .withColumn("host", lower(callUDF("parse_url", $"url", lit("HOST"))))
      .withColumn("domain", regexp_replace($"host", "www.", ""))
      .withColumn("domain", regexp_replace($"domain", "[.]", "-"))
      .groupBy("uid")
      .agg(collect_list("domain").alias("features"))

    println("="*150)
    println("LOAD MODEL")

    val predictions = sameModel
      .transform(df)
      .select($"uid", $"predictedLabel".alias("gender_age"))

    def createSink(df: DataFrame) = {
      predictions
        .selectExpr("CAST(uid AS STRING) AS key", "to_json(struct(*)) AS value")
        .writeStream
        .trigger(Trigger.ProcessingTime("5 seconds"))
        .format("kafka")
        .option("checkpointLocation", "chk/dmitry_korniltsev_lab04b")
        .option("kafka.bootstrap.servers", "10.0.1.13:6667")
        .option("topic", output_topic)
        .option("maxOffsetsPerTrigger", 200)
        .outputMode("update")
    }

    println("="*150)
    println("START SINK")
    val sink = createSink(df).start

    while (true) {
      sink.awaitTermination(1000)
      println(sink.status)
      println(sink.lastProgress)
    }

  }
}
