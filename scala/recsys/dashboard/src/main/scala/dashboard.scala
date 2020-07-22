import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.{DataFrame, SparkSession}

object dashboard extends App {

  override def main(args: Array[String]): Unit = {
    val spark = new SparkSession
    .Builder()
      .appName("dmitry.korniltsev")
      .getOrCreate()

    import spark.implicits._
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    val pwd = spark.conf.get("spark.dashboard.pwd")

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
        $"explode.url".alias("url"),
        $"explode.timestamp". alias("date")
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
      .select($"predictedLabel".alias("gender_age"), $"uid", $"date")

    predictions
        .write
        .format("es")
        .option("org.elasticsearch.spark.sql", "10.0.1.9:9200")
        .option("user", "dmitry_korniltsev")
        .option("password", pwd)
        .save("dmitry_korniltsev_lab08/_doc")

  }
}
