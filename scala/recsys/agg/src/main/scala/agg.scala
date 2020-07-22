import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.ScalaReflection

case class KafkaLogs(
                      event_type: String,
                      category: String,
                      item_id: String,
                      item_price: String,
                      uid: String,
                      timestamp: Long
                    )

object agg extends App{

  override def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("dmitry.korniltsev")
      .getOrCreate()

    import spark.implicits._

//    val schema = StructType(Seq(
//      StructField("category", StringType, true),
//      StructField("event_type", StringType, true),
//      StructField("item_id", StringType, true),
//      StructField("item_price", StringType, true),
//      StructField("timestamp", LongType, true),
//      StructField("uid", StringType, true)
//    ))

    val schema = ScalaReflection.schemaFor[KafkaLogs].dataType.asInstanceOf[StructType]

    val dfInput = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "10.0.1.13:6667")
      .option("subscribe", "dmitry_korniltsev")
      .load()

    var df = dfInput.select('value.cast("string"))

    df = df
      .withColumn("jsonData", from_json(col("value"), schema)).select("jsonData.*")
      .withColumn("date", ($"timestamp" / 1000).cast(TimestampType))
      .groupBy(window(col("date"), "1 hours"))
      .agg(
        sum(when(col("event_type") === "buy", col("item_price")).otherwise(0)).alias("revenue"),
        sum(when(col("uid").isNotNull, 1).otherwise(0)).alias("visitors"),
        sum(when(col("event_type") === "buy", 1).otherwise(0)).alias("purchases")
      )
      .withColumn("aov", col("revenue") / col("purchases"))
      .withColumn("start_ts", col("window.start").cast("long"))
      .withColumn("end_ts", col("window.end").cast("long"))
      .drop(col("window"))

    def createConsoleSink(df: DataFrame) = {
      df
        .selectExpr("CAST(start_ts AS STRING) AS key", "to_json(struct(*)) AS value")
        .writeStream
        .trigger(Trigger.ProcessingTime("5 seconds"))
        .format("kafka")
        .option("checkpointLocation", "chk/dmitry_korniltsev")
        .option("kafka.bootstrap.servers", "10.0.1.13:6667")
        .option("topic", "dmitry_korniltsev_lab04b_out")
        .option("maxOffsetsPerTrigger", 200)
        .outputMode("update")
    }

    val sink = createConsoleSink(df)
    sink.start.awaitTermination()

    spark.close()

  }

}