import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

object filter extends App {
  override def main(args: Array[String]): Unit = {

    val spark = new SparkSession.Builder()
      .appName("dmitry.korniltsev")
      .getOrCreate()

    val topic_name = spark.conf.get("spark.filter.topic_name")
    val topicOffsetParam = spark.conf.get("spark.filter.offset")
    val pathDir = spark.conf.get("spark.filter.output_dir_prefix")
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    import spark.implicits._

    val buyPath = pathDir + "/buy/"
    val viewPath = pathDir + "/view/"
    val topicOffset = if (topicOffsetParam == "earliest") "earliest" else s"""{"$topic_name":{"0":$topicOffsetParam}}"""

    //READ STREAM
    val kafkaParams = Map (
      "kafka.bootstrap.servers" -> "10.0.1.13:6667",
      "subscribe" -> s"$topic_name",
      "startingOffsets" -> topicOffset,
      "endingOffsets" -> "latest"
    )

    val sdf = spark
      .read
      .format("kafka")
      .options(kafkaParams)
      .load()

    val jsonString = sdf.select('value.cast("string")).as[String]
    val parsed = spark.read.json(jsonString)
      .withColumn("date",  date_format( (col("timestamp")/1000).cast("timestamp"), "YYYYMMDD") )

    //WRITE STREAM
        parsed
          .filter( 'event_type === "view" )
          .write
          .mode("overwrite")
          .partitionBy("date")
          .json(viewPath)

        parsed
          .filter( 'event_type === "buy" )
          .write
          .mode("overwrite")
          .partitionBy("date")
          .json(buyPath)

    spark.close()
  }
}
