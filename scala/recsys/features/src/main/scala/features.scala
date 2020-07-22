import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.linalg.{SparseVector, Vector}

object features extends App {

    override def main(args: Array[String]): Unit = {
        val spark = new SparkSession
        .Builder()
          .appName("dmitry.korniltsev")
          .getOrCreate()

        import spark.implicits._
        spark.conf.set("spark.sql.session.timeZone", "UTC")

        val df = spark.read.json("hdfs:///labs/laba03/weblogs.json")
          .select('uid, explode(col("visits")).alias("explode"))
          .select('uid, col("explode.timestamp"), col("explode.url"))
          .withColumn("host", lower(callUDF("parse_url", $"url", lit("HOST"))))
          .withColumn("domain", regexp_replace($"host", "www.", ""))
          .withColumn("hour", hour(from_unixtime(col("timestamp")/1000)))
          .withColumn("week_day", date_format(from_unixtime(col("timestamp")/1000), "E"))

        val df_agg_list = df
          .groupBy("uid")
          .agg(collect_list("domain").alias("agg_domain"))

        val cvModel: CountVectorizerModel = new CountVectorizer()
          .setInputCol("agg_domain")
          .setOutputCol("domain_features")
          .setVocabSize(1000)
          .fit(df_agg_list)

        val asDense = udf((v: Vector) => v.toDense)

        val domainFeatures = cvModel
          .transform(df_agg_list)
          .withColumn("domain_features", asDense(col("domain_features")))
          .drop("agg_domain")

        val weekAgg = df
          .withColumn("week_day", lower(concat(lit("web_day_"), $"week_day")))
          .groupBy("uid")
          .pivot("week_day")
          .agg(count(lit(1)))

        val hourAgg = df
          .withColumn("hour", lower(concat(lit("web_hour_"), $"hour")))
          .groupBy("uid")
          .pivot("hour")
          .agg(count(lit(1)))

        val fractionDF = df
          .groupBy("uid")
          .agg(sum(when($"hour".between(9,17), 1).otherwise(0)).alias("working_time_cnt"),
              sum(when($"hour".between(18,23), 1).otherwise(0)).alias("evening_time_cnt"),
              count(lit(1)).alias("total")
          )
          .withColumn("web_fraction_work_hours", $"working_time_cnt"/$"total")
          .withColumn("web_fraction_evening_hours", $"evening_time_cnt"/$"total")
          .select("uid", "web_fraction_work_hours", "web_fraction_evening_hours")

        val visits = domainFeatures
          .join(weekAgg, Seq("uid"))
          .join(hourAgg, Seq("uid"))
          .join(fractionDF, Seq("uid"))

        val user_items = spark.read.parquet("hdfs:///user/dmitry.korniltsev/users-items/20200429")

        val joined = visits.join(user_items, Seq("uid"))

        joined
          .write
          .mode("overwrite")
          .format("parquet")
          .save("hdfs:///user/dmitry.korniltsev/features")

    }
}
