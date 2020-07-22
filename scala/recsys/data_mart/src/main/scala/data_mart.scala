import org.apache.spark.sql.functions._

import scala.collection.mutable.WrappedArray
import org.apache.spark.sql.cassandra._
import com.datastax.spark.connector.cql.CassandraConnectorConf
import com.datastax.spark.connector.rdd.ReadConf
import org.apache.spark.sql.SparkSession

class data_mart extends App {
  override def main(args: Array[String]): Unit = {

    import spark.implicits._

    val spark = new SparkSession.Builder()
      .appName("dmitry.korniltsev")
      .master("yarn-client")
      .enableHiveSupport()
      .getOrCreate()

    spark.conf.set("spark.cassandra.connection.host", "10.0.1.9")
    spark.conf.set("spark.cassandra.output.consistency.level", "ANY")
    spark.conf.set("spark.cassandra.input.consistency.level", "ONE")

    //Elastic
    val addShop = udf { v: AnyRef =>
      val concat = ("shop_" + v)
      concat
    }

    val esDf = spark.read.format("org.elasticsearch.spark.sql")
      .option("pushdown", "true")
      .option("es.batch.write.refresh", "false")
      .option("es.nodes", "10.0.1.9:9200")
      .load("visits")
      .withColumn("category", lower(regexp_replace('category, "-| ", "_")))
      .withColumn("shop_cat", addShop(col("category")))
      .groupBy("uid")
      .pivot("shop_cat")
      .agg(count(lit(1)))

    //Casandra
    val tableOpts = Map("table" -> "clients","keyspace" -> "labdata")

    val cond_1 = (when($"age".between(18,24), "18-24"))
    val cond_2 = (when($"age".between(25,34), "25-34"))
    val cond_3 = (when($"age".between(35,44), "35-44"))
    val cond_4 = (when($"age".between(45,54), "45-54"))

    val casDf = spark
      .read
      .format("org.apache.spark.sql.cassandra")
      .options(tableOpts)
      .load()
      .withColumn("age_cat", cond_1.otherwise(cond_2.otherwise(cond_3.otherwise(cond_4.otherwise(">=55")))))
      .drop("age")

    //Json
    val jsDf = spark.sqlContext.read.json("hdfs:///labs/laba03/weblogs.json")
      .select('uid, explode(col("visits")).alias("explode"))
      .select('uid, col("explode.timestamp"), col("explode.url"))
      .withColumn("url", regexp_replace('url, "http://http://", "http://"))
      .withColumn("url", regexp_replace('url, "www.", ""))
      .withColumn("url", lower(trim(regexp_replace('url, "\\s+", " "))))
      .withColumn("domain", callUDF("parse_url", $"url", lit("HOST")))
      .groupBy("uid", "domain").agg(count(lit(1)).alias("cnt_visits"))

    //Postgre
    val jdbcUrlLabdata = "jdbc:postgresql://10.0.1.9:5432/labdata?user=dmitry_korniltsev&password=password"
    val addWeb = udf { v: AnyRef =>
      val concat = ("web_" + v)
      concat
    }

    val postgreDf = spark
      .read
      .format("jdbc")
      .option("url", jdbcUrlLabdata)
      .option("driver", "org.postgresql.Driver")
      .option("dbtable", "domain_cats")
      .load()
      .withColumn("category", lower(regexp_replace('category, "-| ", "_")))
      .withColumn("web_cat", addWeb(col("category")))

    //Join all together
    val webCatDf = jsDf
      .join(postgreDf, Seq("domain"), "left")
      .filter(col("web_cat").isNotNull)
      .groupBy("uid")
      .pivot("web_cat")
      .agg(sum("cnt_visits"))

    val joinedDf = casDf
      .join(esDf, Seq("uid"), "left")
      .join(webCatDf, Seq("uid"), "left")

    //Write to database
    val pwd = "password"
    joinedDf.write.mode("overwrite")
      .format("jdbc")
      .option("url", "jdbc:postgresql://10.0.1.9:5432/dmitry_korniltsev")
      .option("dbtable", "clients")
      .option("user", "dmitry_korniltsev")
      .option("password", pwd)
      .option("driver", "org.postgresql.Driver")
      .save()

  }
}
