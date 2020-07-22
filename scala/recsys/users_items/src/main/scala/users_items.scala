import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, concat, date_format, lit, lower, max, regexp_replace, sum}
import org.apache.spark.sql.types.TimestampType
import java.io.File


object users_items {

  def main(args: Array[String]) = {

    val spark = SparkSession
      .builder
      .appName("Lab05")
      .getOrCreate()

    spark.conf.set("spark.sql.session.timeZone", "UTC")
    val update = spark.conf.get("spark.users_items.update")
    var output_dir = spark.conf.get("spark.users_items.output_dir")
    val input_dir = spark.conf.get("spark.users_items.input_dir")

    var viewDF = spark.read.format("json").json(input_dir + "/view/*")
    var buyDF = spark.read.format("json").json(input_dir + "/buy/*")

    viewDF = viewDF.withColumn("timestamp", date_format((col("timestamp") / 1000).cast(TimestampType), "yyyyMMdd"))
    buyDF = buyDF.withColumn("timestamp", date_format((col("timestamp") / 1000).cast(TimestampType), "yyyyMMdd"))

    val date1 = viewDF.agg(max(col("timestamp"))).collect()(0)(0).asInstanceOf[String]
    val date2 = buyDF.agg(max(col("timestamp"))).collect()(0)(0).asInstanceOf[String]

    var resDate = date1
    if (date2 > date1) {
      resDate = date2
    }

    viewDF = viewDF.groupBy(col("uid"), col("item_id")).count()
    viewDF = viewDF.withColumn("item_id", lower(col("item_id")))
    viewDF = viewDF.withColumn("item_id", regexp_replace(col("item_id"), lit("-"), lit("_")))
    viewDF = viewDF.withColumn("item_id", regexp_replace(col("item_id"), lit(" "), lit("_")))
    viewDF = viewDF.withColumn("item_id", concat(lit("view_"), col("item_id")))
    viewDF = viewDF.groupBy(col("uid")).pivot("item_id").agg(sum(col("count")))

    buyDF = buyDF.groupBy(col("uid"), col("item_id")).count()
    buyDF = buyDF.withColumn("item_id", lower(col("item_id")))
    buyDF = buyDF.withColumn("item_id", regexp_replace(col("item_id"), lit("-"), lit("_")))
    buyDF = buyDF.withColumn("item_id", regexp_replace(col("item_id"), lit(" "), lit("_")))
    buyDF = buyDF.withColumn("item_id", concat(lit("buy_"), col("item_id")))
    buyDF = buyDF.groupBy(col("uid")).pivot("item_id").agg(sum(col("count")))

    var df = viewDF.join(buyDF, Seq("uid"), "full")

    if (update == "1") {
      var prevDFPath = ""
      print("output dir  =  " + output_dir)
      try {

        val fs = FileSystem.get(new Configuration())
        val status = fs.listStatus(new Path(output_dir))

        status.foreach(x => {
          if (x.getPath.toString > prevDFPath) {
            prevDFPath = x.getPath.toString
          }
        })

      } catch {

        case _: Throwable => {

          var prevDFPath = ""
          val prev_dir = output_dir.replace("file://", "")

          val folders: Array[File] = (new File(prev_dir))
            .listFiles
            .filter(_.isDirectory)

          folders.foreach(x => {
            if (x.getPath > prevDFPath) {
              prevDFPath = x.getPath
            }
          })
        }
      }
      if (prevDFPath != "") {
        var prevDF = spark.read.parquet(prevDFPath)
        val sortedCols = df.columns.sorted.map(str => col(str))
        df = df.select(sortedCols:_*)
        prevDF = prevDF.select(sortedCols:_*)
        df = df.union(prevDF)
        var colNames: Seq[String] = df.columns
        colNames = colNames.filter(_ != "uid")
        val exprs = colNames.map(c => sum(c).alias(c))
        df = df.groupBy(col("uid")).agg(exprs.head, exprs.tail: _*)
      }
    }
    df.write.mode("overwrite").format("parquet").save(output_dir + "/" + resDate)
  }

}