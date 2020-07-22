package lab02

import scala.math.{sqrt, pow}
import org.apache.spark.sql.{functions => F}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.linalg.{SparseVector}
import org.apache.spark.sql.SparkSession

object RecSysCourses extends App {
  override def main(args: Array[String]): Unit = {

    val id_list = Seq(23325, 15072, 24506, 3879, 1067, 17019)

    def scalarMultiplication(left: SparseVector, right: SparseVector): SparseVector = {
      val intersection = right.indices.intersect(left.indices)
      new SparseVector(right.size, intersection, intersection.map(x => left(x) * right(x)))
    }

    def cosineDist(left: SparseVector, right: SparseVector): Double = {
      scalarMultiplication(left, right).values.sum /
        (sqrt(left.values.map(pow(_, 2)).sum) * sqrt(right.values.map(pow(_, 2)).sum))
    }
    val cosineDistUdf = F.udf(cosineDist _)

    val spark = SparkSession.builder().appName(name = "test").getOrCreate()
    import spark.implicits._

    val df = spark.read.json("/labs/laba02/DO_record_per_line.json")
      .withColumn("desc", F.regexp_replace('desc, "[^\\w\\sа-яА-ЯЁё]", ""))
      .withColumn("desc", F.lower(F.trim(F.regexp_replace('desc, "\\s+", " "))))
      .where(F.length('desc) > 0)

    val courses_target = df.filter(F.col("id").isin(id_list: _*))

    val tokenizer = new Tokenizer().setInputCol("desc").setOutputCol("words")
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20000)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val tf = hashingTF.transform(tokenizer.transform(df))
    val tfidf = idf.fit(tf)

    val courses_features = tfidf
      .transform(tf)
      .select('id, 'features, 'lang)

    val target_features = tfidf
      .transform(hashingTF.transform(tokenizer.transform(courses_target)))
      .select('id.alias("target_id"), 'features.alias("target_features"), 'lang.alias("target_lang"))

    val joined = courses_features
      .join(target_features, 'id =!= 'target_id && 'lang === 'target_lang)
      .withColumn("dist", cosineDistUdf('features, 'target_features))

    val top10courses = joined
      .withColumn("rownum", F.row_number.over(Window.partitionBy('target_id).orderBy('dist.desc)))
      .where('rownum < 11).drop("rownum")

    val grouped_courses = top10courses.groupBy('id)
      .agg(F.collect_list('id).as("collect_ids"))
      .withColumn("target_id", 'id.cast("string"))

    val json = "{" + grouped_courses.withColumn("t",
      F.concat(F.lit("\""), 'id, F.lit("\""),F.lit(":"), F.to_json('collect_ids)))
      .collect().map(_.getString(2)).mkString(",") + "}"

    import java.io.PrintWriter
    new PrintWriter("/data/home/dmitry.korniltsev/lab02.json") { write(json); close }

  }
}
