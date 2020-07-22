name := "lab03"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.5",
  "org.apache.spark" %% "spark-sql" % "2.4.5",
  "org.apache.spark" %% "spark-mllib" % "2.4.5",
  "org.elasticsearch" %% "elasticsearch-spark-20" % "7.6.2",
  "com.datastax.spark" %% "spark-cassandra-connector" % "2.4.0",
  "org.postgresql" % "postgresql" % "42.2.12"
)

//// https://mvnrepository.com/artifact/org.elasticsearch/elasticsearch-spark-20
//libraryDependencies += "org.elasticsearch" %% "elasticsearch-spark-20" % "7.6.2"
//
//// https://mvnrepository.com/artifact/com.datastax.spark/spark-cassandra-connector
//libraryDependencies += "com.datastax.spark" %% "spark-cassandra-connector" % "2.4.0"
//
//// https://mvnrepository.com/artifact/org.postgresql/postgresql
//libraryDependencies += "org.postgresql" % "postgresql" % "42.2.12"

