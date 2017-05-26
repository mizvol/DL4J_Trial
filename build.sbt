name := "D4JTest"

version := "1.0"

scalaVersion := "2.10.4"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "2.0.0",
  "org.apache.spark" % "spark-sql_2.10" % "2.0.0",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.8.0",
  "com.typesafe.scala-logging" % "scala-logging_2.11" % "3.5.0",
  "org.slf4j" % "slf4j-api" % "1.7.25",
  "org.nd4j" % "nd4j-native" % "0.8.0" classifier "linux-x86_64",
  "org.nd4j" % "nd4j-native" % "0.8.0"
)
