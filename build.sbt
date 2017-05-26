name := "D4JTest"

version := "1.0"

scalaVersion := "2.11.0"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.1.1",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.8.0",
  "com.typesafe.scala-logging" % "scala-logging_2.11" % "3.5.0",
  "org.slf4j" % "slf4j-api" % "1.7.25",
  "org.nd4j" % "nd4j-native" % "0.8.0" classifier "linux-x86_64",
  "org.nd4j" % "nd4j-native" % "0.8.0",
  "org.deeplearning4j" % "dl4j-spark_2.11" % "0.8.0_spark_2"
)
