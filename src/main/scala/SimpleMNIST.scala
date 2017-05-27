import org.apache.spark.sql.SparkSession
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.slf4j.LoggerFactory
import org.slf4j.Logger
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster

/**
  * Created by volodymyrmiz on 26.05.17.
  */

object MLPMnistSingleLayerExample {
  def main(args: Array[String]): Unit = {

    val log: Logger = LoggerFactory.getLogger(this.getClass)

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("DL4J MNIST")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.executor.memory", "50g")
      .getOrCreate()

    val sc = spark.sparkContext

    val examplesPerDataSetObject = 32

    val numRows = 28
    val numColumns = 28

    val rngSeed = 123
    val outputNum = 10 // number of output classes
    val numEpochs = 5
    val batchSize = 128

    /**
      * Reading data
      */

    log.info("Reading data")
    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    val trainData = scala.collection.mutable.ListBuffer.empty[DataSet]
    while(mnistTrain.hasNext) trainData += mnistTrain.next()

    val testData = scala.collection.mutable.ListBuffer.empty[DataSet]
    while(mnistTest.hasNext) testData += mnistTest.next()

    val trainRDD = sc.parallelize(trainData)
    val testRDD = sc.parallelize(testData)

    /**
      * Building a NN
      */
    log.info("Building a model")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1) //specify the learning rate
      .updater(Updater.ADAM).momentum(0.9) //specify the rate of change of the learning rate.
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
        .nIn(numRows * numColumns)
        .nOut(1000)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
        .nIn(1000)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .build())
      .pretrain(false).backprop(true) //use backpropagation to adjust weights
      .build()

    val model = new MultiLayerNetwork(conf)

    model.init()

    val tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
      .workerPrefetchNumBatches(0)
      .saveUpdater(true) //save things like adagrad squared gradient histories
      .averagingFrequency(5) //Do 5 minibatch fit operations per worker, then average and redistribute parameters
      .batchSizePerWorker(examplesPerDataSetObject) //Number of examples that each worker uses per fit operation
      .build()

    val sparkNetwork = new SparkDl4jMultiLayer(sc, model, tm)

    log.info("--- Starting network training ---")

    for (i <- 0 to numEpochs) {
      sparkNetwork.fit(trainRDD)
      println("----- Epoch " + i + " complete -----")

      val evaluation = sparkNetwork.evaluate(testRDD)
      println(evaluation.stats())
    }

//
//    model.setListeners(new ScoreIterationListener(1))
//
//    log.info("Train model....")
//
//    for (i <- 0 until numEpochs) model.fit(mnistTrain)
//
//    log.info("Evaluate model....")
//
//    val eval = new Evaluation(outputNum)
//
//    while(mnistTest.hasNext){
//      val next = mnistTest.next()
//      val output = model.output(next.getFeatureMatrix)
//      eval.eval(next.getLabels, output)
//    }
//
//    log.info(eval.stats())

    log.info("End....")
  }
}
