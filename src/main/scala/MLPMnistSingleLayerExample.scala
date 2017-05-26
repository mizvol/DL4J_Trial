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
import org.slf4j.LoggerFactory
import org.slf4j.Logger
/**
  * Created by volodymyrmiz on 26.05.17.
  */

object MLPMnistSingleLayerExample {
  def main(args: Array[String]): Unit = {

    val log: Logger = LoggerFactory.getLogger(this.getClass)

    val numRows = 28
    val numColumns = 28

    val rngSeed = 123
    val outputNum = 10 // number of output classes
    val numEpochs = 15
    val batchSize = 128

    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.006) //specify the learning rate
      .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
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
      .build();

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)

    model.init()

    model.setListeners(new ScoreIterationListener(1))

    log.info("Train model....")

    for (i <- 0 until numEpochs) model.fit(mnistTrain)

    log.info("Evaluate model....")

    val eval = new Evaluation(outputNum)

    while(mnistTest.hasNext){
      val next = mnistTest.next()
      val output = model.output(next.getFeatureMatrix)
      eval.eval(next.getLabels, output)
    }

    log.info("End....")
  }
}
