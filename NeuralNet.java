import java.util.ArrayList;
import java.util.Random;


public class NeuralNet extends SupervisedLearner {
  protected Vec weights;
  protected ArrayList<Layer> layers;

  String name() { return ""; }

  NeuralNet() {

    layers = new ArrayList<Layer>();
  }

  // Initialize a NN with a weights vector
  NeuralNet(int featureSize, int labelSize) {
    layers = new ArrayList<Layer>();

    /// For our project we know we will have 62800 weights (given by gashler)
    // LL1: 62800
    // LL2: 2430
    // LL3: 310
    //weights = new Vec(featureSize + (featureSize * labelSize));
  }

  void initWeights() {
    Random random = new Random(1234);
    int weightsSize = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      weightsSize += l.getNumberWeights();
    }
    weights = new Vec(weightsSize);

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      if(l.getNumberWeights() > 0) {
        for(int j = 0; j < l.getNumberWeights(); ++j) {
          weights.set(pos, Math.max(0.03, (1.0 / l.inputs) * random.nextGaussian()));
          ++pos;
        }
      }
    }

    // for(int i = 0; i < layers.size(); ++i) {
    //   Layer l = layers.get(i);
    //   for(int j = 0; j < l.activation.size(); ++j) {
    //     l.activation.set(j, Math.max(0.03, (1.0 / l.inputs) * random.nextGaussian()));
    //   }
    // }
  }

  void activate() {
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      Vec v = new Vec(weights, 0, 0);
    }
  }

  void backProp(Vec weights, Vec target) {
    weights = this.weights;

    //Vec blame = layers.get(layers.size() - 1).blame;
    layers.get(layers.size() - 1).blame = target;
    layers.get(layers.size() - 1).blame.addScaled(-1, layers.get(layers.size()-1).activation);
    //System.out.println(layers.get(layers.size()-1).blame);

    int pos = weights.size() - 1;
    Vec prevBlame;
    for(int i = layers.size()-1; i > 0; --i) {
      Layer l = layers.get(i);
      //System.out.println(i + " " + l.blame + " " + l.blame.size());
      prevBlame = new Vec(l.inputs);

      int weightsChunk = l.getNumberWeights();
      pos -= weightsChunk;
      Vec w = new Vec(weights, pos, weightsChunk);
      System.out.println(i + ": " + weightsChunk);

      l.backProp(w, prevBlame);
      //System.out.println(prevBlame);
      System.out.println("blame size: " + layers.get(i-1).blame.size());
      layers.get(i-1).blame = new Vec(prevBlame);
      //System.out.println(i + " " + layers.get(layers.size() - (i + 1)).blame);
    }
  }

  void updateGradient() {
    for(int i = 0; i < layers.size(); ++i) {
      //layers.get(i).updateGradient();
    }
  }

  Vec predict(Vec in) {
    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int weightsChunk = (l.outputs + (l.inputs * l.outputs));
      Vec v = new Vec(weights, pos, weightsChunk);
      l.activate(v, in);
      in = l.activation;
      pos += l.activation.size();
    }

    return new Vec(layers.get(layers.size()-1).activation);
  }

  void refineWeights(Vec x, Vec y, Vec weights, double learning_rate) {

  }

  /// Train this supervised learner
  void train(Matrix features, Matrix labels) {
    layers.clear();
    LayerLinear ll = new LayerLinear(features.cols(), labels.cols());
    weights = new Vec(labels.cols() + (features.cols() * labels.cols()));
    layers.add(ll);


    for(int i = 0; i < layers.size(); ++i) {
      //layers.get(i).ordinary_least_squares(features, labels, weights);
    }
  }

}
