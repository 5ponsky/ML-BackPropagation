import java.util.ArrayList;
import java.util.Random;


public class NeuralNet extends SupervisedLearner {
  protected Vec weights;
  protected Vec gradient;
  protected ArrayList<Layer> layers;

  String name() { return ""; }

  NeuralNet() {
    layers = new ArrayList<Layer>();
  }

  void initWeights() {
    Random random = new Random(1234);

    // Calculate the total number of weights
    int weightsSize = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      weightsSize += l.getNumberWeights();
    }
    weights = new Vec(weightsSize);
    gradient = new Vec(weightsSize);
    gradient.fill(0.0);

    // Randomize the values of the weights
    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      if(l.getNumberWeights() > 0) {
        for(int j = 0; j < l.getNumberWeights(); ++j) {
          weights.set(pos, Math.max(0.03, (1.0 / l.inputs)) * random.nextGaussian());
          ++pos;
        }
      }
    }
  }

  // TODO: backprop strips the wrong set of weights

  void backProp(Vec weights, Vec target) {
    weights = this.weights;

    // Blame for the outputs layer is computed as:
    // blame = target - activation
    layers.get(layers.size() - 1).blame = target;
    layers.get(layers.size() - 1).blame.addScaled(-1, layers.get(layers.size()-1).activation);
    System.out.println("Output layer blame: " + layers.get(layers.size()-1).blame);


    int pos = weights.size();
    Vec prevBlame;
    for(int i = layers.size()-1; i > 0; --i) {
      Layer l = layers.get(i);

      // prevBlame := inputs of this layer (outputs of prevLayer)
      prevBlame = new Vec(l.inputs);

      // Calculate the chunk of the weights vector
      // That we need for backProp
      int weightsChunk = l.getNumberWeights();
      pos -= weightsChunk;
      Vec w = new Vec(weights, pos, weightsChunk);

      System.out.println("Segmented chunk of weights: " + w.toString());
      System.out.println("Starts at: " + pos + " with length: " + weightsChunk);

      // Compute the blame for the preceding layer
      // preceding := closer to layers.get(0)
      l.backProp(w, prevBlame);
      layers.get(i-1).blame = new Vec(prevBlame);
      System.out.println("Computed blame on prev layer: " + prevBlame.toString());
    }
  }

  void updateGradient(Vec x) {
    gradient = new Vec(weights.size());
    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int gradChunk = l.getNumberWeights();
      Vec v = new Vec(gradient, pos, gradChunk);

      l.updateGradient(x, v);
      x = l.activation; // I think the problem with update gradient is here
      // I'm not calling activate each epoch to compute a different
      // distance from the preferred value?
      // I need to work in activate into my epoch somehow
      pos += gradChunk;
    }
  }

  void refineWeights(Vec x, Vec y, Vec weights, double learning_rate) {
    weights = this.weights;

    // Compute the blame on each layer
    backProp(weights, y);

    // Compute the gradient
    updateGradient(x);

    // Adjust the weights per the learning_rate
    weights.addScaled(learning_rate, gradient);
    System.out.println("weights: " + weights.toString());
  }

  void central_difference(Vec x) {

  }

  Vec predict(Vec in) {
    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int weightsChunk = l.getNumberWeights();
      Vec v = new Vec(weights, pos, weightsChunk);
      System.out.println("Weights: " + v.toString());
      l.activate(v, in);
      in = l.activation;
      pos += l.activation.size();
    }

    System.out.println("FINAL LAYER ACTIVATION: " + layers.get(layers.size()-1).activation);
    return new Vec(layers.get(layers.size()-1).activation);
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
