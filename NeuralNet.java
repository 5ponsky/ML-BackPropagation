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

  void initWeights(Random r) {

    // Calculate the total number of weights
    int weightsSize = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      weightsSize += l.getNumberWeights();
    }
    weights = new Vec(weightsSize);
    gradient = new Vec(weightsSize);

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);

      int weightsChunk = l.getNumberWeights();
      Vec w = new Vec(weights, pos, weightsChunk);

      l.initWeights(w, r);

      pos += weightsChunk;
    }

    // for(int i = 0; i < weights.size(); ++i) {
    //   System.out.println(weights.vals[i]);
    // }
  }

  // TODO: backprop strips the wrong set of weights

  void backProp(Vec target) {
    weights = this.weights;
    //layers.get(layers.size() - 1).blame = new Vec(target);
    //layers.get(layers.size() - 1).blame.addScaled(-1, layers.get(layers.size()-1).activation);

    Vec blame = new Vec(target.size());
    blame.add(target);
    blame.addScaled(-1, layers.get(layers.size()-1).activation);



    int pos = weights.size();
    for(int i = layers.size()-1; i >= 0; --i) {
      Layer l = layers.get(i);
      //prevBlame = new Vec(l.inputs);

      int weightsChunk = l.getNumberWeights();
      pos -= weightsChunk;
      Vec w = new Vec(weights, pos, weightsChunk);

      blame = l.backProp(w, blame);

      //layers.get(i-1).blame = prevBlame;
    }
  }

  void updateGradient(Vec x) {

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int gradChunk = l.getNumberWeights();
      Vec v = new Vec(gradient, pos, gradChunk);

      l.updateGradient(x, v);
      x = new Vec(l.activation);
      pos += gradChunk;
    }
  }

  void refineWeights(Vec x, Vec y, Vec weights, double learning_rate) {
    gradient.fill(0.0);

    predict(x);

    // Compute the blame on each layer
    backProp(y);

    // Compute the gradient
    updateGradient(x);

    // Adjust the weights per the learning_rate
    this.weights.addScaled(learning_rate, gradient);
  }

  void central_difference(Vec x) {

  }

  Vec predict(Vec in) {
    Vec curr = new Vec(in.size());
    curr.add(in);

    int pos = 0;
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      int weightsChunk = l.getNumberWeights();
      Vec v = new Vec(weights, pos, weightsChunk);
      l.activate(v, in);
      in = l.activation;
      pos += weightsChunk;
    }

    return (layers.get(layers.size()-1).activation);
  }

  /// Train this supervised learner
  void train(Matrix features, Matrix labels) {
    // layers.clear();
    // LayerLinear ll = new LayerLinear(features.cols(), labels.cols());
    // weights = new Vec(labels.cols() + (features.cols() * labels.cols()));
    // layers.add(ll);
    for(int i = 0; i < layers.size(); ++i) {
      //layers.get(i).ordinary_least_squares(features, labels, weights);
    }
  }



}
