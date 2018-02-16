import java.util.ArrayList;


public class NeuralNet extends SupervisedLearner {
  protected Vec weights;
  //protected ArrayList<Layer> layers;
  protected ArrayList<LayerLinear> layers;

  NeuralNet() {
    layers = new ArrayList<LayerLinear>();
  }

  String name() {
    return "Linear Regression";
  }

  void addLayer(int inputs, int outputs) {
    layers.add(new LayerLinear(inputs, outputs));
  }

  void activate() {
    for(int i = 0; i < layers.size(); ++i) {
      Layer l = layers.get(i);
      Vec v = new Vec(weights, 0, 0);
    }
  }

  void backProp(Vec weights, Vec target) {
    // Backpropagate

    // Compute blame for output layer
    Vec outputBlame = layers.get(layers.size() - 1).blame;
    outputBlame.copy(target);
    Vec nActvation = layers.get(layers.size() - 1).activation;
    nActvation.scale(-1);
    outputBlame.add(nActvation);

    for(int i = layers.size() - 2; i >= 0; --i) {
      layers.get(i).backProp(weights, outputBlame);
    }
  }

  void updateGradient() {
    for(int i = 0; i < layers.size(); ++i) {
      layers.get(i).updateGradient();
    }
  }

  Vec predict(Vec in) {
    for(int i = 0; i < layers.size(); ++i) {
      layers.get(i).activate(weights, in);
    }

    return new Vec(layers.get(0).activation);
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
      layers.get(i).ordinary_least_squares(features, labels, weights);
    }
  }

}
