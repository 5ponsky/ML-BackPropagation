


public class LayerTanh extends Layer {

  int getNumberWeights() { return 0; }

  LayerTanh(int outputs) {
    super(outputs, outputs);
  }

  // NOTE: this layer has no weights, so the weights vector is unused
  void activate(Vec weights, Vec x) {
    for(int i = 0; i < outputs; ++i) {
      activation.set(i, Math.tanh(x.get(i)));
    }
    // System.out.println("------------------------------------------");
    // System.out.println("TANH activate");
    // System.out.println("input: " + x.toString());
    // System.out.println("weights: " + weights.toString());
    // System.out.println("Computed TANH activation: " + activation.toString());
  }

  // NOTE: this layer contains no weights, so the weights parameter is unused
  void backProp(Vec weights, Vec prevBlame) {
    if(activation.size() != blame.size())
      throw new IllegalArgumentException("derivative problem, vector size mismatch");

    for(int i = 0; i < inputs; ++i) {
      double derivative = blame.get(i) * (1.0 - (activation.get(i) * activation.get(i)));
      prevBlame.set(i, derivative);
    }
  }

  void updateGradient(Vec x, Vec gradient) {
  } // Tanh contains no weights so this is empty
}
