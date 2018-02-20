


public class LayerTanh extends Layer {

  int getNumberWeights() { return 0; }

  LayerTanh(int outputs) {
    super(outputs, outputs);
  }

  void activate(Vec weights, Vec x) {
    for(int i = 0; i < outputs; ++i) {
      activation.set(i, Math.tanh(x.get(i)));
    }
    System.out.println("------------------------------------------");
    System.out.println("TANH activate");
    System.out.println("input: " + x.toString());
    System.out.println("weights: " + weights.toString());
    System.out.println("Computed TANH activation: " + activation.toString());
  }

  void backProp(Vec weights, Vec prevBlame) {
    if(activation.size() != blame.size())
      throw new IllegalArgumentException("derivative problem, vector size mismatch");

    System.out.println("Blame on TANH layer: " + blame.toString());
    for(int i = 0; i < outputs; ++i) {
      double derivative = blame.get(i) * (1.0 - (activation.get(i) * activation.get(i)));
      prevBlame.set(i, derivative);
    }
  }

  void updateGradient(Vec x, Vec gradient) {
    System.out.println("TANH updateGradient");
    System.out.println("input: " + x.toString());
    System.out.println("blame: " + blame.toString() + "\n");
    //System.out.println("computed weights gradient: " + m.toString() + "\n");
  } // Tanh contains no weights so this is empty
}
