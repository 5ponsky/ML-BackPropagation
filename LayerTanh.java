


public class LayerTanh extends Layer {

  int getNumberWeights() { return 0; }

  LayerTanh(int outputs) {
    super(outputs, outputs);
  }

  void activate(Vec weights, Vec x) {
    for(int i = 0; i < outputs; ++i) {
      activation.set(i, Math.tanh(x.get(i)));
    }
  }

  void backProp(Vec weights, Vec prevBlame) {
    System.out.println("tanH blame: " + outputs);
    System.out.println("tanH act: " + activation.size());
    System.out.println("tanh prevB: " + prevBlame.size());
    if(activation.size() != blame.size())
      throw new IllegalArgumentException("derivative problem, vector size mismatch");

    for(int i = 0; i < outputs; ++i) {
      double derivative = blame.get(i) * (1.0 - (activation.get(i) * activation.get(i)));
      prevBlame.set(i, derivative);
    }
  }

  void updateGradient(Vec x, Vec gradient) {} // Tanh contains no weights so this is empty
}
