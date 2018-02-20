import java.util.Arrays;


public class LayerLinear extends Layer {

  int getNumberWeights() { return (outputs + (inputs * outputs)); }

  LayerLinear(int inputs, int outputs) {
    super(inputs, outputs);
  }

  void activate(Vec weights, Vec x) {
    Vec b = new Vec(weights, 0, outputs);
    int pos = outputs;
    for(int i = 0; i < outputs; ++i) {
      Vec temp = new Vec(weights, pos, inputs);
      double newEntry = x.dotProduct(temp);
      activation.set(i, newEntry);
      pos += inputs;
    }
    activation.add(b);
    //
    System.out.println("------------------------------------------");
    System.out.println("LIN activate");
    System.out.println("input: " + x.toString());
    System.out.println("weights: " + weights.toString());
    System.out.println("Computed LINEAR activation: " + activation.toString());
  }

  void backProp(Vec weights, Vec prevBlame) {
    System.out.println("LIN Backprop weights: " + weights.toString());
    int pos = outputs; // Ignore b
    Matrix mTranspose = new Matrix(inputs, outputs);
    for(int i = 0; i < outputs; ++i) {
      Vec v = new Vec(weights, pos, inputs);
      for(int j = 0; j < inputs; ++j) {
        Vec w = mTranspose.row(j);
        w.set(i, v.get(j));
      }
      pos += inputs;
    }

    for(int i = 0; i < mTranspose.rows(); ++i) {
      System.out.println("Row: " + i + " has: " + mTranspose.row(i));
    }

    for(int i = 0; i < inputs; ++i) {
      Vec v = mTranspose.row(i);
      System.out.println(v.toString() + " times " + blame);
      double newEntry = v.dotProduct(blame);
      System.out.println(newEntry);
      prevBlame.set(i, newEntry);
    }
    System.out.println("Blame on LINEAR layer: " + blame.toString());

  }

  void updateGradient(Vec x, Vec gradient) {
    System.out.println("LIN updateGradient");
    System.out.println("input: " + x.toString());
    System.out.println("blame: " + blame.toString());
    System.out.println("gradient: " + gradient.toString());

    // Remove b
    Vec b = new Vec(gradient, 0, outputs);
    System.out.println("b: " + b.toString());

    // add the blame to our bias
    b.add(blame);


    int pos = outputs;
    for(int i = 0; i < x.size(); ++i) { // Review outer_product for help
      double x_i = x.get(i);
      Vec temp = new Vec(gradient, pos, outputs);
      // System.out.println("grad: " + gradient.size());
      // System.out.println("x: " + x.size());
      // System.out.println("temp: " + temp.size());
      // System.out.println("blame: " + blame.size());
      // System.out.println("--------");
      // System.out.println(x_i + " * " + blame);
      temp.addScaled(x_i, blame);
      //double newEntry = x.dotProduct(blame);
      //gradient.addEntry(pos, newEntry);
      pos += outputs;
    }

    System.out.println("Computed weights gradient: " + gradient.toString());
    System.out.println("Computed bias grad: " + b.toString());
    // int mIndex = 0;
    // for(int i = 0; i < blameCrossX.rows(); ++i) {
    //   for(int j = 0; j < blameCrossX.cols(); ++j) {
    //     m.set(mIndex, m.get(mIndex) + blameCrossX.row(i).get(j));
    //   }
    // }

  }


  void ordinary_least_squares(Matrix x, Matrix y, Vec weights) {
    /// x are features
    /// y are labels

    Matrix xCentroid = new Matrix();
    Matrix yCentroid = new Matrix();
    xCentroid.newColumns(x.cols());
    yCentroid.newColumns(y.cols());

    double[] xRow = new double[x.cols()];
    for(int i = 0; i < x.cols(); ++i) {
      double xMean = x.columnMean(i);
      xRow[i] = xMean;
      for(int j = 0; j < x.rows(); ++j) {
        double value = x.row(j).get(i) - xMean;
        x.row(j).set(i, value);
      }
    }
    xCentroid.takeRow(xRow);

    double[] yRow = new double[y.cols()];
    for(int i = 0; i < y.cols(); ++i) {
      double yMean = y.columnMean(i);
      yRow[i] = yMean;
      for(int j = 0; j < y.rows(); ++j) {
        double value = y.row(j).get(i) - yMean;
        y.row(j).set(i, value);
      }
    }
    yCentroid.takeRow(yRow);

    Matrix featuresCrossLabels = Matrix.multiply(y.transpose(), x, false, false); // heeeelp

    Matrix xTranspose = new Matrix(x.transpose());
    Matrix featuresCrossFeatures = Matrix.multiply(xTranspose, x, false, false);
    Matrix fcfInverse = featuresCrossFeatures.pseudoInverse();
    Matrix weightsMatrix = Matrix.multiply(featuresCrossLabels, fcfInverse, false, false);

    //
    // Calculate bias
    Matrix mx = Matrix.multiply(weightsMatrix, xCentroid.transpose(), false, false);
    yCentroid.addScaled(mx, -1);

    //
    // Push the bias Matrix (yCentroid) and weightsMatrix into one long vector
    int weightsIndex = 0;
    for(int i = 0; i < yCentroid.rows(); ++i) {
      Vec temp = yCentroid.row(i);
      for(int j = 0; j < yCentroid.cols(); ++j) {
        weights.set(weightsIndex, temp.get(j));
        ++weightsIndex;
      }
    }

    for(int i = 0; i < weightsMatrix.rows(); ++i) {
      Vec temp = weightsMatrix.row(i);
      for(int j = 0; j < weightsMatrix.cols(); ++j) {
        weights.set(weightsIndex, temp.get(j));
        ++weightsIndex;
      }
    }
  }

}
