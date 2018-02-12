import java.util.Arrays;


public class LayerLinear extends Layer {

  LayerLinear(int inputs, int outputs) {
    super(inputs, outputs);
  }

  void activate(Vec weights, Vec x) {
    int totalEntries = weights.size();
    int computedEntries = 0;
    int i = 0;

    double[] data = new double[outputs];
    Vec b = new Vec(weights, 0, outputs);
    computedEntries = computedEntries + outputs;

    while(computedEntries < totalEntries && i < outputs) {
      Vec temp = new Vec(weights, computedEntries, inputs);
      double newEntry = x.dotProduct(temp);
      activation.set(i, newEntry);
      computedEntries = computedEntries + inputs;
      ++i;
    }
    activation.add(b);

  }

  void backProp(Vec weights, Vec prevBlame) {
    // Remove b from the weights
    int totalEntries = weights.size();
    int computedEntries = outputs;
    int i = 0;

    while(computedEntries < totalEntries && i < outputs) {
      Vec temp = new Vec(weights, computedEntries, inputs);
      System.out.println(temp.size());
      System.out.println(prevBlame.size());
      System.out.println(blame.size());
      double newEntry = blame.dotProduct(temp);
      prevBlame.set(i, newEntry);
      computedEntries = computedEntries + inputs;
      ++i;
    }

    // int mSize = (outputs * inputs);
    // Vec m = new Vec(weights, outputs, mSize);
    //
    // int i = 0;
    // int processedValues = 0;
    // while(processedValues < mSize) {
    //   Vec temp = new Vec(weights, i, inputs);
    //   System.out.println(temp.toString());
    //   prevBlame.set(i, temp.dotProduct(blame));
    //   ++i;
    //   processedValues = processedValues + (i * inputs);
    // }
    // System.out.println(prevBlame.toString());

  }

  void updateGradient(Vec x, Vec gradient) {
    // Remove b
    int bSize = outputs;
    Vec b = new Vec(gradient, 0, outputs);

    // Remove m
    int mSize = (outputs * inputs);
    Vec m = new Vec(gradient, outputs, mSize);

    // add the blame to our bias
    b.add(blame);

    Matrix blameCrossX = Matrix.outer_product(blame, x);
    int mIndex = 0;
    for(int i = 0; i < blameCrossX.rows(); ++i) {
      for(int j = 0; j < blameCrossX.cols(); ++j) {
        m.set(mIndex, m.get(mIndex) + blameCrossX.row(i).get(j));
      }
    }

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
