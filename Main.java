// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;

class Main
{
	static void test(SupervisedLearner learner, String challenge) {
		// Load the training data
		String fn = "data/" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		learner.train(trainFeatures, trainLabels);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	public static void run(SupervisedLearner learner) {
		int folds = 10;
		int repititions = 5;

		// Load the training data
		Matrix featureData = new Matrix();
		featureData.loadARFF("data/housing_features.arff");
		Matrix labelData = new Matrix();
		labelData.loadARFF("data/housing_labels.arff");

		double rmse = learner.cross_validation(repititions, folds, featureData, labelData);
		System.out.println("RMSE: " + rmse);
	}

	public static void testCV(SupervisedLearner learner) {
		Matrix f = new Matrix();
		f.newColumns(1);
		double[] f1 = {0};
		double[] f2 = {0};
		double[] f3 = {0};
		f.takeRow(f1);
		f.takeRow(f2);
		f.takeRow(f3);

		Matrix l = new Matrix();
		l.newColumns(1);
		double[] l1 = {2};
		double[] l2 = {4};
		double[] l3 = {6};
		l.takeRow(l1);
		l.takeRow(l2);
		l.takeRow(l3);

		double rmse = learner.cross_validation(1, 3, f, l);
		System.out.println("RMSE: " + rmse);
	}

	public static void testOLS() {
		LayerLinear ll = new LayerLinear(13, 1);
		Random random = new Random(1234);
		Vec weights = new Vec(14);

		for(int i = 0; i < 14; ++i) {
			weights.set(i, random.nextGaussian());
		}

		Matrix x = new Matrix();
		x.newColumns(13);
		for(int i = 0; i < 100; ++i) {
			double[] temp = new double[13];
			for(int j = 0; j < 13; ++j) {
				temp[j] = random.nextGaussian();
			}
			x.takeRow(temp);
		}

		Matrix y = new Matrix(100, 1);
		for(int i = 0; i < y.rows(); ++i) {
			ll.activate(weights, x.row(i));
			for(int j = 0; j < ll.activation.size(); ++j) {
				double temp = ll.activation.get(j) + random.nextGaussian();
				y.row(i).set(j, temp);
			}
		}

		for(int i = 0; i < weights.size(); ++i) {
    	System.out.println(weights.get(i));
		}

		Vec olsWeights = new Vec(14);
		ll.ordinary_least_squares(x,y,olsWeights);

		System.out.println("-----------------------------");

		for(int i = 0; i < olsWeights.size(); ++i) {
			System.out.println(olsWeights.get(i));
		}
	}


	public static void testLayer() {
		double[] x = {0, 1, 2};
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		System.out.println(ll.activation.toString());
	}

	public static void testBackProp() {
		double[] x = {0, 1, 2};
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		double[] yhat = {9, 6};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		ll.blame = new Vec(yhat);
		ll.backProp(new Vec(m), new Vec(x));
		System.out.println(ll.blame.toString());
	}


	public static void testNet() {
		NeuralNet nn = new NeuralNet();
		nn.layers.add(new LayerLinear(1, 2));
		nn.layers.add(new LayerTanh(2));
		nn.layers.add(new LayerLinear(2, 1));

		double[] w = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3};
		nn.weights = new Vec(w);
		//nn.initWeights();

		double[] x = {0.3};
		Vec in = new Vec(x);

		double[] t = {0.7};
		Vec target = new Vec(t);
		for(int i = 0; i < 3; ++i) {
			System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
			System.out.println("EPOCH: " + i);
			//nn.predict(in);
			nn.refineWeights(in, target, null, 0.1);
			System.out.println(nn.weights);
		}

		System.out.println(nn.predict(in));

	}

	public static void opticalCharacterRecognition() {
		Random random = new Random(1234); // used for shuffling data


		/// Load training and testing data
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF("data/train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF("data/train_lab.arff");

		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF("data/test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF("data/test_lab.arff");

		/// Normalize our training/testing data by dividing by 256.0
		/// There are 256 possible values for any given entry
		trainFeatures.scale((1.0 / 256.0));
		testFeatures.scale((1.0 / 256.0));

		/// Build index arrays to shuffle training and testing data
		int[] trainingIndices = new int[trainFeatures.rows()];
		int[] testIndices = new int[testFeatures.rows()];

		// populate the index arrays with indices
		for(int i = 0; i < trainingIndices.length; ++i) { trainingIndices[i] = i; }
		for(int i = 0; i < testIndices.length; ++i) { testIndices[i] = i; }

		/// Assemble and initialize a neural net
		NeuralNet nn = new NeuralNet();

		nn.layers.add(new LayerLinear(784, 80));
		nn.layers.add(new LayerTanh(80));

		nn.layers.add(new LayerLinear(80, 30));
		nn.layers.add(new LayerTanh(30));

		nn.layers.add(new LayerLinear(30, 10));
		nn.layers.add(new LayerTanh(10));

		nn.initWeights();

		/// Training and testing
		int mis = 10000;
		int epoch = 0;

		/// Continue to train until we get less than 350 misclassifications
		while(mis > 350) {
			System.out.println("=====================================");
			System.out.println("TRAINING EPOCH #" + epoch + '\n');

			// Calculate the number of misclassifications
			mis = nn.countMisclassifications(testFeatures, testLabels);
			System.out.println("Misclassifications: " + mis);

			// Train on the entire training dataset
			for(int i = 0; i < trainFeatures.rows(); ++i) {
				//Vec in = trainFeatures.row(trainingIndices[i]);
				//double label = trainLabels.row(trainingIndices[i]).get(0);
				Vec in = trainFeatures.row(i);

				double label = trainLabels.row(i).get(0);
				Vec target = nn.formatLabel((int)label);

				// Refine the weights
				nn.refineWeights(in, target, null, 0.03);
			}

			// Try to predict the first label in the test data
			// This is only here for debugging
			System.out.println("Predicted: " + nn.predict(testFeatures.row(0)));
			double label = testLabels.row(0).get(0);
			Vec target = new Vec(nn.formatLabel((int)label));
			System.out.println("Actual: " + target);

			/// Shuffle training and testing indices
			// for(int i = 0; i < trainingIndices.length; ++i) {
			// 	int randomIndex = random.nextInt(trainingIndices.length);
			// 	int temp = trainingIndices[i];
			// 	trainingIndices[i] = trainingIndices[randomIndex];
			// 	trainingIndices[randomIndex] = temp;
			//
			// }
			//
			// for(int i = 0; i < testIndices.length; ++i) {
			// 	int randomIndex = random.nextInt(testIndices.length);
			// 	int temp = testIndices[i];
			// 	testIndices[i] = testIndices[randomIndex];
			// 	testIndices[randomIndex] = temp;
			// }

			++epoch;
		}
	}


	public static void testNet1() {
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF("data/test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF("data/test_lab.arff");

		NeuralNet nn = new NeuralNet();

		nn.layers.add(new LayerLinear(784, 10));
		nn.initWeights();
		for(int j = 0; j < 10; ++j) { // just trying 10 reps
			for(int i = 0; i < testFeatures.rows(); ++i) {
				Vec label = nn.formatLabel((int)testLabels.row(i).get(0));
				nn.refineWeights(testFeatures.row(i), label, nn.weights, 0.001);
				if(i % 100 == 0)
					System.out.println(nn.weights);
			}
			//System.out.println(nn.weights);
		}
		System.out.println("done");

		Vec w = new Vec(nn.weights.size());
		LayerLinear l = new LayerLinear(784, 10);
		l.ordinary_least_squares(testFeatures, testLabels, w);

		System.out.println(w);

	}

	public static void testNet2() {
		NeuralNet nn = new NeuralNet();
		nn.layers.add(new LayerLinear(784, 80));
		nn.layers.add(new LayerTanh(80));

		nn.layers.add(new LayerLinear(80, 30));
		nn.layers.add(new LayerTanh(30));

		nn.layers.add(new LayerLinear(30, 10));
		nn.layers.add(new LayerTanh(10));

		//double[] w = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3};
		//nn.weights = new Vec(w);
		nn.initWeights();

		double[] x = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		Vec in = new Vec(x);
		in.scale((1.0/256.0));

		double[] t = {0,0,0,0,0,0,0,1,0,0};
		Vec target = new Vec(t);
		for(int i = 0; i < 10; ++i) {
			System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
			System.out.println("EPOCH: " + i);
			//System.out.println(nn.weights);
			Vec o = nn.predict(in);
			System.out.println(o);
			//nn.refineWeights(in, target, null, 0.03);
			nn.backProp(nn.weights, target);
			nn.updateGradient(in);
			nn.weights.addScaled(0.03, nn.gradient);
		}

		System.out.println(nn.predict(in));

	}

	public static void testGrad() {

	}

	public static void main(String[] args)
	{
		//testSomething();
		//testChunks();
		//run(new NeuralNet());
		opticalCharacterRecognition();
		//testNet2();
		//testNet1();
		//testNet();
	}
}
