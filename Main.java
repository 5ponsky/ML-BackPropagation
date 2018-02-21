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

	// public static void buildNet() {
	// 	NeuralNet nn = new NeuralNet();
	// 	nn.layers.add(new LayerLinear(784, 80));
	// 	nn.layers.add(new LayerTanh(80));
	//
	// 	nn.layers.add(new LayerLinear(80, 30));
	// 	nn.layers.add(new LayerTanh(30));
	//
	// 	nn.layers.add(new LayerLinear(30, 10));
	// 	nn.layers.add(new LayerTanh(10));
	//
	// 	nn.initWeights();
	// 	Vec o = nn.predict();
	// 	System.out.println("prediction: " + o.toString());
	//
	// 	double[] t = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
	// 	Vec target = new Vec(t);
	// 	nn.backProp(null, target);
	// 	//System.out.println("L1 blame size: " + o.toString());
	// }

	public static void testNet() {
		NeuralNet nn = new NeuralNet();
		nn.layers.add(new LayerLinear(1, 2));
		nn.layers.add(new LayerTanh(2));
		nn.layers.add(new LayerLinear(2, 1));

		double[] w = {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3};
		nn.weights = new Vec(w);

		nn.gradient = new Vec(nn.weights.size());
		nn.gradient.fill(0.0);

		double[] x = {0.3};
		Vec in = new Vec(x);

		double[] t = {0.7};
		Vec target = new Vec(t);
		for(int i = 0; i < 3; ++i) {
			System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
			System.out.println("EPOCH: " + i);
			Vec o = nn.predict(in);
			nn.refineWeights(in, target, null, 0.1);
		}

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
		trainFeatures.scale((1 / 256.0));
		testFeatures.scale((1 / 256.0));

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
		while(mis > 350) {
			System.out.println("==============================");
			System.out.println("TRAINING EPOCH #" + epoch + '\n');

			int break_for_testing = 0;
			for(int i = 0; i < trainFeatures.rows(); ++i) {

				// Train the network on a single input
				Vec in = new Vec(trainFeatures.row(trainingIndices[i]));
				Vec x = nn.predict(in);

				double label = trainLabels.row(trainingIndices[i]).get(0);
				Vec target = new Vec(nn.formatLabel((int)label));
				//System.out.println(target.toString());
				nn.refineWeights(in, target, null, 0.003);

				// Take a break every to often to test
				if(break_for_testing > 5000) {
					mis = nn.countMisclassifications(testFeatures, testLabels);
					System.out.println("Misclassifications: " + mis);
					break_for_testing = 0;

					if(mis < 350) // if misclassifications drop below 350 we're done
						break;
				}

				++break_for_testing;
			}

			/// Shuffle training and testing indices
			for(int i = 0; i < trainingIndices.length; ++i) {
				int randomIndex = random.nextInt(trainingIndices.length);
				int temp = trainingIndices[i];
				trainingIndices[i] = trainingIndices[randomIndex];
				trainingIndices[randomIndex] = temp;

			}

			for(int i = 0; i < testIndices.length; ++i) {
				int randomIndex = random.nextInt(testIndices.length);
				int temp = testIndices[i];
				testIndices[i] = testIndices[randomIndex];
				testIndices[randomIndex] = temp;
			}

			++epoch;
		}

		/// Test on the fully trained net
		// System.out.println("now testing on fully trained set")
		// for(int i = 0; i < testFeatures.rows(); ++i) {
		//
		// }
	}

	public static void testChunks() {
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
		trainFeatures.scale((1 / 256.0));
		testFeatures.scale((1 / 256.0));

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
		System.out.println("==============================");
		System.out.println("TRAINING EPOCH #" + epoch + '\n');

		int break_for_testing = 0;
		for(int i = 0; i < trainFeatures.rows(); ++i) {

			// Train the network on a single input
			Vec in = new Vec(trainFeatures.row(trainingIndices[i]));
			double label = trainLabels.row(trainingIndices[i]).get(0);
			Vec target = new Vec(nn.formatLabel((int)label));
			//System.out.println(target.toString());

			/// This method dives the training
			nn.refineWeights(in, target, null, 0.003);

			// Take a break every to often to test

		}

		mis = nn.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications: " + mis);
	}


	public static void main(String[] args)
	{
		testChunks();
		//run(new NeuralNet());
		//opticalCharacterRecognition();
		//testNet();

	}
}
