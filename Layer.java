abstract class Layer
{
	protected Vec activation;
	protected Vec blame;
	protected int inputs, outputs;

	Layer(int inputs, int outputs)
	{
		activation = new Vec(outputs);
		blame = new Vec(outputs);
		this.inputs = inputs;
		this.outputs = outputs;
	}

	abstract void activate(Vec weights, Vec x);

	abstract void backprop(Vec weights, Vec prevBlame);

	abstract void updateGradient(Vec x, Vec gradient);
}
