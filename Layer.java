abstract class Layer
{
	protected Vec activation;
	protected int inputs, outputs;

	Layer(int inputs, int outputs)
	{
		activation = new Vec(outputs);
		this.inputs = inputs;
		this.outputs = outputs;
	}

	abstract void activate(Vec weights, Vec x);
}
