public class HiddenNeuron extends Neuron{

    public HiddenNeuron(double[] w, int limit){
        super(w);
        this.oldWeights = new double[limit][w.length];
    }

    public double calcDelta(double outputDelta, double weightToOutput) {
        return outputDelta*weightToOutput*this.activationFunctionDerivative();
    }

}
