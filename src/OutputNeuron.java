public class OutputNeuron extends Neuron{

    protected double expectedOut;

    public OutputNeuron(double[] w, int limit){
        super(w);
        this.oldWeights = new double[limit][w.length];
    }

    public void setExpectedOut(double o){
        this.expectedOut = o;
    }

    public double calcDelta() {
        return (this.expectedOut-this.activationFunction())*this.activationFunctionDerivative();
    }

    public double getInputWeight(int n){
        return this.weights[n];
    }

    public double[] getInputs(){
        return this.inputs;
    }

}
