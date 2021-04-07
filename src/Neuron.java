abstract class Neuron {

    protected double[] inputs;
    protected double[][] oldWeights;
    protected double[] weights;

     Neuron(double[] weights){
        this.weights = weights;
    }

    public double weightedSum() {
        double val = 0.0;
        for(int i = 0; i < this.weights.length; i++){
            val += (this.weights[i] * this.inputs[i]);
        }
        return val;
    }

    public double activationFunctionDerivative() {
        return this.activationFunction() * (1- this.activationFunction());
    }

    public double activationFunction() {
        return (1.0)/(1.0 + Math.exp(-this.weightedSum()));
    }

//    public double activationFunction(){
//         return (Math.exp(this.weightedSum()) - Math.exp(-this.weightedSum()))/(Math.exp(this.weightedSum()) + Math.exp(-this.weightedSum()));
//    }
//
//    public double activationFunctionDerivative() {
//         return 1.0 - Math.pow(this.activationFunction(),2);
//    }

    public double[] getWeights() {
        return this.weights;
    }

    public double[] getWeightDiff(){
        double[] last = this.oldWeights[this.oldWeights.length-1];
        double[] curr = this.getWeights();
        double[] diff = new double[this.getWeights().length];
        if(last[0] != last[last.length-1]) {
            for (int i = 0; i < this.getWeights().length; i++) {
                diff[i] = curr[i] - last[i];
            }
        }
        return diff;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public void addToWeights(double[] w){
        for(int i = 0; i < this.oldWeights.length -1; i++){
            this.oldWeights[i] = this.oldWeights[i+1];
        }
        for(int i = 0; i < this.weights.length; i++){
            this.oldWeights[this.oldWeights.length-1][i] = this.weights[i];
            this.weights[i] += w[i];
        }
    }

    public void setWeights(double[] w){
        this.weights = w;
    }
}
