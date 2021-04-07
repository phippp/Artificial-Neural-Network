import java.util.Arrays;

public class MLP {

    protected double[] errorHistory = new double[25];
    protected HiddenNeuron[] nodes;
    protected OutputNeuron output;
    protected double learning = 0.1;
    protected final double MOMENTUM = 0.9;

    public MLP(int h, int predictorsLength,int limit){
        nodes = new HiddenNeuron[h];
        for(int i = 0; i < h ; i++){
            nodes[i] = new HiddenNeuron(createWeights(predictorsLength),limit);
        }
        output = new OutputNeuron(createWeights(this.nodes.length),limit);
    }

    public double[] createWeights(int n){
        //takes number of inputs to the node and then assigns weights
        double min = -2/ (double) n;
        double max = 2/ (double) n;

        double[] weights = new double[n+1];
        for(int k = 0; k <= n; k++){
            weights[k] = ((Math.random() * (max - min)) + min);
        }
        return weights;
    }

    //------------------------------------------BATCH---------------------------------------------//

    public double calcBatchError(double[][] predictors, double[] predictand, boolean logging){
        //calculates the MSE across the whole data set supplied
        double count = predictors.length;
        double error = 0;
        for(int i = 0; i < predictors.length; i++) {
            forwardPass(predictors[i]); //sets inputs
            output.setExpectedOut(predictand[i]);
            if(logging) {
                System.out.println(predictand[i] + " ,was:" + this.output.activationFunction());
            }
            error += Math.pow((this.output.activationFunction() - predictand[i]),2);
        }
        return error/count;
    }

    public double batchTrain(double[][] predictors, double[] predictand, boolean momentum){
        //trains on the whole data set
        double[][] adjustments = new double[this.nodes.length][this.nodes[0].getWeights().length];
        double[] outAdjustments = new double[this.output.getWeights().length];
        //visit locations
        for(int i = 0; i < predictors.length; i++){
            double[] input = new double[this.nodes.length+1];
            input[0] = 1;
            //calculates output from the hidden neurons
            for(int node = 0; node < this.nodes.length; node++){
                this.nodes[node].setInputs(predictors[i]);
                input[node+1] = this.nodes[node].activationFunction();
            }
            //supplies the expected result and outputs from hidden neurons
            this.output.setInputs(input);
            this.output.setExpectedOut(predictand[i]);
            //increases the weight adjustments for output neuron (doesnt change weights)
            for(int weight = 0; weight < outAdjustments.length; weight++){
                outAdjustments[weight] += this.learning * this.output.calcDelta() * this.output.getInputs()[weight];// + (MOMENTUM * this.output.getWeightDiff()[weight]);
            }
            //increases weight adjustments for hidden neurons (doesnt change weights)
            for(int nodeNum = 0; nodeNum < this.nodes.length; nodeNum++){
                for(int weight = 0; weight < adjustments[0].length; weight++){
                    adjustments[nodeNum][weight] += this.learning * this.nodes[nodeNum].calcDelta(this.output.calcDelta(), this.output.getInputWeight(nodeNum + 1)) * predictors[i][weight];// + (MOMENTUM * this.nodes[nodeNum].getWeightDiff()[weight]);
                }
            }
        }
        //update adjustments to account for momentum and also to take average
        for (int weight = 0; weight < outAdjustments.length; weight++) {
            if(momentum) {
                outAdjustments[weight] = (outAdjustments[weight] / (double) predictors.length ) + (MOMENTUM * this.output.getWeightDiff()[weight]);
            }else {
                outAdjustments[weight] = (outAdjustments[weight] / (double) predictors.length );
            }
        }
        for (int nodeNum = 0; nodeNum < this.nodes.length; nodeNum++) {
            for (int weight = 0; weight < adjustments[0].length; weight++) {
                if(momentum){
                    adjustments[nodeNum][weight] = (adjustments[nodeNum][weight] / (double) predictors.length) + (MOMENTUM * this.nodes[nodeNum].getWeightDiff()[weight]);
                }else{
                    adjustments[nodeNum][weight] = (adjustments[nodeNum][weight] / (double) predictors.length);
                }
            }
        }
        //actually update weights
        this.output.addToWeights(outAdjustments);
        for (int nodeNum = 0; nodeNum < this.nodes.length; nodeNum++) {
            this.nodes[nodeNum].addToWeights(adjustments[nodeNum]);
        }
        //return the new error after adjustment
        double error = calcBatchError(predictors,predictand,false);
        this.newError(error);
        return error;
    }

    //--------------------------------------------------------------------------------------------//

    public double train(double[][] predictors, double[] predictand, boolean momentum, int epoch, boolean decay){
        //visit locations
        for(int i = 0; i < predictors.length; i++){
            double[] input = new double[this.nodes.length+1];
            input[0] = 1;
            //calculates output from the hidden neurons
            for(int node = 0; node < this.nodes.length; node++){
                this.nodes[node].setInputs(predictors[i]);
                input[node+1] = this.nodes[node].activationFunction();
            }
            //supplies the expected result and outputs from hidden neurons
            this.output.setInputs(input);
            this.output.setExpectedOut(predictand[i]);
            //update weights for the output neuron
            double[] alterations = new double[this.output.getWeights().length];
            for(int weight = 0; weight < alterations.length; weight++){
                if(momentum) {
                    alterations[weight] = this.learning * this.output.calcDelta() * this.output.getInputs()[weight] + (MOMENTUM * this.output.getWeightDiff()[weight]);
                } else {
                    if(decay) {
                        alterations[weight] = this.learning * (this.output.calcDelta() + decay(epoch)) * this.output.getInputs()[weight];
                    } else {
                        alterations[weight] = this.learning * this.output.calcDelta() * this.output.getInputs()[weight];
                    }
                }
            }
            this.output.addToWeights(alterations);
            //updates weights to hidden neurons
            for(int nodeNum = 0; nodeNum < this.nodes.length; nodeNum++){
                alterations = new double[this.nodes[nodeNum].getWeights().length];
                for(int weight = 0; weight < alterations.length; weight++){
                    if(momentum){
                        alterations[weight] = this.learning * this.nodes[nodeNum].calcDelta(this.output.calcDelta(), this.output.getInputWeight(nodeNum + 1)) * predictors[i][weight] + (MOMENTUM * this.nodes[nodeNum].getWeightDiff()[weight]);
                    } else {
                        if(decay){
                            alterations[weight] = this.learning * this.nodes[nodeNum].calcDelta((this.output.calcDelta() + decay(epoch)), this.output.getInputWeight(nodeNum + 1)) * predictors[i][weight];
                        } else {
                            alterations[weight] = this.learning * this.nodes[nodeNum].calcDelta(this.output.calcDelta(), this.output.getInputWeight(nodeNum + 1)) * predictors[i][weight];
                        }
                    }
                }
                this.nodes[nodeNum].addToWeights(alterations);
            }
        }
        //return error after adjustments
        double error = calcBatchError(predictors,predictand,false);
        this.newError(error);
        return error;
    }

    public void forwardPass(double[] predictors){
        double[] input = new double[this.nodes.length+1];
        input[0] = 1;
        for(int i = 0; i < nodes.length; i++){
            this.nodes[i].setInputs(predictors);
            input[i+1] = this.nodes[i].activationFunction();
        }
        this.output.setInputs(input);
    }

    public void newError(double e){
        if (this.errorHistory.length - 1 >= 0) {
            System.arraycopy(this.errorHistory, 1, this.errorHistory, 0, this.errorHistory.length - 1);
        }
        this.errorHistory[this.errorHistory.length-1] = e;
    }

    public String[] getWeights(){
        String[] weights = new String[this.nodes.length+1];
        for(int i = 0; i < this.nodes.length; i++){
            weights[i] = Arrays.toString(this.nodes[i].getWeights());
        }
        weights[weights.length-1] = Arrays.toString(this.output.getWeights());
        return weights;
    }

    public void setWeights(String[] weights){
        //this is for test purposes to allow the user to supply weights
        for(int i = 0 ; i < this.nodes.length; i++){
            String[] parts = weights[i].replace("[","").replace("]", "").split(", ");
            double[] weight = new double[parts.length];
            for(int j = 0 ; j < weight.length; j++){
                weight[j] = Double.parseDouble(parts[j]);
            }
            this.nodes[i].setWeights(weight);
        }
        String[] parts = weights[weights.length-1].replace("[","").replace("]", "").split(", ");
        double[] weight = new double[parts.length];
        for(int j = 0 ; j < weight.length; j++){
            weight[j] = Double.parseDouble(parts[j]);
        }
        this.output.setWeights(weight);
    }

    public void anneal(int epoch){
        double q = 0.01;
        double p = 0.1;
        int max = 2000;
        if(epoch<max){
            this.learning = p + (q - p) * (1 - (1/ (1+ Math.exp(10- ( (20*epoch) / (float)max ) ) ) ) );
        }
    }

    public double decay(int epoch){
        double omega = 0.0;
        for(HiddenNeuron neuron : this.nodes){
            for(double n: neuron.getWeights()){
                omega += Math.pow(n,2);
            }
        }
        for(double n: this.output.getWeights()){
            omega += Math.pow(n,2);
        }
        double epsilon = 1.0 / (this.learning * (double) epoch);
        return (0.5 * omega) * epsilon;
    }
}
