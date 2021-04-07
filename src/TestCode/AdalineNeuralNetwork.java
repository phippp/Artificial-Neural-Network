package TestCode;

public class AdalineNeuralNetwork {

    static double[][] data = {
            {1,4,-1},
            {2,9,1},
            {5,6,1},
            {4,5,1},
            {6,0.7,-1},
            {1,1.5,-1}
    };

    static double[] weights = {0.3,0.3,0.7};

    static double p = 0.01;

    public static void main(String[] args) {
        int limit = 500;
        while(limit > 0) {
            for (double[] datum : data) {
                double sum = weights[0] + (weights[1] * datum[0]) + (weights[2] * datum[1]);
                weights[0] += (p * (datum[2] - sum) * Double.valueOf(1));
                weights[1] += (p * (datum[2] - sum) * datum[0]);
                weights[2] += (p * (datum[2] - sum) * datum[1]);
            }
            limit--;
        }
        System.out.println("Final Outputs!\nBias: " + weights[0] + "\nWeight 1: " + weights[1] + "\nWeight 2: " + weights[2]);
    }
}
