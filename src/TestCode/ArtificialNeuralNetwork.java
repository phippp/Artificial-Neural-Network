package TestCode;

public class ArtificialNeuralNetwork {

    static double[][] data = {
            {1,4,-1},
            {2,9,1},
            {5,6,1},
            {4,5,1},
            {6,0.7,-1},
            {1,1.5,-1}
    };

    static double[] weights = {0,0,0};

    public static void main(String[] args) {
        boolean consecutive = false;
        while(!consecutive) {
            consecutive = true;
            for (double[] datum : data) {
                double sum = weights[0] + (weights[1] * datum[0]) + (weights[2] * datum[1]);
                double assigned = (sum > 0)? Double.valueOf(1): Double.valueOf(-1);
                if(assigned != datum[2]) {
                    consecutive = false;
                    weights[0] += (datum[2] * 1);
                    weights[1] += (datum[2] * datum[0]);
                    weights[2] += (datum[2] * datum[1]);
                }
            }
        }
        System.out.println("Final Outputs!\nBias: " + weights[0] + "\nWeight 1: " + weights[1] + "\nWeight 2: " + weights[2]);
    }
}
