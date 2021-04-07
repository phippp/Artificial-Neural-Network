import java.io.*;
import java.util.*;

public class MLPHandler {

    protected static MLP model;
    protected static DataSet[] training = new DataSet[10];
    protected static DataSet validation;
    protected static DataSet test;
    protected static String[][] weightHistory;
    protected static double[] errorHistory;
    protected static int INPUTS = 8; //number of predictors
    protected static int HIDDEN = 4; //number of hidden nodes
    protected static int LIMIT = 5; //how many times the error increases before stopping
    protected static boolean BATCH = false;
    protected static boolean DECAY = false;
    protected static boolean ANNEAL = false;
    protected static boolean MOMENTUM = false;
    protected static String[] bestWeights = {}; // hard-coded best weights for loading weights for test set

    public static void main(String[] args){
        //getConstants();
        model = new MLP(HIDDEN,INPUTS,LIMIT);
        trainModel();
        bestWeights = weightHistory[0];
        System.out.println("TESTING:");
        testModel();
    }

    private static void getConstants() {
        Scanner scan = new Scanner(System.in);  // Create a Scanner object
        System.out.println("Enter number of predictors:");
        INPUTS = scan.nextInt();
        System.out.println("Enter number of Hidden nodes:");
        HIDDEN = scan.nextInt();
        System.out.println("Enter how many increases before the training ends (recommended is 5):");
        LIMIT = scan.nextInt();
        System.out.println("Enter y for momentum:");
        String t = scan.next();
        MOMENTUM = t.toLowerCase().charAt(0) == 'y';
        System.out.println("Enter y for batch processing:");
        t = scan.next();
        BATCH = t.toLowerCase().charAt(0) == 'y';
        System.out.println("Enter y for annealing:");
        t = scan.next();
        ANNEAL = t.toLowerCase().charAt(0) == 'y';
        System.out.println("Enter y for weight decay:");
        t = scan.next();
        DECAY = t.toLowerCase().charAt(0) == 'y';
    }

    private static void testModel() {
        loadTestSet();
        model.setWeights(bestWeights);
        double error = model.calcBatchError(test.getAllPredictors(),test.getAllPredictands(),true);
        System.out.println("Error :" + error);
    }

    private static void trainModel() {
        weightHistory = new String[LIMIT][INPUTS+1];
        errorHistory = new double[LIMIT];
        loadTrainingSets();
        loadValidationSet();

        //outputs the starting weights to file
        String[] s = model.getWeights();
        String[] str = new String[s.length+1];
        str[0] = "Start Weights:\n";
        for(int i = 0; i < s.length-1; i++){
            str[i+1] = "Hidden Neuron #" + (i+1) + "\n\t" +s[i] + "\n";
        }
        str[str.length-1] = "Output Neuron \n\t" + s[s.length-1] + "\n";

        printToFile(str,false);

        //create variables for training
        int incrementing = 0;
        int epoch = 1;

        //loop until the error on validation set increases
        while(incrementing < LIMIT){
            for(int train = 0; train < training.length; train++){
                double[][] data = training[0].getAllPredictors();
                double[] out = training[0].getAllPredictands();
                if(ANNEAL){
                    model.anneal(epoch);
                }
                if(BATCH) {
                    model.batchTrain(data, out, MOMENTUM);
                } else {
                    model.train(data,out,MOMENTUM,epoch,DECAY);
                }
            }
            //error for ending
            updateHistory((epoch % 100 == 0));
            System.out.println("After epoch #"+epoch+"\n\t"+errorHistory[errorHistory.length-1]);
            if(errorHistory[errorHistory.length-1] > errorHistory[errorHistory.length-2]){ incrementing++; }else{ incrementing = 0; }
            epoch++;
        }
        //final error on validation set
        model.setWeights(weightHistory[0]);
        model.calcBatchError(validation.getAllPredictors(),validation.getAllPredictands(),true);

        //output to file
        s = model.getWeights();
        str = new String[s.length+2];
        str[0] = "Final Weights:\n";
        str[1] = "Error of:" + errorHistory[0] + "\n";
        for(int i = 0; i < s.length-1; i++){
            str[i+2] = "Hidden Neuron #" + (i+1) + "\n\t" +s[i] + "\n";
        }
        str[str.length-1] = "Output Neuron \n\t" + s[s.length-1] + "\n";

        printToFile(str,true);
    }

    private static void updateHistory(boolean log) {
        if (weightHistory.length - 1 >= 0) {
            System.arraycopy(weightHistory, 1, weightHistory, 0, weightHistory.length - 1);
        }
        weightHistory[weightHistory.length-1] = model.getWeights();
        if (errorHistory.length - 1 >= 0) {
            System.arraycopy(errorHistory,1,errorHistory,0,errorHistory.length-1);
        }
        errorHistory[errorHistory.length-1] = model.calcBatchError(validation.getAllPredictors(),validation.getAllPredictands(),log);
    }

    private static void printToFile(String[] str, boolean append) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("mse.txt",append))) {
            for(String s : str){
               writer.write(s);
            }
            writer.flush();
        } catch (IOException e){
            System.out.println("Failed");
            e.printStackTrace();
        }
    }

    private static void loadValidationSet() {
        List<double[]> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("ValidationSet.csv"))) {
            String line = br.readLine();
            while((line = br.readLine()) != null){

                try {
                    records.add(Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray());
                } catch (NumberFormatException e) {
                    e.getStackTrace();
                }//try

            }//while

            validation = new DataSet(records.toArray(new double[0][0]));

        } catch (IOException e) {
            e.printStackTrace();
        }//try
    }

    private static void loadTrainingSets() {
        List<double[]> records = new ArrayList<>();

        //this gets the number of lines that the file has so that we can set the boundaries and split the data up
        FileInputStream stream;
        int count = 0;
        try {
            stream = new FileInputStream("TrainingSet.csv");
            byte[] buffer = new byte[8192];
            int n;
            while ((n = stream.read(buffer)) > 0) {
                for (int i = 0; i < n; i++) {
                    if (buffer[i] == '\n') count++;
                }
            }
            stream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        int[] brackets = {0,(count/10),(count/10)*2,(count/10)*3,(count/10)*4,(count/10)*5,(count/10)*6,(count/10)*7,(count/10)*8,(count/10)*9,count};

        try (BufferedReader br = new BufferedReader(new FileReader("TrainingSet.csv"))) {
            String line = br.readLine();
            for(int n = 0; n <10 ;n++) {
                for (int i = brackets[n]; i < brackets[n+1] && (line = br.readLine()) != null ; i++) {
                    try {
                        records.add(Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray());
                    } catch (NumberFormatException e) {
                        e.getStackTrace();
                    }//try

                }//for

                training[n] = new DataSet(records.toArray(new double[records.size()][0]));
                records.clear();

            }//for

        } catch (IOException e) {
            e.printStackTrace();
        }//try
    }

    private static void loadTestSet() {
        List<double[]> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("TestingSet.csv"))) {
            String line = br.readLine();
            while((line = br.readLine()) != null){

                try {
                    records.add(Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray());
                } catch (NumberFormatException e) {
                    e.getStackTrace();
                }//try

            }//while

            test = new DataSet(records.toArray(new double[0][0]));

        } catch (IOException e) {
            e.printStackTrace();
        }//try
    }
}
