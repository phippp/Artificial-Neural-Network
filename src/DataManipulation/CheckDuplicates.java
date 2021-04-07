package DataManipulation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CheckDuplicates {

    public static void main(String[] args) {
        String COMMA_DELIMITER = ",";
        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("data_csv.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(COMMA_DELIMITER);
                records.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0; i < records.size(); i++){
            for(int j = 0; j < records.get(i).size(); j++){
                if (records.get(i).get(j).equals("")){
                    System.out.println("row: " + i + " column: " + j + " has no value");
                }
            }
        }
    }
}
