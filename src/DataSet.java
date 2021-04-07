public class DataSet {

    protected double[][] data;

    public DataSet(double[][] data){
        this.data = data;
    }

    public double getPredictand(int i){
        return this.data[i][this.data[i].length-1];
    }

    public double[] getPredictors(int i){
        double[] row = this.data[i];
        double[] temp = new double[this.data[0].length];
        temp[0] = 1;
        System.arraycopy(row,0,temp,1,row.length-1);
        return temp;
    }

    public int getLength(){
        return  this.data.length;
    }

    public int getWidth(){
        return  this.data[0].length;
    }

    public double[] getAllPredictands(){
        double[] temp = new double[this.getLength()];
        for(int i = 0; i < this.getLength(); i++){
            temp[i] = this.getPredictand(i);
        }
        return temp;
    }

    public double[][] getAllPredictors(){
        double[][] temp = new double[this.getLength()][this.getWidth()-1];
        for(int i = 0; i < this.getLength(); i++){
            temp[i] = this.getPredictors(i);
        }
        return temp;
    }

}
