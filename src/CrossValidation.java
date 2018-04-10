import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CrossValidation {
    
    //Number of folds in the cross-validation.
    //Set folds to 1 to train and validate on the entire training dataset.
    static int folds = 5;
    
    /**
     * This method runs cross-validation on N shuffles of the training data and reports the average validation error.
     * @param originalData
     */
    public static void crossValidate(ArrayList<ArrayList<String>> originalData) {
        List<Double> averageCVacc = new ArrayList<>();
        int N = 10;
        
        for(int i = 0; i < N; i++) {
            averageCVacc.add(accessory_crossValidate(originalData));
        }
        
        double sum = 0;
        for(Double d : averageCVacc)
            sum += d;
        
        System.out.println("Validation average accuracy with " + folds + " folds = " + sum/averageCVacc.size());
    }
    
    
    /**
     * This method shuffles the given data, splits it into validation and training sets.
     * It then trains a decision tree on each training slice and tests on the validation slice.
     * Each finds the average error on the validation slices for each max_depth of the decision tree.
     * @param originalData
     * @return average validation error for a shuffle of the data
     */
    private static double accessory_crossValidate(ArrayList<ArrayList<String>> originalData) {
        List<Double> accuracy = new ArrayList<>();
        ArrayList<ArrayList<String>> shuffled = shuffleData(originalData);
        
        for(int i = 0; i < folds; i++) {
            ArrayList<ArrayList<ArrayList<String>>> datasets = sliceData(shuffled, i);
            ArrayList<ArrayList<String>> data = datasets.get(0);
            
            //Test on this validation set.
            ArrayList<ArrayList<String>> validation = datasets.get(1);
            
            //System.out.println(data.size());
            //System.out.println(validation.size());
            
            TreeNode root = new TreeNode();
            decisionTree.trainTree(data, root, 0);
            accuracy.add(decisionTree.predict(root, validation));
            
            //System.out.println("---------------------------------");
        }
        double average = 0;
        for(Double d : accuracy)
            average += d;
        return average/accuracy.size();
    }
    
    /**
     * Given a dataset, it shuffles the order of the records.
     * @param originalData
     * @return
     */
    private static ArrayList<ArrayList<String>> shuffleData(ArrayList<ArrayList<String>> originalData) {
        ArrayList<ArrayList<String>> data = new ArrayList<>();
        for(ArrayList<String> row : originalData) {
            
            ArrayList<String> temp = new ArrayList<>();
            for(String s : row)
                temp.add(new String(s));
            
            data.add(temp);
        }
        Collections.shuffle(data);
        return data;
    }
    
    
    /**
     * This method slices the original dataset for crossvalidation and returns the validation and training slices.
     * Return 2 datasets --> 1) Training dataset(toReturn.get(0)) , 2) Validation dataset(toReturn.get(1))
     * @param originalData
     * @param k Number of folds
     * @return 2 datasets - validation and training
     */
    private static ArrayList<ArrayList<ArrayList<String>>> sliceData(ArrayList<ArrayList<String>> originalData, int k) {
        int start = (originalData.size()/folds) * k;
        int end = (k+1) * (originalData.size()/folds) - 1;
        
        ArrayList<ArrayList<String>> train = new ArrayList<>();
        ArrayList<ArrayList<String>> validation = new ArrayList<>();
        
        for(int j = 0; j < start; j++) {
            train.add(originalData.get(j));
        }
        
        for(int j = start; j < end; j++) {
            validation.add(originalData.get(j));
        }
        
        for(int j = end; j < originalData.size(); j++) {
            train.add(originalData.get(j));
        }
        
        ArrayList<ArrayList<ArrayList<String>>> toReturn = new ArrayList<>();
        toReturn.add(train);
        toReturn.add(validation);
        return toReturn;
    }

}
