import java.util.ArrayList;
import java.util.List;

//Decision Tree Node.
class TreeNode {
    List<TreeNode> adj;           //The node's children
    BestAttribute attribute;      //The attribute on which this node splits the data
    String prev_SplitVal;         //Value of the parent node. This property tells the branch (after the split at the parent node) on which this node lies
    boolean isLeaf;               //True if this node is a leaf
    String label;                 //null if (isLeaf == false), else hold the classifying label value
    
    TreeNode() {
        adj = new ArrayList<>();
        attribute = null;
        prev_SplitVal = null;
        isLeaf = false;
        label = null;
    }
}