package com.raylew.ml.id3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import static com.raylew.ml.id3.ID3Algorithm.stringBuilder;

/**
 * Created by Raymond on 2017/3/14.
 * 1.calculate total entropy of the set
 * 2.calculate gain entropy of every attribute and choose the max one
 * 3.divide the set into sub-sets by the attribute's different value
 * 4.go back to 1 till the set is empty
 */
public class ID3Algorithm {
    public final static int SAMPLE_COUNT=17;
    public final static int ATTRIBUTE_COUNT=6;
    public static String[][] matrix;
    public static List<Attribute> attributes;
    public static Tree decisionTree;
    public static StringBuilder stringBuilder;

    public ID3Algorithm(){
        stringBuilder=new StringBuilder();
        //load data from csv into matrix
        matrix=ID3Util.loadFromFile("watermelon_v2.txt");
        attributes=new ArrayList<>(ATTRIBUTE_COUNT);
        Attribute rootData=new Attribute();
        decisionTree=new Tree<Attribute>(rootData);
        //initialize attributes
        for(int col=1;col<=ATTRIBUTE_COUNT;col++){
            Attribute attribute=new Attribute();
            attribute.setIndex(col);
            attribute.setName(matrix[0][col]);
            Set<String> values=new HashSet<>();
            for(int row=1;row<matrix.length;row++){
                values.add(matrix[row][col]);
            }
            attribute.setValues(values);
            attributes.add(attribute);
        }
    }

    /**
     * get decision tree
     */
    public void getDecisionTree(){
        Set<Integer> sampleIndexes=new HashSet<Integer>();
        for(int i=1;i<matrix.length;i++){
            int sampleIndex=Integer.parseInt(matrix[i][0]);
            sampleIndexes.add(sampleIndex);
        }
        ID3Util.selectAttribute(sampleIndexes,decisionTree.getRoot(),null);
    }

    public static void main(String[] args) {
        ID3Algorithm id3Algorithm=new ID3Algorithm();
        id3Algorithm.getDecisionTree();
        //decisionTree.traversal(decisionTree.getRoot());
        decisionTree.traversalJson((Tree.Node)decisionTree.getRoot().getChildren().get(0));
        String json=stringBuilder.toString();
        System.out.println(json);
    }

    static class ID3Util{
        public static double getTotalEntropy(Set<Integer> sampleIndexes){
            Map<String,Integer> numbersMap=new HashMap<String, Integer>();
            Iterator<Integer> sampleIterator = sampleIndexes.iterator();
            while(sampleIterator.hasNext()){
                int index=sampleIterator.next();
                if("是".equals(matrix[index][ATTRIBUTE_COUNT+1])){
                    if(numbersMap.containsKey("是")){
                        int numbers=numbersMap.get("是");
                        numbers++;
                        numbersMap.put("是",numbers);
                    }else{
                        numbersMap.put("是",1);
                    }
                }else if("否".equals(matrix[index][ATTRIBUTE_COUNT+1])){
                    if(numbersMap.containsKey("否")){
                        int numbers=numbersMap.get("否");
                        numbers++;
                        numbersMap.put("否",numbers);
                    }else{
                        numbersMap.put("否",1);
                    }
                }
            }
            double ent=getEntropy(numbersMap);
            return ent;
        }

        /**
         * get entropy of map:like {'yes':3,'or':4}
         * @param numbersMap
         * @return
         */
        public static double getEntropy(Map<String,Integer> numbersMap){
            double ent=0;
            Collection<Integer> numbers = numbersMap.values();
            Iterator<Integer> iterator = numbers.iterator();
            int total=0;
            while (iterator.hasNext()){
                Integer next = iterator.next();
                total+=next;
            }
            iterator = numbers.iterator();
            while (iterator.hasNext()){
                Integer next = iterator.next();
                double p=next/(double)total;
                ent+=(-1)*p* (Math.log(p) / Math.log(2));
            }
            return ent;
        }

        /**
         * get gain entropy of an attribute in the samples
         * @param attribute
         * @param sampleIndexes
         * @return
         */
        public static double getGainEntropy(Attribute attribute,Set<Integer> sampleIndexes){
            double totalEntropy=getTotalEntropy(sampleIndexes);
            Set<String> values = attribute.getValues();
            Map<String,List<Integer>> valuesMap=new HashMap<String, List<Integer>>(values.size());
            int attrColIndex=attribute.getIndex();
            Iterator<Integer> sampleIterator = sampleIndexes.iterator();
            while(sampleIterator.hasNext()){
                int index=sampleIterator.next();
                if(valuesMap.containsKey(matrix[index][attrColIndex])){
                    List<Integer> list=valuesMap.get(matrix[index][attrColIndex]);
                    list.add(index);
                }else{
                    List<Integer> list=new LinkedList<>();
                    list.add(index);
                    valuesMap.put(matrix[index][attrColIndex],list);
                }
            }
            double ent=0;
            Iterator<String> valuesIterator = values.iterator();
            while(valuesIterator.hasNext()){
                String value=valuesIterator.next();
                List<Integer> indexes = valuesMap.get(value);
                if(indexes==null||indexes.size()==0){
                    continue;
                }
                Map<String,Integer> numbersMap=new HashMap<String, Integer>();
                for(int i=0;i<indexes.size();i++){
                    int sampleIndex=indexes.get(i);
                    if(numbersMap.containsKey(matrix[sampleIndex][ATTRIBUTE_COUNT+1])){
                        int number=numbersMap.get(matrix[sampleIndex][ATTRIBUTE_COUNT+1]);
                        number++;
                        numbersMap.put(matrix[sampleIndex][ATTRIBUTE_COUNT+1],number);
                    }else{
                        numbersMap.put(matrix[sampleIndex][ATTRIBUTE_COUNT+1],1);
                    }
                }
                ent+=(indexes.size()/(double)(sampleIndexes.size()))*getEntropy(numbersMap);
            }
            return totalEntropy-ent;
        }

        /**
         * get leaf decision result by max attrValue in samples
         * @param sampleIndexes samples indexes
         * @param pAttr parent attribute value
         * @param pAttrValue parent attribute
         * @return
         */
        public static String getLeafResult(Set<Integer> sampleIndexes,Attribute pAttr,String pAttrValue){
            String result=null;
            Map<String,Integer> map=new HashMap<>();
            int pAttrCol=-1;
            for(int col=1;col<=ATTRIBUTE_COUNT;col++){
                if(matrix[0][col].equals(pAttr.getName())){
                    pAttrCol=col;
                    break;
                }
            }
            if(pAttrCol==-1){
                return result;
            }
            Iterator<Integer> sampleIndexesIterator = sampleIndexes.iterator();
            while(sampleIndexesIterator.hasNext()){
                int row=sampleIndexesIterator.next();
                if(matrix[row][pAttrCol].equals(pAttrValue)){
                    String key=matrix[row][ATTRIBUTE_COUNT+1];
                    if(map.containsKey(key)){
                        map.put(key,map.get(key)+1);
                    }else{
                        map.put(key,1);
                    }
                }
            }
            Iterator<String> iterator = map.keySet().iterator();
            int max=Integer.MIN_VALUE;
            while(iterator.hasNext()){
                String key=iterator.next();
                if(map.get(key)>max){
                    max=map.get(key);
                    result=key;
                }
            }
            return result;
        }

        /**
         * get leaf decision result by max attrValue in samples
         * @param sampleIndexes samples indexes
         * @param pAttr parent attribute value
         * @return
         */
        public static String getLeafResult(Set<Integer> sampleIndexes,Attribute pAttr){
            String result=null;
            Map<String,Integer> map=new HashMap<>();
            int pAttrCol=-1;
            for(int col=1;col<=ATTRIBUTE_COUNT;col++){
                if(matrix[0][col].equals(pAttr.getName())){
                    pAttrCol=col;
                    break;
                }
            }
            if(pAttrCol==-1){
                return result;
            }
            Iterator<Integer> sampleIndexesIterator = sampleIndexes.iterator();
            while(sampleIndexesIterator.hasNext()){
                int row=sampleIndexesIterator.next();
                String key=matrix[row][ATTRIBUTE_COUNT+1];
                if(map.containsKey(key)){
                    map.put(key,map.get(key)+1);
                }else{
                    map.put(key,1);
                }
            }
            Iterator<String> iterator = map.keySet().iterator();
            int max=Integer.MIN_VALUE;
            while(iterator.hasNext()){
                String key=iterator.next();
                if(map.get(key)>max){
                    max=map.get(key);
                    result=key;
                }
            }
            return result;
        }

        /**
         * select the best attribute in the samples
         * @param sampleIndexes
         */
        public static void selectAttribute(Set<Integer> sampleIndexes,Tree.Node<Attribute> node,String pAttrValue){
            if(sampleIndexes.size()==0){
                return;
            }else {
                double maxGainEntry = 0;
                Attribute attribute = null;
                for (int i = 0; i < attributes.size(); i++) {
                    double gainEnt = getGainEntropy(attributes.get(i), sampleIndexes);
                    if (gainEnt > maxGainEntry) {
                        maxGainEntry = gainEnt;
                        attribute = attributes.get(i);
                    }
                }
                if(attribute!=null){
                    //add node to tree
                    Tree.Node<Attribute> curNode = node.addChild(attribute,pAttrValue);

                    Set<String> values = attribute.getValues();
                    Iterator<String> iterator = values.iterator();
                    while(iterator.hasNext()){
                        String value=iterator.next();
                        Set<Integer> subSamples=new HashSet<Integer>();
                        Iterator<Integer> sampleIterator = sampleIndexes.iterator();
                        while(sampleIterator.hasNext()){
                            int index=sampleIterator.next();
                            if(matrix[index][attribute.getIndex()].equals(value)){
                                int sampleIndex=Integer.parseInt(matrix[index][0]);
                                subSamples.add(sampleIndex);
                            }
                        }
                        if(subSamples.size()==0){
                            //leaf node 色泽-浅白-好瓜
                            String leafResult=getLeafResult(sampleIndexes,curNode.getData());
                            Attribute attribute1=new Attribute();
                            attribute1.setName(leafResult);
                            curNode.addLeafChild(attribute1,value,leafResult);
                        }else{
                            selectAttribute(subSamples,curNode,value);
                        }
                    }
                }else{
                    //leaf node
                    String leafResult=getLeafResult(sampleIndexes,node.getData(),pAttrValue);
                    Attribute attribute1=new Attribute();
                    attribute1.setName(leafResult);
                    node.addLeafChild(attribute1,pAttrValue,leafResult);
                }
            }
            return;
        }

        /**
         * load data from file into a matrix
         * @param fileName
         * @return
         */
        public static String[][] loadFromFile(String fileName){
            String[][] matrix=new String[SAMPLE_COUNT+1][ATTRIBUTE_COUNT+2];
            ClassLoader classLoader = ID3Algorithm.class.getClassLoader();
            int rowIndex=0;
            try(BufferedReader br = new BufferedReader(new FileReader(classLoader.getResource(fileName).getFile()))) {
                for(String line; (line = br.readLine()) != null; ) {
                    String[] split = line.split(",");
                    matrix[rowIndex++]=split;
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return matrix;
        }

        public static void writeToXml(){

        }
    }
}

class Attribute{
    private String name;
    private Integer index;
    private Set<String> values;

    public Attribute(){

    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getIndex() {
        return index;
    }

    public void setIndex(Integer index) {
        this.index = index;
    }

    public Set<String> getValues() {
        return values;
    }

    public void setValues(Set<String> values) {
        this.values = values;
    }
}

class Tree<T> {
    private Node<T> root;

    public Node<T> getRoot() {
        return root;
    }

    public void setRoot(Node<T> root) {
        this.root = root;
    }

    public Tree(T rootData) {
        root = new Node<T>();
        root.data = rootData;
        root.children = new ArrayList<Node<T>>();
    }

    public static class Node<T> {
        private T data;
        private String pAttrName;
        private String pAttrValue;
        private String leafResult;
        private Node<T> parent;
        private List<Node<T>> children;

        public Node(){
        }

        public Node(T node){
            this.data=node;
            children=new ArrayList<>();
        }

        public Node(T node,String pAttrValue){
            this.data=node;
            this.pAttrValue=pAttrValue;
            children=new ArrayList<>();
        }

        public T getData() {
            return data;
        }

        public void setData(T data) {
            this.data = data;
        }

        public String getpAttrName() {
            return pAttrName;
        }

        public void setpAttrName(String pAttrName) {
            this.pAttrName = pAttrName;
        }

        public String getpAttrValue() {
            return pAttrValue;
        }

        public void setpAttrValue(String pAttrValue) {
            this.pAttrValue = pAttrValue;
        }

        public String getLeafResult() {
            return leafResult;
        }

        public void setLeafResult(String leafResult) {
            this.leafResult = leafResult;
        }

        public Node<T> getParent() {
            return parent;
        }

        public void setParent(Node<T> parent) {
            this.parent = parent;
        }

        public List<Node<T>> getChildren() {
            return children;
        }

        public void setChildren(List<Node<T>> children) {
            this.children = children;
        }

        public Node<T> addChild(T child) {
            Node<T> childNode = new Node<T>(child);
            childNode.parent = this;
            this.children.add(childNode);
            return childNode;
        }

        public Node<T> addChild(T child,String pAttrValue) {
            Node<T> childNode = new Node<T>(child);
            childNode.parent = this;
            childNode.pAttrValue=pAttrValue;
            this.children.add(childNode);
            return childNode;
        }

        public Node<T> addLeafChild(T child,String pAttrValue,String leafResult) {
            Node<T> childNode = new Node<T>(child);
            childNode.parent = this;
            childNode.pAttrValue=pAttrValue;
            childNode.leafResult=leafResult;
            this.children.add(childNode);
            return childNode;
        }
    }

    /**
     * traversal tree and output node
     * @param node
     */
    public void traversal(Node<Attribute> node){
        if(node.getData().getName()!=null){
            System.out.println(node.getData().getName());
        }
        if(node.children==null||node.children.size()==0){
            return;
        }else{
            List children = node.children;
            Iterator iterator = children.iterator();
            while(iterator.hasNext()){
                traversal((Node)iterator.next());
            }
        }
    }

    /**
     * traversal tree and output node
     * @param node
     */
    public void traversalJson(Node<Attribute> node){
        System.out.println(node.getData().getName());
        if(node.children==null||node.children.size()==0){
            String leafResult=node.getLeafResult();
            stringBuilder.append("{");
            stringBuilder.append("\"error\"").append(":").append(0).append(",");
            stringBuilder.append("\"samples\"").append(":").append(0).append(",");
            int result=leafResult.equals("好瓜")?1:0;
            stringBuilder.append("\"value\"").append(":").append("["+result+"]").append(",");
            stringBuilder.append("\"label\"").append(":").append("\"" + result + "\"").append(",");
            stringBuilder.append("\"type\"").append(":").append("\"leaf\"");
            stringBuilder.append("}");
            return;
        }else{
            stringBuilder.append("{");
            stringBuilder.append("\"error\"").append(":").append(0).append(",");
            stringBuilder.append("\"samples\"").append(":").append(0).append(",");
            stringBuilder.append("\"value\"").append(":").append("[]").append(",");
            stringBuilder.append("\"label\"").append(":").append("\"" + node.getData().getName()
                    +"("+node.getpAttrValue()+")"+ "\"").append(",");
            stringBuilder.append("\"feature\"").append(":").append("\"\"").append(",");
            stringBuilder.append("\"type\"").append(":").append("\"split\"").append(",");
            stringBuilder.append("\"children\"").append(":").append("[");
            List children = node.children;
            for(int i=0;i<children.size();i++){
                Node childNode=(Node)children.get(i);
                traversalJson(childNode);
                if(i<children.size()-1){
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append("]");
            stringBuilder.append("}");
        }
    }

}
