package com.raylew.ml;

import com.raylew.ml.test.ID3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

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

    public ID3Algorithm(){
        //load data from csv into matrix
        matrix=ID3Util.loadFromFile("watermelon_v2.txt");
        attributes=new ArrayList<>(ATTRIBUTE_COUNT);
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
        ID3Util.selectAttribute(sampleIndexes);
    }

    public static void main(String[] args) {
        ID3Algorithm id3Algorithm=new ID3Algorithm();
        id3Algorithm.getDecisionTree();
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
         * select the best attribute in the samples
         * @param sampleIndexes
         */
        public static void selectAttribute(Set<Integer> sampleIndexes){
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
                        selectAttribute(subSamples);
                    }
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
            ClassLoader classLoader = ID3.class.getClassLoader();
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

    public Tree(T rootData) {
        root = new Node<T>();
        root.data = rootData;
        root.children = new ArrayList<Node<T>>();
    }

    public static class Node<T> {
        private T data;
        private String attrName;
        private String attrValue;
        private Node<T> parent;
        private List<Node<T>> children;

        public Node(){
        }

        public Node(T node){
        }

        public T getData() {
            return data;
        }

        public void setData(T data) {
            this.data = data;
        }

        public String getAttrName() {
            return attrName;
        }

        public void setAttrName(String attrName) {
            this.attrName = attrName;
        }

        public String getAttrValue() {
            return attrValue;
        }

        public void setAttrValue(String attrValue) {
            this.attrValue = attrValue;
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
    }
}
