package com.tcd.ie.weka.ProjectWeka;

import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class SVMWeka {
	
	public static String DATASET_FILE = "src/main/java/resources/svm.arff";
	public static int DATASET_SIZE = 9366;
	public static int DATASET_ATTRIBUTES_NUM = 2;

	private Instances dataset;
	private LinearRegression model;
	private Normalize normalizer;

	public SVMWeka() throws Exception {
	        //Instances dataset = loadDataset();
	        LinearRegression lr = new LinearRegression();
	        lr.setRidge(1.0E-8);

	        // normalize data
	        Normalize normalizer = new Normalize();
	        normalizer.setInputFormat(dataset);
	        dataset = Filter.useFilter(dataset, normalizer);

	        lr.buildClassifier(dataset);

	        this.dataset = dataset;
	        this.model = lr;
	        this.normalizer = normalizer;
	    }

	private Instances createEmptyDataset() throws Exception {
		ArrayList<Attribute> header = this.createHeader();
        Instances instances = new Instances(DATASET_FILE, header, DATASET_SIZE);
        instances.setClassIndex(DATASET_ATTRIBUTES_NUM - 1);
		return dataset;
	}

	private ArrayList<Attribute> createHeader() {
		 ArrayList<Attribute> header = new ArrayList<Attribute>();
	        header.add(new Attribute("Rating"));
	        header.add(new Attribute("Reviews"));
		return header;
	}
	
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource(DATASET_FILE);
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		/**
		 * svm regression model
		 */
		SMO smo = new SMO();
		smo.buildClassifier(dataset);
		Evaluation svmregeval = new Evaluation(dataset);
		svmregeval.evaluateModel(smo, dataset);
		System.out.println(svmregeval.toSummaryString());
	}
}
