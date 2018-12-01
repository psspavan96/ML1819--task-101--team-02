package com.tcd.ml.ProjectWeksAssign2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class LinearRegressionWeka2 {

	public static String DATASET_FILE = "src/main/java/resources/linearReg.arff";
	public static int DATASET_SIZE = 4999;
	public static int DATASET_ATTRIBUTES_NUM = 2;

	private Instances dataset;
	private LinearRegression model;
	private Normalize normalizer;

	public LinearRegressionWeka2() throws Exception {
	        Instances dataset = loadDataset();
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
	        header.add(new Attribute("NoOfContacts"));
	        header.add(new Attribute("LastContactDay"));
		return header;
	}

	private Instances loadDataset() throws RuntimeException {
		Instances dataset = null;
		BufferedReader br = null;
		FileReader fr = null;
		try {
			ClassLoader classLoader = getClass().getClassLoader();
			fr = new FileReader(DATASET_FILE);
			br = new BufferedReader(fr);
			String sCurrentLine;
			int line = 1;

			dataset = this.createEmptyDataset();
			while ((sCurrentLine = br.readLine()) != null) {
				if (line > 1) {
					try {
						double[] values = new double[DATASET_ATTRIBUTES_NUM];
						int i = 0;
						for (String val : sCurrentLine.split(";")) {
							values[i] = Double.parseDouble(val);
							i++;
						}
						dataset.add(new DenseInstance(1.0, values));
					} catch (NumberFormatException ex) {
						System.err.println(ex.getMessage());
					}
				}
				line++;
			}
			br.close();
		} catch (final Exception e) {
			throw new RuntimeException(e);
		} finally {
			try {
				if (br != null)
					br.close();
				if (fr != null)
					fr.close();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return dataset;
	}

	public double crossValidate(int numFolds) throws Exception {
		// cross validate
		Evaluation evaluation = new Evaluation(this.dataset);
		evaluation.crossValidateModel(this.model, this.dataset, numFolds, new Random(1));
		return evaluation.rootMeanSquaredError();
	}

	public double predict(double[] data) throws Exception {
		Instances instances = this.createEmptyDataset();
		DenseInstance instance = new DenseInstance(1.0, data);
		instances.add(instance);

		instances = Filter.useFilter(instances, this.normalizer);
		return this.model.classifyInstance(instances.instance(0));
	}

	public Instances getDataset() {
	        return dataset;
	    }
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource(DATASET_FILE);
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		/**
		 * linear regression model
		 */
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		//lr.classifyInstance(dataset.)
		//System.out.println(lr);
		Evaluation lreval = new Evaluation(dataset);
		lreval.evaluateModel(lr, dataset);
		//System.out.println(lreval.toSummaryString());
		//System.out.println("cross-validate");
		lreval.crossValidateModel(lr, dataset, 10, new Random(1));
		System.out.println(lreval.toSummaryString());
	}
}
