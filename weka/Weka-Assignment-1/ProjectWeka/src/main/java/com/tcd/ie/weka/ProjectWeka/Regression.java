package com.tcd.ie.weka.ProjectWeka;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Regression {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("src/main/java/resources/svm.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		/**
		 * linear regression model
		 *//*
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		//lr.classifyInstance(dataset.)
		//System.out.println(lr);
		Evaluation lreval = new Evaluation(dataset);
		lreval.evaluateModel(lr, dataset);
		System.out.println(lreval.toSummaryString());*/
		/**
		 * svm regression model
		 */
		SMO smoreg = new SMO();
		smoreg.buildClassifier(dataset);
		Evaluation svmregeval = new Evaluation(dataset);
		svmregeval.evaluateModel(smoreg, dataset);
		System.out.println(svmregeval.toSummaryString());
		
	}
}