package com.tcd.ie.weka.ProjectWeka;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SVMWeka {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("src/main/java/resources/svm.arff");
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
