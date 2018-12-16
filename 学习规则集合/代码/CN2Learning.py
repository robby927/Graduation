# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:30:35 2018

@author: robby
"""

import Orange
from Orange.widgets.visualize.owruleviewer import OWRuleViewer
from AnyQt.QtWidgets import QApplication
from Orange.classification import CN2Learner
data = Orange.data.Table("D://Projects//Robby//RuleLearning//规则集学习数据集.csv")

tabs = Orange.evaluation.testing.sample(data, n=0.7, stratified=False, replace=False)
#print(tabs[0])
#print(tabs[1])


learner = Orange.classification.CN2Learner()
model = learner(tabs[0])

rule_nums = 0
for r in model.rule_list:
    rule_nums += 1
    print(r)
    
print(rule_nums)
  

  
result = Orange.evaluation.TestOnTestData(tabs[0], tabs[1], [learner])
print(result)

precision = Orange.evaluation.Precision(result)
print("precision", precision)

recall = Orange.evaluation.Recall(result)
print("recall", recall)

f1 = Orange.evaluation.F1(result)
print("f1", f1)