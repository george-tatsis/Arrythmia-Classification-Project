import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import classes

import data_preprocessing
from model_definition import Res_Conv_NN, Inc_Conv_NN, SVM_rbf
import plots_tables

data_obj = data_preprocessing.ProcessData()

# Residual Convolutional Network
model = Res_Conv_NN()
y_test, pred = model.with_resampling(data_obj=data_obj)

plots_tables.print_report(y_true=y_test, 
                        y_pred=pred, 
                        model_name=model._name_,
                        display_labels=classes)

# Inception Convolutional Network
model = Inc_Conv_NN()
y_test, pred = model.with_resampling(data_obj=data_obj)

plots_tables.print_report(y_true=y_test, 
                        y_pred=pred, 
                        model_name=model._name_,
                        display_labels=classes)

# SVM without PCA
model = SVM_rbf()
y_test, pred = model.with_resampling(data_obj=data_obj)

plots_tables.print_report(y_true=y_test, 
                        y_pred=pred, 
                        model_name=model._name_,
                        display_labels=classes)

# SVM with PCA
model = SVM_rbf()
y_test, pred = model.with_resampling(data_obj=data_obj,PCA=True)

plots_tables.print_report(y_true=y_test, 
                        y_pred=pred, 
                        model_name=model._name_,
                        display_labels=classes)