import pandas as pd
from pycaret.regression import *
from clearml import Task, Logger

task = Task.init(project_name="lab02", 
                 task_name="TPS_Jan_2022_fixed")
               

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["date"] = pd.to_datetime(train["date"])
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek
train["dayofmonth"] = train["date"].dt.days_in_month
train["dayofyear"] = train["date"].dt.dayofyear
train["weekday"] = train["date"].dt.weekday

test["date"] = pd.to_datetime(test["date"])
test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day"] = test["date"].dt.day
test["dayofweek"] = test["date"].dt.dayofweek
test["dayofmonth"] = test["date"].dt.days_in_month
test["dayofyear"] = test["date"].dt.dayofyear
test["weekday"] = test["date"].dt.weekday

train.drop("date", axis=1, inplace=True)
test.drop("date", axis=1, inplace=True)

parameters = {
'n_jobs' : -1,
'use_gpu' : False,
'data_split_shuffle' : False,
'ignore_features': None
}

ignore_features = None
if(~(parameters["ignore_features"] is None)):
    ignore_features = [ignore_features]

parameters = task.connect(parameters)

reg = setup(
    data=train,
    target="num_sold",
    data_split_shuffle=parameters['data_split_shuffle'],
    use_gpu=parameters['use_gpu'],
    n_jobs=parameters['n_jobs'],
    ignore_features=ignore_features
)

N = 7
top = compare_models(sort="MAPE", n_select=N)

blend = blend_models(top)

final_blend = finalize_model(blend)
result = predict_model(final_blend)

unseen_predictions_blend = predict_model(final_blend, data=test)

sub = pd.DataFrame(
    list(zip(test.row_id, unseen_predictions_blend.prediction_label)),
    columns=["row_id", "num_sold"],
)

task.upload_artifact(name="submission", artifact_object=sub)

mape = pull()['MAPE'][0]
task.upload_artifact(name="MAPE", artifact_object=mape)

logger = Logger.current_logger()
logger.report_single_value(name="MAPE",value=mape)

task.mark_stopped(status_message="Done")







