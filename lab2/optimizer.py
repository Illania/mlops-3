from pycaret.regression import *
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer)
from clearml.automation.optuna import OptimizerOptuna

task = Task.init(project_name="lab02", 
                 task_name="TPS_Jan_2022_optimizer",
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

args = {
    'template_task_id': "621c7d4af6e54e00b6cca42a6f321605",
    'run_as_service': False,
}


an_optimizer = HyperParameterOptimizer(
    # specifying the task to be optimized, task must be in system already so it can be cloned
    base_task_id=args['template_task_id'],

    # setting the hyperparameters to optimize
     hyper_parameters=[
     DiscreteParameterRange('General/ignore_features', values=['country', 'store', 'product']),
    ],

   # setting the objective metric we want to maximize/minimize
   objective_metric_title='accuracy',
   objective_metric_series='MSE',
   objective_metric_sign='min',
      
   # setting optimizer  
   optimizer_class=OptimizerOptuna,
  
   # configuring optimization parameters
    execution_queue='default',  
    max_number_of_concurrent_tasks=2,  
    optimization_time_limit=60., 
    compute_time_limit=120, 
    total_max_jobs=20,  
    min_iteration_per_job=15000,  
    max_iteration_per_job=150000,
)

an_optimizer.start()
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
task.upload_artifact('top_exp', top_exp)
an_optimizer.wait()
an_optimizer.stop()






