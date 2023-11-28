from src.experiment import ExperimentLauncher

config_path = 'configuration/pruebas/'
# experiment_launcher = ExperimentLauncher(config_path, save_file='results/pruebas/results_nn_optimizing_bs_and_lr.csv', search_type='bayesian', iterations=25)
experiment_launcher = ExperimentLauncher(config_path, save_file='results/pruebas/test.csv', search_type='grid')
experiment_launcher.run()