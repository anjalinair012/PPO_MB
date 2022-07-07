import neptune.new as neptune
import yaml

class Neptune_Logger:
    def __init__(self):
        self.run = neptune.init(
        project="anjalinair012/Anjali-MB",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZWFjNmViZi00NmE5LTQzZjktYjcwNi03ZjU0MjBlY2M2NmEifQ==",
    )  # your credentials


with open('Configs/config.yaml') as cf_file:
    config = yaml.safe_load(cf_file.read())
    load_model=config.get("load_model")
    mb_layers = int(config.get("mb_layers"))
    mb_members = int(config.get("mb_members"))
    mb_ensemble = config.get("mb_ensemble")
    mb_networkUnits = int(config.get("mb_networkUnits"))
    mb_batchSize = int(config.get("mb_batchSize"))
    mb_init_epochs = int(config.get("mb_init_epochs"))
    mb_scaler = str(config.get("mb_scaler"))
    mb_aggregate = config.get("mb_aggregate")
    #mb_aggregate_epochs = int(config.get("mb_aggregate_epochs"))
    mpc_max_rollot = int(config.get("mpc_max_rollot"))
    mpc_collection_length = int(config.get("mpc_collection_length"))
    mb_num_aggregate = int(config.get("mb_num_aggregate"))
    aggregate_every_iter = int(config.get("aggregate_every_iter"))
    fraction_use_new = float(config.get("fraction_use_new"))
    mb_train_every_iter = int(config.get("mb_train_every_iter"))
    timesteps_on_mb = int(config.get("timesteps_on_mb"))
    mb_epoch = int(config.get("mb_epoch"))
params = {"load_model": load_model,"mb_layers":mb_layers,"mb_members":mb_members,"mb_ensemble": mb_ensemble,
          "mb_networkUnits": mb_networkUnits, "mb_batchSize" :mb_batchSize,"mb_init_epochs":mb_init_epochs,
          "mb_scaler":mb_scaler,"mb_aggregate":mb_aggregate,"mpc_max_rollot":mpc_max_rollot,
          "mb_num_aggregate":mb_num_aggregate,"aggregate_every_iter":aggregate_every_iter,
          "fraction_use_new":fraction_use_new,"mb_train_every_iter":mb_train_every_iter,
          "timesteps_on_mb":timesteps_on_mb,"mb_epoch":mb_epoch}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()