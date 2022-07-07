import sys
import yaml
import ast
from input_MLP import MultiModelCreate

with open('Configs/config.yaml') as cf_file:
    config = yaml.safe_load(cf_file.read())
    load_model=ast.literal_eval(config.get("load_model"))
    mb_layers = int(config.get("mb_layers"))
    mb_members = int(config.get("mb_members"))
    mb_ensemble = ast.literal_eval(config.get("mb_ensemble"))
    mb_networkUnits = int(config.get("mb_networkUnits"))
    mb_batchSize = int(config.get("mb_batchSize"))
    mb_init_epochs = int(config.get("mb_init_epochs"))
    mb_scaler = str(config.get("mb_scaler"))
    mb_aggregate = ast.literal_eval(config.get("mb_aggregate"))
    mb_aggregate_epochs = int(config.get("mb_aggregate_epochs"))
    mb_max_rollot = int(config.get("mb_max_rollot"))
    mb_collection_length = int(config.get("mb_collection_length"))
    mb_num_aggregate = int(config.get("mb_num_aggregate"))
    aggregate_every_iter = int(config.get("aggregate_every_iter"))
    fraction_use_new = float(config.get("fraction_use_new"))
    mb_train_every_iter = int(config.get("mb_train_every_iter"))
    timesteps_on_mb = int(config.get("timesteps_on_mb"))
    mb_epoch = int(config.get("mb_epoch"))

def create_ensemble(self,input_dim = 111,out_dim = 94, activation_d='lrelu', activation_op='lrelu', networkUnits=64, nlayers=5, load_model = False):
    members = []
    for i in range(self.nmembers):
        model = MultiModelCreate(input_dim, out_dim, activation_d, activation_op, networkUnits,
                                 nlayers=nlayers, model_name=str(i))
        members.append(model)
    head_input_dim = out_dim*self.nmembers
    head_out_dim = out_dim
    if self.ensemble:
        meta_model = MultiModelCreate(head_input_dim, head_out_dim, activation_d, activation_op, networkUnits,
                             nlayers=1, model_name="meta")
    if load_model:
        for i in range(len(members)):
            members[i].load_weights("best_model_{}".format(str(i)))
        if self.ensemble:
            meta_model.load_weights("best_model_meta")
    if self.ensemble:
        return members,meta_model
    return (members)

create_ensemble()