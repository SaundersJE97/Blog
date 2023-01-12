import wandb
import time
import datetime
from typing import Union, Dict
import os
import torch
import pathlib

class WandBLogger(object):
    def __init__(self, config:dict = None, project_name:str = "test", run_name:str = "algo"):
        self.config = config
        self.project_name = project_name
        self.run_name = run_name + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

        self.run = wandb.init(
            project="Lux_{}".format(project_name),
            config=config,
            name=self.run_name,
            dir="{}/../../".format(pathlib.Path(__file__).parent.resolve())
        )

    def log(self, variable_name:str, variable, step:int):
        wandb.log({
            variable_name: variable
        }, step=step)

    def log_dic(self, variable_dic: Dict, step:int):
        wandb.log(variable_dic, step=step)

    def save_model(self, at_global_step:int, state_dict:dict):

        if not os.path.exists("{}/../../model_saves/".format(pathlib.Path(__file__).parent.resolve())):
            os.mkdir("{}/../../model_saves/".format(pathlib.Path(__file__).parent.resolve()))
            print("============== making directory =============")

        torch.save(state_dict, "{}/../../model_saves/model_save_{}_{}_{}.pth".format(pathlib.Path(__file__).parent.resolve(), self.project_name, self.run_name, at_global_step))
        artifact = wandb.Artifact("model_save_{}_{}_{}".format(self.project_name, self.run_name, at_global_step), type='model', description='Episode {}'.format(at_global_step))
        artifact.add_file("{}/../../model_saves/model_save_{}_{}_{}.pth".format(pathlib.Path(__file__).parent.resolve(), self.project_name, self.run_name, at_global_step))
        self.run.log_artifact(artifact)

    def load_model(self, weights_file_name: str):
        return torch.load("{}/../../model_saves/{}".format(pathlib.Path(__file__).parent.resolve(), weights_file_name))