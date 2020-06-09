# Copyright (c) 2019 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import time
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
from pfl.core.strategy import WorkModeStrategy
from pfl.core.job_manager import JobManager
from pfl.utils.utils import LoggerFactory
from pfl.entity.runtime_config import WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST, CONNECTED_TRAINER_LIST

LOCAL_AGGREGATE_FILE = os.path.join("tmp_aggregate_pars", "avg_pars")


class Aggregator(object):

    """
    Aggregator is responsible for aggregating model parameters

    """

    def __init__(self, work_mode, job_path, base_model_path, concurrent_num=5):
        self.job_path = job_path
        self.base_model_path = base_model_path
        self.aggregate_executor_pool = ThreadPoolExecutor(concurrent_num)
        self.work_mode = work_mode


    def load_model_pars(self, job_model_pars_path, fed_step):
        """

        :param job_model_pars_path:
        :param fed_step:
        :return:
        """
        fed_step = 0 if fed_step is None else fed_step
        job_model_pars = []
        last_model_par_file_num = 0
        # print("job_model_pars_path: ", job_model_pars_path)
        for f in os.listdir(job_model_pars_path):
            if f.find("models_") != -1:
                client_id = int(f.split("_")[-1])
                one_model_par_path = os.path.join(job_model_pars_path, f)
                # print("one_model_par_path: ", one_model_par_path)
                one_model_par_files = os.listdir(one_model_par_path)
                if one_model_par_files and len(one_model_par_files) != 0:
                    last_model_par_file_num = self._find_last_model_file_num(one_model_par_files)
                    if last_model_par_file_num > fed_step:
                        model_par = torch.load(os.path.join(one_model_par_path, one_model_par_files[-1]))
                        job_model_pars.append({"client_id":client_id, "model_par":model_par})
                    else:
                        return None, 0
                else:
                    # wait for other clients finish training
                    return None, 0

        return job_model_pars, last_model_par_file_num

    def _find_last_model_file_num(self, files):
        last_num = 0
        for file in files:
            file_num = int(file.split("_")[-1])
            last_num = file_num if last_num < file_num else last_num
        return last_num


class FedAvgAggregator(Aggregator):
    """
    FedAvgAggregator is responsible for aggregating model parameters by using FedAvg Algorithm

    """

    def __init__(self, work_mode, job_path, base_model_path):
        super(FedAvgAggregator, self).__init__(work_mode, job_path, base_model_path)
        self.fed_step = {}
        self.logger = LoggerFactory.getLogger("FedAvgAggregator", logging.INFO)



    def calc_client_priority(self, fed_step, job_id):
        client_priority_weights = {}
        # kl_loss_dir = os.path.join(self.base_model_path,"models_{}".format(job_id),"kl_loss_dir")
        # if not os.path.exists(kl_loss_dir):
        #     os.mkdir(kl_loss_dir)
        kl_loss_dir = os.path.join(self.base_model_path,"models_{}".format(job_id),"kl_loss_dir")
        kl_loss_path = os.path.join(self.base_model_path,"models_{}".format(job_id),"kl_loss_dir","{}_{}".format("kl_loss", fed_step))

        # if not os.path.exists(kl_loss_dir):
        #     return None

        with open(kl_loss_path, "r") as f:
            lines = f.readlines()

        sum = 0
        for line in lines:
            line_split = line.split(":")
            client_priority_weights[int(line_split[0])] = float(line_split[1])
            sum += float(line_split[1])

        for key in client_priority_weights.keys():
            flag = client_priority_weights[key]
            client_priority_weights[key] = flag / sum

        return client_priority_weights


    def aggregate(self):
        """

        :return:
        """

        job_list = JobManager.get_job_list(self.job_path)
        WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST.clear()
        for job in job_list:
            job_model_pars, fed_step = self.load_model_pars(
                os.path.join(self.base_model_path, "models_{}".format(job.get_job_id())),
                self.fed_step.get(job.get_job_id()))
            # print("fed_step: {}, self.fed_step: {}, job_model_pars: {}".format(fed_step, self.fed_step.get(job.get_job_id()), job_model_pars))
            job_fed_step = 0 if self.fed_step.get(job.get_job_id()) is None else self.fed_step.get(job.get_job_id())
            if job_fed_step != fed_step and job_model_pars is not None:
                self.logger.info("计算每个节点的权重....")
                self.client_priority_weights = self.calc_client_priority(fed_step, job.get_job_id())
                print("每个节点权重为：", self.client_priority_weights)
                self.logger.info("Aggregating......")
                # # if self.client_priority_weights is None:
                #     self._exec_avg(job_model_pars, self.base_model_path, job.get_job_id(), fed_step)
                # else:
                self._exec_with_weights(job_model_pars, self.base_model_path, job.get_job_id(), fed_step, self.client_priority_weights)
                self.fed_step[job.get_job_id()] = fed_step
                WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST.append(job.get_job_id())
                if job.get_epoch() <= self.fed_step[job.get_job_id()]:
                    self._save_final_model_pars(job.get_job_id(), os.path.join(self.base_model_path,
                                                                               "models_{}".format(job.get_job_id()),
                                                                               "tmp_aggregate_pars"),
                                                self.fed_step[job.get_job_id()])
                if self.work_mode == WorkModeStrategy.WORKMODE_CLUSTER:
                    self._broadcast(WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST, CONNECTED_TRAINER_LIST,
                                    self.base_model_path)


    def _exec_with_weights(self, job_model_pars, base_model_path, job_id, fed_step, client_priority_weights):
        avg_model_par = job_model_pars[0]['model_par']
        avg_client_id = job_model_pars[0]['client_id']
        for key in avg_model_par.keys():
            avg_model_par[key] = avg_model_par[key] * client_priority_weights[avg_client_id]
        for key in avg_model_par.keys():
            for i in range(1, len(job_model_pars)):
                client_id = job_model_pars[i]['client_id']
                avg_model_par[key] += job_model_pars[i]['model_par'][key] * client_priority_weights[client_id]
            # avg_model_par[key] = torch.div(avg_model_par[key], len(job_model_pars))
        tmp_aggregate_dir = os.path.join(base_model_path, "models_{}".format(job_id))
        tmp_aggregate_path = os.path.join(base_model_path, "models_{}".format(job_id),
                                          "{}_{}".format(LOCAL_AGGREGATE_FILE, fed_step))
        if not os.path.exists(tmp_aggregate_dir):
            os.makedirs(tmp_aggregate_dir)
        torch.save(avg_model_par, tmp_aggregate_path)

        self.logger.info("job: {} the {}th round parameters aggregated successfully!".format(job_id, fed_step))

    def _exec_avg(self, job_model_pars, base_model_path, job_id, fed_step):
        avg_model_par = job_model_pars[0]['model_par']
        for key in avg_model_par.keys():
            for i in range(1, len(job_model_pars)):
                avg_model_par[key] += job_model_pars[i]['model_par'][key]
            avg_model_par[key] = torch.div(avg_model_par[key], len(job_model_pars))
        tmp_aggregate_dir = os.path.join(base_model_path, "models_{}".format(job_id))
        tmp_aggregate_path = os.path.join(base_model_path, "models_{}".format(job_id),
                                          "{}_{}".format(LOCAL_AGGREGATE_FILE, fed_step))
        if not os.path.exists(tmp_aggregate_dir):
            os.makedirs(tmp_aggregate_dir)
        torch.save(avg_model_par, tmp_aggregate_path)

        self.logger.info("job: {} the {}th round parameters aggregated successfully!".format(job_id, fed_step))

    def _broadcast(self, job_id_list, connected_client_list, base_model_path):
        """

        :param job_id_list:
        :param connected_client_list:
        :param base_model_path:
        :return:
        """
        aggregated_files = self._prepare_upload_aggregate_file(job_id_list, base_model_path)
        self.logger.info("connected client list: {}".format(connected_client_list))
        for client in connected_client_list:
            client_url = "http://{}".format(client)
            response = requests.post("/".join([client_url, "aggregatepars"]), data=None, files=aggregated_files)
            # print(response)

    def _prepare_upload_aggregate_file(self, job_id_list, base_model_path):
        """

        :param job_id_list:
        :param base_model_path:
        :return:
        """
        aggregated_files = {}
        for job_id in job_id_list:
            tmp_aggregate_dir = os.path.join(base_model_path, "models_{}".format(job_id), "tmp_aggregate_pars")
            fed_step = self._find_last_model_file_num(os.listdir(tmp_aggregate_dir))
            send_aggregate_filename = "tmp_aggregate_{}_{}".format(job_id, fed_step)
            tmp_aggregate_path = os.path.join(tmp_aggregate_dir, "avg_pars_{}".format(fed_step))
            aggregated_files[send_aggregate_filename] = (send_aggregate_filename, open(tmp_aggregate_path, "rb"))
        return aggregated_files

    def _save_final_model_pars(self, job_id, tmp_aggregate_dir, fed_step):
        """

        :param job_id:
        :param tmp_aggregate_dir:
        :param fed_step:
        :return:
        """
        job_model_dir = os.path.join(self.base_model_path, "models_{}".format(job_id))
        final_model_pars_path = os.path.join(os.path.abspath("."), "final_model_pars_{}".format(job_id))
        last_aggregate_file = os.path.join(tmp_aggregate_dir, "avg_pars_{}".format(fed_step))
        with open(final_model_pars_path, "wb") as final_f:
            with open(last_aggregate_file, "rb") as f:
                for line in f.readlines():
                    final_f.write(line)

        self.logger.info("job {} save final aggregated parameters successfully!".format(job_id))


class DistillationAggregator(Aggregator):
    """
    Model distillation does not require a centralized server that we don't need to provide a Distillation aggregator
    """
    def __init__(self, work_mode, job_path, base_model_path):
        super(DistillationAggregator, self).__init__(work_mode, job_path, base_model_path)
        self.fed_step = {}

    def aggregate(self):
        pass
