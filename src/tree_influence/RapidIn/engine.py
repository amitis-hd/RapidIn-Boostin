from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ctypes import c_bool, c_int
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tree_influence.RapidIn.utils import save_json, display_progress, load_json
from tree_influence.RapidIn.RapidGrad import RapidGrad
import numpy as np
import time
import json
from pathlib import Path
from copy import copy
import logging
import datetime
import os
import gc
from torch.autograd import grad
from sys import getsizeof
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


from tree_influence.explainers import BoostIn
# import deepspeed


MAX_CAPACITY = 2048
MAX_DATASET_SIZE = int(1e8)


def MP_run_calc_infulence_function(rank, world_size, process_id, mp_engine, X_train, X_test, y_train, y_test , Ks , multi_k_save_path_list, restart = False):
    model = None

    print(f"rank: {rank}, world_size: {world_size}")
    

    train_dataset_size = len(X_train)
    with mp_engine.train_dataset_size.get_lock():
        mp_engine.train_dataset_size.value = train_dataset_size
    with mp_engine.test_dataset_size.get_lock():
        mp_engine.test_dataset_size.value = len(X_test)

    grad_reshape = True
    oporp_eng = None

    oporp_eng = RapidGrad( f"cuda:{rank}")
    grad_reshape = False


    with mp_engine.gpu_locks[rank].get_lock():
        
        # train GBDT model
        model = LGBMClassifier().fit(X_train, y_train)

        # fit influence estimator
        explainer = BoostIn().fit(model, X_train, y_train , oporp_eng , rank)
        
        print(f"CUDA {rank}: Model loaded!")

        temp = []
        temp = explainer.train_leaf_dvs_

        

        #for i, (path, k) in enumerate(zip(multi_k_save_path_list, Ks)):
                            #torch.save(leaf_dvs[i], path + f"/train_grad_K{k}_{real_id:08d}.pt")

        with mp_engine.gpu_locks_num.get_lock():
            display_progress("Model Loading: ", mp_engine.gpu_locks_num.value, 1*world_size, cur_time=time.time())
            mp_engine.gpu_locks_num.value += 1


    #if restart == False:
        #mp_engine.start_barrier.wait()

    
        
    influence = torch.zeros((train_dataset_size, len(X_test)), device=torch.device(f'cuda:{rank}'))
    s_test_vec_t = None
    test_gradients = torch.zeros((90 , 100) , device = torch.device(f'cuda:{rank}'))
    filled = False
    grad_path_name = None
    temp = []
    ops = []

    
    #grad_path_name = "./grads_path/" + f"/train_grad_{real_id:08d}.pt"
    #test_gradients = torch.load(grad_path_name, map_location=f"cuda:{rank}")

    """""
    idx = 0
    while True:

        while True:
            with mp_engine.train_idx.get_lock(), mp_engine.finished_idx.get_lock():
                idx = mp_engine.train_idx.value
                mp_engine.train_idx.value = (mp_engine.train_idx.value + 1)%train_dataset_size
                if mp_engine.finished_idx[idx] == False:
                    mp_engine.finished_idx[idx] = True
                    break
            time.sleep(0.002)
    
        if idx >= train_dataset_size:
            break

    """
    temp = explainer._compute_gradients(X_train , y_train , oporp_eng , rank)
    for i in range(temp.size(0)):
        print(f"shape of gradient at {i} : {temp[i].shape}")
        print(f"gradient: {temp[i]}")
        test_gradients[i] = oporp_eng(temp[i] , 100)
        print(f"rapid grad: {test_gradients[i]}")
    leaf_dvs = torch.zeros((temp.size(0), 100),  device=torch.device(f'cuda:{rank}') )
    
    for i in range (temp.size(0)):
        leaf_dvs[i] = (oporp_eng(temp[i],100))
    

    #for i, (path, k) in enumerate(zip(mp_engine.multi_k_save_path_list, len(test_gradients[i]))):
        #torch.save(test_gradients[i], path + f"/train_grad_K{k}_{real_id:08d}.pt")


    #torch.save(test_gradients, grad_path_name)
    print("before for loop")
    train_leaf_idxs = explainer.model_.apply(X_train)
    test_leaf_idxs = explainer.model_.apply(X_test)  # shape=(X.shape[0], no. boost, no. class
    for i in range(len(X_test)):
        
        mask = np.where(train_leaf_idxs == test_leaf_idxs[i], 1, 0)  # shape=(no. train, no. boost, no. class)
        print(f"mask is {mask}")
        print(f"shape of mask: {mask.shape}")
        print(f"shape of prod: {(leaf_dvs * test_gradients[i]).shape}")
        mask_tensor = torch.tensor(mask , dtype=torch.float32, device = torch.device('cuda:0'))
        mask_tensor = mask_tensor.squeeze() 
        #prod = torch.matmul(train_leaf_dvs , test_gradients[i])* mask  # shape=(no. train, no. boost, no. class)
        
        #prod = torch.einsum('ijk,jk->ijk', train_leaf_dvs, test_gradients[i]) * mask  # shape=(no. train, no. boost, no. class)
            # Debug prints
        #print(f"leaf_dvs shape: {len(leaf_dvs)}")
        #print(f"one element of it: {leaf_dvs[0].shape}")
        #print(f"length of test gradients: {len(test_gradients)}")
        #print(f"test_gradients[{i}] shape: {test_gradients[i].shape}, type: {type(test_gradients[i])}")
        #print(f"mask_tensor shape: {mask_tensor.shape}, type: {type(mask_tensor)}")

        # Ensure all tensors are on the same device
        
        
        prod = leaf_dvs * test_gradients[i] * mask_tensor   # shape=(no. train, no. boost, no. class)

        # sum over boosts and classes
        influence[:, i] = torch.sum(prod)  # shape=(no. train,)
        print(f"influence: {influence}")
        influence_cpu = influence.cpu().numpy()
        values = influence_cpu[:, 0]  # shape=(no. train,)

        # sort training examples from:
        # - most positively influential (decreases loss of the test instance the most), to
        # - most negatively influential (increases loss of the test instance the most)
        training_idxs = np.argsort(values)[::-1]
        print(values[0])
        index = training_idxs[0]
        print(f"Most influential data index: {index}")
        print(X_train[index])
        print(y_train[index])
        print(f"The entire array: {training_idxs}")
        print(f"The entire values array: {values}")
        #if influence != influence: # check if influence is Nan
            #raise Exception('Got unexpected Nan influence!')
        #mp_engine.result_q.put((i, idx, real_id, influence), block=True, timeout=None)



def MP_run_get_result( mp_engine , X_train, X_test, y_train, y_test):
    train_dataset_size = 0
    test_dataset_size = 0
    while True:
        with mp_engine.train_dataset_size.get_lock():
            train_dataset_size = mp_engine.train_dataset_size.value
        with mp_engine.test_dataset_size.get_lock():
            test_dataset_size = mp_engine.test_dataset_size.value
        if train_dataset_size != 0 and (test_dataset_size != 0 ):
            break
        time.sleep(1)
    print(f"train_dataset_size: {train_dataset_size}, test_dataset_size: {test_dataset_size}")

    with mp_engine.train_dataset_size.get_lock(), mp_engine.finished_idx.get_lock():
        if mp_engine.train_dataset_size.value > len(mp_engine.finished_idx):
            raise Exception(f"Size of train dataset larger than MAX_DATASET_SIZE")

    outdir = Path("output_dir")
    outdir.mkdir(exist_ok=True, parents=True)
    influences_path = outdir.joinpath(f"influence_results_"
                                      f"{train_dataset_size}.json")
    influences_path = save_json({}, influences_path, unique_fn_if_exists=True)

   

    

    test_data_dicts = X_test
    #with open(data.test_data_path) as f:
        #list_data_dicts = [json.loads(line) for line in f]
    
    
    #test_data_dicts = read_data(config.data.test_data_path)

    influences = {}
    #TODO: change this config string
    influences['config'] = str("config")
    for k in range(test_dataset_size):
        influences[k] = {}
        influences[k]['test_data'] = test_data_dicts[k]
    
    infl_list = [[0 for _ in range(train_dataset_size)] for _ in range(max(test_dataset_size, 1))]
    real_id2shuffled_id = {}
    shuffled_id2real_id = {}
    
    total_size = max(test_dataset_size, 1) * train_dataset_size
    
    i = 0
    while True:
        try:
            result_item = mp_engine.result_q.get(block=True)
        except Exception as e:
            print("Cal Influence Function Finished!")
            break
    
        if result_item is None:
            save_json(influences, influences_path, overwrite_if_exists=True)
            raise Exception("Get unexpected result from queue.")
        test_id, shuffled_id, real_id, influence = result_item
        if influence != influence: # check if influence is Nan
            raise Exception('Got unexpected Nan influence!')

        infl_list[test_id][shuffled_id] = influence
        real_id2shuffled_id[real_id] = shuffled_id
        shuffled_id2real_id[shuffled_id] = real_id
        with mp_engine.finished_idx.get_lock():
            mp_engine.finished_idx[shuffled_id] = True # due to the calculating retrive data by shuffled_id
        display_progress("Calc. influence function: ", i, total_size, cur_time=time.time())
    
        topk_num = 1000
    
        if ((i + 1)%(total_size//50) == 0 or i == total_size - 1):
            for j in range(test_dataset_size):
                harmful_shuffle_ids = np.argsort(infl_list[j]).tolist()
                harmful = [ shuffled_id2real_id[x] for x in harmful_shuffle_ids if x in shuffled_id2real_id.keys() ]
                helpful = harmful[::-1]
            
                infl = [ x.tolist() if not isinstance(x, int) else x for x in infl_list[j] ]
                # words_infl = [ x.tolist() if not isinstance(x, list) else x for x in words_infl_list ]
                # influences[test_id]['influence'] = infl
                helpful_topk = helpful[:topk_num]
                harmful_topk = harmful[:topk_num]
                influences[j]['helpful'] = copy(helpful_topk)
                influences[j]['helpful_infl'] = copy([infl[x] for x in harmful_shuffle_ids[-topk_num:][::-1]])
                influences[j]['harmful'] = copy(harmful_topk)
                influences[j]['harmful_infl'] = copy([infl[x] for x in harmful_shuffle_ids[:topk_num]])
            influences['finished_cnt'] = f"{i + 1}/{total_size}"
            influences_path = save_json(influences, influences_path, overwrite_if_exists=True)

        i += 1
        if i >= total_size:
            finished = True
            with mp_engine.finished_idx.get_lock():
                for idx in range(train_dataset_size):
                    if mp_engine.finished_idx[idx] == False:
                        print("Warning: i >= total_size, but it have not finished!")
                        finished = False
                        break
            if finished == True:
                break


    influences_path = save_json(influences, influences_path, overwrite_if_exists=True)
    print(influences_path)
    return influences 
    

class MPEngine:
    def __init__(self, world_size , multi_k_save_path_list):
        self.result_q = Queue(maxsize=MAX_CAPACITY)

        self.train_idx = Value(c_int, 0)

        self.start_barrier = Barrier(world_size + 1)
        self.finished_a_test = Value(c_int, 0)
        self.cur_processes_num = Value(c_int, 0)

        self.gpu_locks = [Value(c_int, 0) for _ in range(world_size)]
        self.gpu_locks_num = Value(c_int, 0)


        self.train_dataset_size = Value(c_int, 0)
        self.test_dataset_size = Value(c_int, 0)

        self.finished_idx = Array(c_bool, [False for _ in range(MAX_DATASET_SIZE)])

        # -1, doesn't compute word infl.
        # > -1, compute word infl for # test data
        

        self.multi_k_save_path_list = multi_k_save_path_list

    def action_finished_a_test(self):
        with self.train_idx.get_lock():
            self.train_idx.value = 0


def calc_infl_mp():
    # load iris data
    data = load_iris()
    X, y = data['data'], data['target']

    # use two classes, then split into train and test
    idxs = np.where(y != 2)[0]
    X, y = X[idxs], y[idxs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    Ks = []
    
    gpu_num = torch.cuda.device_count()
    print(f"{gpu_num} GPUs available!")
    grads_path = "./grads_path/"
    threads_per_gpu = 1
    os.makedirs(grads_path, exist_ok=True)
    multi_k_save_path_list = []
    for k in Ks:
        path = grads_path + f"/K{k}"
        multi_k_save_path_list.append(path)
        os.makedirs(path, exist_ok=True)

    num_processing = gpu_num * threads_per_gpu
    mp_engine = MPEngine(num_processing , multi_k_save_path_list )
    MP_run_calc_infulence_function(0, gpu_num, 1*threads_per_gpu + 1,  mp_engine, X_train, X_test, y_train, y_test , Ks , multi_k_save_path_list,)

    """""

    mp_handler = []
    mp_args = []
    print(f"GPU Num: {gpu_num}, Threads per GPU: {threads_per_gpu}")
    for i in range(gpu_num):
        for j in range(threads_per_gpu):
            mp_handler.append(mp.Process(target=MP_run_calc_infulence_function, args=(i, gpu_num, i*threads_per_gpu + j,  mp_engine, X_train, X_test, y_train, y_test , Ks , multi_k_save_path_list,)))
            mp_args.append(mp_handler[-1]._args)
    #mp_handler.append(mp.Process(target=MP_run_get_result, args=( mp_engine , X_train, X_test, y_train, y_test)))

    for x in mp_handler:
        x.start()

    while mp_handler[-1].is_alive():
        print("entering handler loop")
        cur_processes_num = len([1 for x in mp_handler if x.is_alive()])
        print(f"Current processes running: {cur_processes_num}/{num_processing}")
        
        if cur_processes_num < num_processing + 1:
            print(f"ready to restart processing, {cur_processes_num}/{num_processing}")
            for i, x in enumerate(mp_handler):
                print(f"in the for loop at i: {i}")
                if x.is_alive() != True:
                    print(f"start {mp_args[i]}")
                    mp_handler[i] = mp.Process(target=MP_run_calc_infulence_function, args=mp_args[i] + (True,))
                    mp_handler[i].start()
                print("After if")
            continue
        print("after continue")
        with mp_engine.cur_processes_num.get_lock():
            mp_engine.cur_processes_num.value = cur_processes_num
        time.sleep(1)

    for x in mp_handler:
        x.terminate()

"""

