import glob
import numpy as np
import logging
import re
import datetime
import json


def calculate_test_error():
    error_map = {}

    for dataset in glob.glob("experiments/*"):
        dataset = dataset.split("/")[-1]
        error_map[dataset] = {}

        for model in glob.glob(f"experiments/{dataset}/*"):
            model = model.split("/")[-1]
            error_map[dataset][model] = {}

            for exp_id in glob.glob(f"experiments/{dataset}/{model}/*"):
                exp_id = exp_id.split("/")[-1]
                error_map[dataset][model][exp_id] = {}

                for test_id in sorted(glob.glob(f"experiments/{dataset}/{model}/{exp_id}/*")):
                    test_id = test_id.split("/")[-1]

                    if model == "lenet":
                        checkpoint_folder = "checkpoint_0010"
                    elif model == "resnet_preact":
                        checkpoint_folder = "checkpoint_0040"
                    elif model == "alexnet":
                        checkpoint_folder = "checkpoint_0015"
                    elif model == "resnet_simple_14":
                        checkpoint_folder = "checkpoint_0015"

                    try:
                        with open(f"experiments/{dataset}/{model}/{exp_id}/{test_id}/eval/{checkpoint_folder}/error.txt") as f:
                            error_val = f.read()
                        error_map[dataset][model][exp_id][test_id] = error_val
                    except:
                        logging.error(f"File 'experiments/{dataset}/{model}/{exp_id}/{test_id}/eval/{checkpoint_folder}/error.txt' not found")
                
                avg = np.mean(list(map(float, error_map[dataset][model][exp_id].values())))
                error_map[dataset][model][exp_id]["avg"] = avg

    return error_map



def calculate_training_time():
    time_map = {}

    timestamp_pattern = re.compile("\[(\d{4}\-\d{2}\-\d{2} \d{2}:\d{2}:\d{2})\]")
    epoch_pattern = re.compile("Epoch (\d{2})")

    for dataset in glob.glob("experiments/*"):
        dataset = dataset.split("/")[-1]
        time_map[dataset] = {}

        for model in glob.glob(f"experiments/{dataset}/*"):
            model = model.split("/")[-1]
            time_map[dataset][model] = {}

            for exp_id in glob.glob(f"experiments/{dataset}/{model}/*"):
                exp_id = exp_id.split("/")[-1]
                time_map[dataset][model][exp_id] = {}

                for test_id in sorted(glob.glob(f"experiments/{dataset}/{model}/{exp_id}/*")):
                    test_id = test_id.split("/")[-1]

                    try:
                        with open(f"experiments/{dataset}/{model}/{exp_id}/{test_id}/log_plain.txt", "r") as f:
                            content = f.read()
                            timestamp_list = re.findall(timestamp_pattern, content)
                            epoch_list = re.findall(epoch_pattern, content)
                        first_timestamp = datetime.datetime.strptime(timestamp_list[0], '%Y-%m-%d %H:%M:%S')
                        last_timestamp = datetime.datetime.strptime(timestamp_list[-1], '%Y-%m-%d %H:%M:%S')
                        timestamp_difference = last_timestamp - first_timestamp
                        difference_per_epoch = timestamp_difference / int(epoch_list[-1])

                        time_map[dataset][model][exp_id][test_id] = difference_per_epoch
                    except:
                        logging.error(f"File 'experiments/{dataset}/{model}/{exp_id}/{test_id}/log_plain.txt' not found")
                
                timedeltas = time_map[dataset][model][exp_id].values()
                avg = sum(timedeltas, datetime.timedelta(0)) / len(timedeltas)
                # avg = np.mean(list(map(float, time_map[dataset][model][exp_id].values())))
                time_map[dataset][model][exp_id]["avg"] = str(avg)
                time_map[dataset][model][exp_id] = {key: str(value) for key, value in time_map[dataset][model][exp_id].items()}
    
    return time_map

if __name__ == '__main__':
    error = calculate_test_error()
    time_map = calculate_training_time()

    print("Test error:")
    print(json.dumps(error, sort_keys=True, indent=4))
    print()
    print("=====================================================")
    print()
    print("Training time:")
    print(json.dumps(time_map, sort_keys=True, indent=4))