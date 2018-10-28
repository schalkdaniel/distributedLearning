from pht.train import Train, cmd_for_train, StationInfo
from pht.response import \
    RunAlgorithmResponse,\
    CheckRequirementsResponse,\
    ListRequirementsResponse,\
    PrintSummaryResponse
import subprocess
import requests
import os
import shutil
from typing import IO, List, Tuple

# PHT Helper Methods
from pht.process import process
from pht.env import env_exists, from_env_without_trailing_slashes

NUMBER_OF_STATIONS = 2
PHT_FILE_DOWNLOAD_SERVICE = 'PHT_FILE_DOWNLOAD_SERVICE'

# The base image from which the successor images should be crated from
BASE_IMAGE = 'personalhealthtrain/train_iris:base'


################################################################################################
# Helper
################################################################################################
def process_script_with_r(script: str) -> Tuple[List[str], List[str]]:
    """
    Opens the provided script with Rscript and returns the stdout of the process upon completion
    :param script:
    :return:
    """
    return process(['Rscript', script])


################################################################################################
# The environment the train lives in
################################################################################################
def file_endpoint_exists():
    return env_exists(PHT_FILE_DOWNLOAD_SERVICE)


def file_endpoint():
    return from_env_without_trailing_slashes(PHT_FILE_DOWNLOAD_SERVICE)


# Where the Iris files are gonna be stored
def data_dir():
    return from_env_without_trailing_slashes('DATA_DIR')


# Where the R code for the training lives
def code_dir():
    return from_env_without_trailing_slashes('CODE_DIR')


def model_dir():
    return from_env_without_trailing_slashes('MODEL_DIR')


def finish_file():
    return os.path.join(model_dir(), 'FINISH')


# File containing the current number of iteration
def iteration_file():
    return os.path.join(model_dir(), 'ITERATION')


# List absolute paths of files that need to be exported
def export_files():
    d = os.path.join(model_dir(), 'train_files')
    return [os.path.join(d, x) for x in os.listdir(d)]


################################################################################################
# Train Routines
################################################################################################
def clear_data():
    d = data_dir()
    shutil.rmtree(d)
    os.mkdir(d)


def fetch_data(station_id) -> bool:

    # The name of the file that we are gonna expect from the service
    iris_filename = 'iris{}.csv'.format(station_id)

    # Fetch the file from the remote service
    r = requests.get(file_endpoint() + '/' + iris_filename)

    # If the file could not be fetched from the service, the execution was unsuccessful
    if r.status_code != 200:
        return False

    iris_filepath = os.path.join(data_dir(), iris_filename)
    with open(iris_filepath, 'w') as f:
        f.write(r.text)
    return True


def process_code(scriptname: str):
    """
    Process some script in the code directory
    """
    return process_script_with_r(os.path.join(code_dir(), scriptname))


def to_message(*args) -> str:
    res = []
    for arg in args:
        res.extend(arg)
    return '\n'.join(res)


def training():
    return process_code('training.R')


def print_summary():
    return process_code('print_summary.R')


def get_next_train_tag(station_id: int):
    """
    Input: The _current_ station_id
    """
    it_file = iteration_file()

    # Init the itfile if it does not exist
    if not os.path.exists(it_file):
        with open(it_file, 'w') as f:
            f.write(str(0))

    if os.path.isfile(finish_file()):
        return 'finish'

    # Increase the current number of iteration
    with open(it_file, 'r') as f:
        iteration = int(f.readline().strip()) + 1
    # Write the iteration back
    with open(it_file, 'w') as f:
        f.write(str(iteration))
    return '{}-station.{}'.format(iteration, (station_id + 1) % NUMBER_OF_STATIONS)


class DistributedLearningTrain(Train):

    def run_algorithm(self, station_info: StationInfo) -> RunAlgorithmResponse:

        # Calculate the new train tag (This train only modulates the number of station)
        next_train_tag = get_next_train_tag(station_info.station_id)

        # First, ensure that there is no data left
        clear_data()

        try:

            # Now fetch the data from the local service
            if not fetch_data(station_info.station_id):
                return RunAlgorithmResponse(
                    success=False,
                    message="Error fetching file from service",
                    next_train_tag=next_train_tag,
                    docker_base_image=BASE_IMAGE,
                    export_files=export_files())

            # Perform the update on the data model
            (stdout, stderr) = training()

            # Ensure the data dir is cleaned
            clear_data()

            return RunAlgorithmResponse(
                success=True,
                message=to_message(stdout, stderr),
                next_train_tag=next_train_tag,
                docker_base_image=BASE_IMAGE,
                export_files=export_files())

        finally:
            clear_data()

    def print_summary(self, station_info: StationInfo) -> PrintSummaryResponse:
        (stdout, stderr) = print_summary()

        return PrintSummaryResponse(to_message(stdout, stderr))

    def check_requirements(self, station_info: StationInfo) -> CheckRequirementsResponse:
        unmet = [] if file_endpoint_exists() else [PHT_FILE_DOWNLOAD_SERVICE]
        return CheckRequirementsResponse(unmet=unmet)

    def list_requirements(self, station_info: StationInfo) -> ListRequirementsResponse:
        return ListRequirementsResponse([PHT_FILE_DOWNLOAD_SERVICE])


if __name__ == '__main__':
    cmd_for_train(DistributedLearningTrain())
