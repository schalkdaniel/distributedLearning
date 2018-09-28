from pht.train import Train, cmd_for_train, RunInfo
from pht.response import RunAlgorithmResponse, CheckRequirementsResponse, ListRequirementsResponse
import subprocess
import requests
import os
import shutil
from typing import IO

SLASH = '/'
NUMBER_OF_STATIONS = 3


################################################################################################
# Helper
################################################################################################
def without_trailing_slash(s: str):
    return s[:-1] if s.endswith(SLASH) else s


def extend(ls, *producer):
    for proc in producer:
        ls.extend(proc.readlines())


def process_script_with_r(script: str):
    """
    Opens the provided script with Rscript and returns the stdout of the process upon completion
    :param script:
    :return:
    """
    out = []
    with subprocess.Popen(['Rscript', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        with p.stdout as stdout:
            with p.stderr as stderr:
                while p.poll() is None:
                    extend(out, stdout, stderr)
                extend(out, stdout, stderr)
    return '\n'.join([x.decode('utf-8') for x in out])


################################################################################################
# The environment the train lives isn
################################################################################################
def file_endpoint():
    return without_trailing_slash(os.environ['PHT_FILE_DOWNLOAD_SERVICE'])


# Where the Iris files are gonna be stored
def data_dir():
    return os.environ['DATA_DIR']


# Where the R code for the training lives
def code_dir():
    return os.environ['CODE_DIR']


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
    r = requests.get(file_endpoint() + SLASH + iris_filename)

    # If the file could not be fetched from the service, the execution was unsuccessful
    if r.status_code != 200:
        return False

    iris_filepath = os.path.join(data_dir(), iris_filename)
    with open(iris_filepath, 'w') as f:
        f.write(r.text)
    return True


def training():
    return process_script_with_r(os.path.join(code_dir(), 'training.R'))


class DistributedLearningTrain(Train):

    @staticmethod
    def run_algorithm(run_info: RunInfo) -> RunAlgorithmResponse:

        # Calculate the new train tag (This train only modulates the number of station)
        next_train_tag = 'station.{}'.format((run_info.station_id + 1) % NUMBER_OF_STATIONS)

        # First, ensure that there is no data left
        clear_data()

        try:
            # Now fetch the data from the local service
            if not fetch_data(run_info.station_id):
                return RunAlgorithmResponse(False, next_train_tag, "Error fetching file from service")

            # Perform the update on the data model
            stdout = training()

            # Ensure the data dir is cleaned
            clear_data()

            return RunAlgorithmResponse(True, next_train_tag, message=stdout)

        finally:
            clear_data()

    def print_summary(self, run_info: RunInfo) -> str:
        return '\n'.join(os.listdir(data_dir()))

    def check_requirements(self, run_info: RunInfo) -> CheckRequirementsResponse:
        pass

    def list_requirements(self, run_info: RunInfo) -> ListRequirementsResponse:
        pass


if __name__ == '__main__':
    cmd_for_train(DistributedLearningTrain())
