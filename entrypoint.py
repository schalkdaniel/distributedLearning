from pht.train import Train, cmd_for_train, RunInfo
from pht.response import RunAlgorithmResponse, CheckRequirementsResponse, ListRequirementsResponse
import subprocess
import requests
import os

SLASH = '/'
NUMBER_OF_STATIONS = 3


def file_endpoint():
    endpoint = os.environ['PHT_FILE_DOWNLOAD_SERVICE']
    return endpoint[:-1] if endpoint.endswith(SLASH) else endpoint


# Where the Iris files are gonna be stored
def data_dir():
    return os.environ['DATA_DIR']


class DistributedLearningTrain(Train):

    def run_algorithm(self, run_info: RunInfo) -> RunAlgorithmResponse:

        station_id = run_info.station_id

        # Assume that the station serves the iris file
        iris_filename = 'iris{}.csv'.format(station_id)
        iris_filepath = os.path.join(data_dir(), iris_filename)
        r = requests.get(file_endpoint() + SLASH + iris_filename)
        with open(iris_filepath, 'w') as f:
            f.write(r.text)

        # Calculate the new train tag
        next_train_tag = 'station.{}'.format((station_id + 1) % NUMBER_OF_STATIONS)
        return RunAlgorithmResponse(True, next_train_tag)

    def print_summary(self, run_info: RunInfo) -> str:
        return '\n'.join(os.listdir(data_dir()))

    def check_requirements(self, run_info: RunInfo) -> CheckRequirementsResponse:
        pass

    def list_requirements(self, run_info: RunInfo) -> ListRequirementsResponse:
        pass


if __name__ == '__main__':
    cmd_for_train(DistributedLearningTrain())
