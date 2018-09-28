from pht.train import Train, cmd_for_train, RunInfo
from pht.response import RunAlgorithmResponse, CheckRequirementsResponse, ListRequirementsResponse


class DistributedLearningTrain(Train):

    def run_algorithm(self, run_info: RunInfo) -> RunAlgorithmResponse:
        pass

    def print_summary(self, run_info: RunInfo) -> str:
        return "foo"

    def check_requirements(self, run_info: RunInfo) -> CheckRequirementsResponse:
        pass

    def list_requirements(self, run_info: RunInfo) -> ListRequirementsResponse:
        return List


if __name__ == '__main__':
    cmd_for_train(DistributedLearningTrain())
