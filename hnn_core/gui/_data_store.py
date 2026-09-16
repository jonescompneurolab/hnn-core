from collections import defaultdict


class DataStore(dict):
    """Centralized store for HNN session data."""

    def __init__(self):

        super().__init__(
            {
                "simulated_data": defaultdict(lambda: dict(net=None, dpls=list())),
                "experimental_data": defaultdict(lambda: dict(net=None, dpls=list())),
                # TODO: Implement load/save networks
                "networks": {},
            }
        )

    @property
    def simulated_data(self):
        return self["simulated_data"]

    @property
    def simulated_data_names(self):
        return self["simulated_data"].keys()

    @property
    def experimental_data_names(self):
        return self["experimental_data"].keys()

    @property
    def experimental_data(self):
        return self["experimental_data"]

    @property
    def networks(self):
        return self["networks"]

    @property
    def all_data_names(self):
        return list(self["simulated_data"].keys()) + list(
            self["experimental_data"].keys()
        )

    def reset(self):
        """Clear all stored simulation/experimental data (i.e. GUI reinitialization)."""
        self["simulated_data"] = defaultdict(lambda: dict(net=None, dpls=list()))
        self["experimental_data"] = defaultdict(lambda: dict(net=None, dpls=list()))
        self["networks"] = {}


# Module-level singleton should be imported by classes that
# want to save data in DataStore
data_store = DataStore()
