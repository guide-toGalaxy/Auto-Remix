from abc import ABC, abstractmethod
import numpy as np

class BatchAggregator(ABC):
    """BatchAggregator is an abstract class that provides an interface
    to apply multiple statistical aggregation on 2d numpy arrays.
    """

    def __init__(self):
        self.aggregators = []

    def add_aggregator(self, aggregator):
        """Add an aggregator function to the aggregators.

        :param aggregator: (Aggregator) Concrete Aggregator
        """
        self.aggregators.append(aggregator)

    @abstractmethod
    def aggregate(self, array):
        pass

class FlatBatchAggregator(BatchAggregator):
    """FlatBatchAggregator is a concrete BatchAggregator that applies multiple
    statistical aggregation on 2d numpy arrays and merges them into a single
    array.
    """

    def aggregate(self, array):
        """Perform statistical aggregations on 2d array and merge
        aggregations.

        :param array: (2d numpy array)

        :return (np.ndarray) Aggregated and merged values
        """
        merged_aggregations = []
        for aggregator in self.aggregators:
            aggregation = aggregator.aggregate(array)
            merged_aggregations.append(aggregation)
        return concatenate_arrays(merged_aggregations)


class Aggregator(ABC):
    """Interface for a concrete statistical aggregator."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def aggregate(self, array):
        pass
        
class MeanAggregator(Aggregator):
    """MeanAggregator is responsible for aggregating a array using mean
    across a specified axis.
    """

    def __init__(self, aggregation_axis):
        super().__init__("mean")
        self.aggregation_axis = aggregation_axis

    def aggregate(self, array):
        """Aggregate array using mean across 1 axis

        :param array: (np.ndarray)

        :return: (np.ndarray) Aggregated array
        """
        return np.mean(array, axis=self.aggregation_axis)
