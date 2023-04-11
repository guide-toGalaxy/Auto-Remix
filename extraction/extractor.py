from abc import ABC, abstractmethod
import os
import sys
import librosa
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

class Extractor(ABC):
    """Interface for feature extractors."""
    def __init__(self, feature_name):
        self.feature_name = feature_name
    @abstractmethod
    def extract(self, signal, sample_rate):
        pass
class ChromogramExtractor(Extractor):
    """Concrete Extractor that extracts chromogram from signal."""

    def __init__(self, frame_size=1024, hop_length=512):
        super().__init__("chromogram")
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal, sample_rate):
        """Extract chromogram from time series using librosa.

        :param signal: (np.ndarray) Audio time series
        :param sr: (int) Sample rate

        :return: (np.ndarray) Chromogram
        """
        chromogram = librosa.feature.chroma_stft(y=signal,
                                                 n_fft=self.frame_size,
                                                 hop_length=self.hop_length,
                                                 sr=sample_rate)
        return chromogram

class MFCCExtractor(Extractor):
    """Concrete Extractor that extracts MFCC sequences from signal."""

    def __init__(self, frame_size=1024, hop_length=512, num_coefficients=13):
        super().__init__("mfcc")
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.num_coefficients = num_coefficients

    def extract(self, signal, sample_rate):
        """Extract MFCC from time series using librosa.

        :param signal: (np.ndarray) Audio time series
        :param sr: (int) Sample rate

        :return: (np.ndarray) MFCC sequence
        """
        mfcc = librosa.feature.mfcc(y=signal,
                                    n_mfcc=self.num_coefficients,
                                    n_fft=self.frame_size,
                                    hop_length=self.hop_length,
                                    sr=sample_rate)
        return mfcc

class BatchExtractor:
    """Batch extract features dynamically from a list of audio files."""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.extractors = []
        self._features = {}

    def add_extractor(self, extractor):
        """Add a concrete Extractor to the extractors.

        :param extractor: (Extractor) Concrete Extractor (e.g., MFCCExtractor)
        """
        self.extractors.append(extractor)

    def extract(self, dir):

        features = {}
        for root, _, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_features = self._extract_features_for_file(file_path)
                features[file_path] = file_features
        print(features)
        return features

    def _extract_features_for_file(self, file_path):
        features = {}
        signal = librosa.load(file_path, sr=self.sample_rate)[0]
        for extractor in self.extractors:
            feature = extractor.extract(signal, self.sample_rate)
            features[extractor.feature_name] = feature
        return features

class Aggregator(ABC):
    """Interface for a concrete statistical aggregator."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def aggregate(self, array):
        pass

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
        return np.hstack(merged_aggregations)

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

class MultiTrackBatchAggregator:
    def __init__(self):
        self.batch_aggregator = None

    def aggregate(self, tracks_features):
   
        tracks_aggregations = {}
        for track_path, track_features in tracks_features.items():
            features_aggregations = {}
            for feature_type, features in track_features.items():
                aggregations = self.batch_aggregator.aggregate(features)
                features_aggregations[feature_type] = aggregations
            tracks_aggregations[track_path] = features_aggregations
        return tracks_aggregations
        
def mapping_creation(agg):
	return list(agg.keys())
	
def dataset_creation(agg):
	dataset = list(agg.values())
	dataset = np.asarray(dataset)
	return dataset
        
def save_feat(save_path,data):
	with open(save_path, "wb") as file:
		pickle.dump(data, file)

def concat_arr(feat):
	features = [feature for feature in feat.values()]
	new_features = np.hstack(features)
	return new_features

def getting_embeddings(features):
	embeddings={}
	for path,features in features.items():
		new_arr=concat_arr(features)
		embeddings[path]=new_arr
	return embeddings
		
	
	


if __name__ == "__main__":

    num_mfccs = 13
    frame_size = 1024
    hop_size = 512

    dir = sys.argv[1]

    mfcc_extractor = MFCCExtractor(frame_size, hop_size, num_mfccs)
    chromogram_extractor = ChromogramExtractor(frame_size, hop_size)

    batch_extractor = BatchExtractor(22050)
    #batch_extractor.add_extractor(mfcc_extractor)
    batch_extractor.add_extractor(chromogram_extractor)
    features = batch_extractor.extract(dir)

    batch_aggregator = FlatBatchAggregator()
    mean_aggregator = MeanAggregator(1)
    batch_aggregator.add_aggregator(mean_aggregator)
    
    
    mtba = MultiTrackBatchAggregator()
    mtba.batch_aggregator = batch_aggregator
    aggregated=mtba.aggregate(features)
    
    embeddings=getting_embeddings(aggregated)
    
    mappings=mapping_creation(embeddings)
    dataset=dataset_creation(embeddings)
    
    save_feat(sys.argv[2],mappings)
    save_feat(sys.argv[3],dataset)
    
    nearest_neighbour = NearestNeighbors()
    nearest_neighbour.fit(dataset)
    
    distances,indices=nearest_neighbour.kneighbors(dataset)
    print("DISTANCES",(distances),"INDICES",(indices))
    
    save_feat(sys.argv[4],nearest_neighbour)
    
    

    
    #print(aggregated.values())
