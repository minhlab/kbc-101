"""
Adopted from pylearn2.sandbox.nlp.datasets.penntree.PennTreebank

See: http://www.cis.upenn.edu/~treebank/
"""

__authors__ = "Minh Ngoc Le"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"


import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class KBCDataset(DenseDesignMatrix):
    """
    Loads the Penn Treebank corpus.

    Parameters
    ----------
    which_set : {'train', 'valid', 'test'}
        Choose the set to use
    context_len : int
        The size of the context i.e. the number of words used
        to predict the subsequent word.
    shuffle : bool
        Whether to shuffle the samples or go through the dataset
        linearly
    """
    def __init__(self, which_set, home_dir, max_labels):
        """
        Loads the data and turns it into n-grams
        """

        self.__dict__.update(locals())
        del self.self

        if which_set not in ('train', 'valid', 'test'):
            raise ValueError("Dataset must be one of 'train', 'valid' "
                             "or 'test', was %s instead" %str(which_set))

        path = ("%s/%s.npz" %(home_dir, which_set))
        with np.load(path) as f:
            self._data = f.iteritems().next()[1] # get the first array
            
        super(KBCDataset, self).__init__(
            X=self._data[:, :-1], X_labels=max_labels, 
            y=self._data[:, -1:], y_labels=2
        )