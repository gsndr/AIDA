import pandas as pd
from math import log


class InfoGain():

    def __init__(self, path):
        self._path=path



    def extractVariables(self):
        self._df = pd.read_csv(self._path + ".csv");
        # put the original column names in a python list
        '''if 'Unnamed: 0' in self._df.columns:
            self._df = self._df.drop(columns=['Unnamed: 0']);
        if 'Unnamed: 0.1' in self._df.columns:
            self._df = self._df.drop(columns=['Unnamed: 0.1']);
            '''
        self._categories=list(self._df.columns.values)
        print(self._categories)
        self._totalRows=self._df.count()


    def splitCategories(self):
        self._dfNormal=self._df
    def entropy(pi):
        '''
            pi is an array that contain classifications
            return the Entropy of a probability distribution:
            entropy(p) = − SUM (Pi * log(Pi) )
            defintion:
                    entropy is a metric to measure the uncertainty of a probability distribution.
            entropy ranges between 0 to 1
            Low entropy means the distribution varies (peaks and valleys).
            High entropy means the distribution is uniform.
            See:
                    http://www.cs.csi.cuny.edu/~imberman/ai/Entropy%20and%20Information%20Gain.htm
            '''
        total = 0
        for p in pi:
            p = p / sum(pi)
            if p != 0:
                total += p * log(p, 2)
            else:
                total += 0
        total *= -1
        return total


    def gain(d, a):
        '''
        return the information gain:
        gain(D, A) = entropy(D)−􏰋 SUM ( |Di| / |D| * entropy(Di) )
        '''

        total = 0
        for v in a:
            total += sum(v) / sum(d) * InfoGain.entropy(v)

        gain = InfoGain.entropy(d) - total
        return gain
