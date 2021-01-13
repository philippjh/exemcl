import itertools
import multiprocessing
import os
import random
import shutil
import unittest

import exemcl
import numpy as np
import pandas
from sklearn.datasets import make_blobs

DIM = 10
N = 500


def L(V, S):
    accu = 0
    for e in V:
        accu += min([pow(np.linalg.norm(e - v), 2.0) for v in S])
    return accu / len(V)


def F(V, S):
    zero = np.zeros(DIM)
    zeroVecValue = L(V, [zero])
    return zeroVecValue - L(V, np.append(S, [zero], axis=0))


def F_gain(V, S, e):
    return F(V, np.append(S, [e], axis=0)) - F(V, S)


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Clean test file dir.
        shutil.rmtree(os.path.join(os.path.dirname(os.path.realpath(__file__)), "testfiles"), ignore_errors=True)

        # Define tolerancies.
        cls.rtol = 1e-02
        cls.atol = 1e-05

    def test_ExemplarBasedClustering(self):
        # Generate ground set.
        V, _ = make_blobs(n_samples=N, centers=3, n_features=DIM, random_state=0)
        V_frame = pandas.DataFrame(V)

        # Generate subsets.
        subsets_df = pandas.DataFrame()
        S = list()
        for i in range(50):
            subset_df = V_frame.iloc[np.random.choice(range(N), random.randint(int(0.1 * N), int(0.5 * N)))]
            S.append(subset_df.values)
            subset_df["subset_idx"] = i
            subsets_df = subsets_df.append(subset_df)

        # Calculate f(s) for every s \in S
        pool = multiprocessing.Pool()
        f_values = pool.starmap(F, [(V, s) for s in S])
        pool.close()
        pool.join()

        # Calculate marginal gains.
        pool = multiprocessing.Pool()
        e = V[len(V) - 1]
        marginal_values = pool.starmap(F_gain, [(V, s, e) for s in S])
        pool.close()
        pool.join()

        # Compare to output.
        for precision, device in itertools.product(["fp16", "fp32", "fp64"], ["cpu", "gpu"]):
            # Skip FP16/CPU combination.
            if precision == "fp16" and device == "cpu":
                continue

            # Make comparison.
            exem = exemcl.ExemplarClustering(ground_set=V, precision=precision, device=device, worker_count=-1)
            f_values_exemcl = exem(S)
            marginal_values_exemcl = exem(S, e)
            self.assertTrue(np.allclose(np.asarray(f_values), np.asarray(f_values_exemcl), atol=self.atol, rtol=self.rtol))
            self.assertTrue(np.allclose(np.asarray(marginal_values), np.asarray(marginal_values_exemcl), atol=self.atol, rtol=self.rtol))

        # Write output files.
        out_dir = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(os.path.join(out_dir, "testfiles", "exem"))
        V_frame.to_csv(os.path.join(out_dir, "testfiles", "exem", "ground_set.csv"), index_label="ObjectID")
        subsets_df.reset_index(drop=True).to_csv(os.path.join(out_dir, "testfiles", "exem", "subsets.csv"), index_label="ObjectID")
        pandas.DataFrame(f_values).to_csv(os.path.join(out_dir, "testfiles", "exem", "f_values.csv"), index_label="ObjectID")
        pandas.DataFrame(marginal_values).to_csv(os.path.join(out_dir, "testfiles", "exem", "marginal_values.csv"), index_label="ObjectID")


if __name__ == '__main__':
    unittest.main()
