import unittest


class EmdTest(unittest.TestCase):

    def test_aginst_pyemd(self):
        from cacgan.sinkhornOT import dummy_distance_matrix, np, torch, DEVICE, emdloss_detach
        from pyemd import emd
        from sklearn.preprocessing import normalize
        a = np.random.random((4, 6))
        b = np.random.random((4, 6))
        a = normalize(a, axis=1, norm="l1")
        b = normalize(b, axis=1, norm="l1")
        d = dummy_distance_matrix(a.shape[1])
        pyemd_results = []
        for i in range(a.shape[0]):
            pyemd_results.append(emd(a[i], b[i], d))

        ta = torch.tensor(a).to(DEVICE, dtype=torch.float32)
        tb = torch.tensor(b).to(DEVICE, dtype=torch.float32)
        td = torch.tensor(d).to(DEVICE, dtype=torch.float32)
        sinkhorn_results = emdloss_detach(ta, tb, td).cpu().detach().numpy()
        same = np.allclose(sinkhorn_results, np.array(pyemd_results))
        self.assertTrue(same)


if __name__ == '__main__':
    unittest.main()
