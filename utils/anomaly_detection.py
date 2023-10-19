import numpy as np
import tensorflow as tf
import os.path as osp

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as auc_fn
from scipy import interpolate
from matplotlib import pyplot as pp
from matplotlib.colors import Normalize

from config import FIG_DIR


class AnomalyDetector:

    def __init__(self):
        self.radiuses = None
        self.xmin = None
        self.xmax = None

    def predict(self):
        raise NotImplemented

    def _set_radiuses(self, data, metric='em', alpha=3, n_generated=1000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        #X_unif = np.random.uniform(np.zeros(n_features), np.ones(n_features), size=(n_generated, n_features))
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        #print(f'Alpha = {alpha}')
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def _em(self, volume_support, s_U, s_X, t_max=0.99, t_step=0.01):
        t = np.arange(0, 1 / volume_support, t_step / volume_support)
        EM_t = np.zeros(t.shape[0])
        n_samples = s_X.shape[0]
        n_generated = s_U.shape[0]
        s_X_unique = np.unique(s_X)
        EM_t[0] = 1.
        for u in s_X_unique:
            EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() - 1 / n_generated * (s_U > u).sum() * t * volume_support)
        amax = np.argmax(EM_t <= t_max) + 1
        if amax == 1:
            amax = -1
        auc = auc_fn(t[:amax], EM_t[:amax])
        return auc, EM_t, amax

    def _mv(self, volume_support, s_U, s_X, alpha_step=0.001, alpha_min=0.9, alpha_max=0.999):
        axis_alpha = np.arange(alpha_min, alpha_max, alpha_step * (alpha_max - alpha_min))
        n_samples = s_X.shape[0]
        n_generated = s_U.shape[0]
        s_X_argsort = s_X.argsort()
        mass = 0
        cpt = 0
        u = s_X[s_X_argsort[-1]]
        mv = np.zeros(axis_alpha.shape[0])
        for i in range(axis_alpha.shape[0]):
            while mass < axis_alpha[i]:
                cpt += 1
                u = s_X[s_X_argsort[-cpt]]
                mass = 1. / n_samples * cpt
            mv[i] = float((s_U >= u).sum()) / n_generated * volume_support
        return auc_fn(axis_alpha, mv), mv

    def evaluate(self, data, alpha, fpr_level):

        predictions, _, probs = self.predict(data[0], alpha)

        data1_01 = data[1].copy()
        data1_01[np.where(data1_01 > 0)] = 1

        if predictions is not None:

            #acc = len(np.where(predictions == data[1])[0]) / data[1].shape[0]

            fpr, tpr, thresholds = roc_curve(data1_01, probs)
            thresholds_lvl = [thr for thr, fpr_val in zip(thresholds, fpr) if fpr_val <= fpr_level]
            thresholds_uniq = np.unique(thresholds_lvl)
            acc_per_thr = np.zeros(len(thresholds_uniq))
            for i, threshold in enumerate(thresholds_uniq):
                preds_01 = np.zeros(len(probs))
                preds_01[np.where(probs > threshold)] = 1
                acc_per_thr[i] = float(len(np.where(preds_01 == data1_01)[0])) / len(probs)

            acc = np.max(acc_per_thr)

            fpr = len(np.where((predictions == 1) & (data[1] == 0))[0]) / (1e-10 + len(np.where(data[1] == 0)[0]))
            tpr = len(np.where((predictions == 1) & (data[1] == 1))[0]) / (1e-10 + len(np.where(data[1] == 1)[0]))
        else:
            acc, fpr, tpr = 0, 0, 0

        auc = roc_auc_score(data1_01, predictions)

        return acc, fpr, tpr, auc, probs


class CentroidClusteringAnomalyDetector(AnomalyDetector):

    def __init__(self):
        super(CentroidClusteringAnomalyDetector, self).__init__()

    def _calculate_distances(self, data, eps=1e-10):
        E_va_ = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_va = np.linalg.norm(E_va_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_va = np.argmin(D_va, axis=1)
        dists_va = np.min(D_va, axis=1)
        self.radiuses = np.zeros((C_.shape[0], 3))
        for k in range(C_.shape[0]):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                self.radiuses[k, 0] = np.mean(dists_va[idx])
                self.radiuses[k, 1] = np.std(dists_va[idx])
                self.radiuses[k, 2] = np.max(dists_va[idx])
            else:
                self.radiuses[k, 0] = 0
                self.radiuses[k, 1] = 0
                self.radiuses[k, 2] = 0

    def _set_radiuses(self, data, metric='em', alpha=3, n_generated=100000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        #X_unif = np.random.uniform(np.zeros(n_features), np.ones(n_features), size=(n_generated, n_features))
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        #print(f'Alpha = {alpha}')
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def predict(self, data, alpha, eps=1e-10, standardize=True):
        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        if standardize:
            E_te_ = (data - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        else:
            E_te_ = data
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_te = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_te = np.argmin(D_te, axis=1)
        dists_te = np.min(D_te, axis=1)
        nte = E_te_.shape[0]
        pred_thrs = radiuses[cl_labels_te]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + 1e-10)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def plot(self, data, fig_dir=FIG_DIR, labels=['Normal', 'Defective'], prefix=None, title=None, eps=1e-10, n_samples=2000):

        nc = self.centroids.shape[0]
        print('Number of centroids:', nc)

        u_labels = np.unique(data[1][:n_samples])
        for u_label in u_labels:
            print(u_label, len(np.where(data[1][:n_samples] == u_label)[0]))

        X_plot = np.vstack([
            (data[0][:n_samples, :] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
            (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
        ])

        #X_plot = np.vstack([
        #    data[0][:n_samples, :],
        #    self.centroids,
        #])

        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        X_tsne = tsne.fit_transform(X_plot)
        pp.style.use('default')
        pp.figure(figsize=(5, 3))
        colormap = 'jet'
        scatter_points = pp.scatter(X_tsne[:-nc, 0], X_tsne[:-nc, 1], c=data[1][:n_samples], s=5, cmap=colormap, marker='x', linewidth=0.5)
        scatter_centroids = pp.scatter(X_tsne[-nc:, 0], X_tsne[-nc:, 1], c=np.zeros(nc), s=self.weights / np.sum(self.weights) * 1000, cmap=colormap, edgecolor='black', linewidth=1.5);
        #pp.xlabel('t-SNE feature 1') #, fontsize=10)
        #pp.ylabel('t-SNE feature 2') #, fontsize=10)
        pp.axis('off')
        if title is not None:
            pp.title(title)
        pp.legend(
            scatter_points.legend_elements()[0] + scatter_centroids.legend_elements()[0],
            [label.capitalize() for label in labels] + ['Cluster centroids'],
            loc=0
        )
        fname = f'{prefix}_' if prefix is not None else ''
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(fig_dir, f'{fname}tsne.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()


class KM(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(KM, self).__init__()

    def fit(self, data, validation_data, hp=2, batch_size=16, l=4, n_iters=100, metric='em'):

        n_features = data[0].shape[1]

        # init min and max values, centroids, and their weights

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        ntr = data[1].shape[0]

        km = KMeans(n_clusters=hp)
        km.fit((data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + 1e-10))

        self.centroids = np.array(km.cluster_centers_ * (self.xmax[None, :] - self.xmin[None, :] + 1e-10) + self.xmin[None, :])

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class SKM(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(SKM, self).__init__()

    def _adjust_params(self, kb_max, inp_shape, n_clusters, batch_size, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        params = np.array([n_clusters, batch_size])
        params_scaled = params / params[0]
        x = param_size_total / (np.prod(inp_shape) * (params_scaled[1] + 2))
        if only_if_bigger:
            n_clusters_adjusted = np.minimum(n_clusters, int(np.round(x)))
            batch_size_adjusted = np.minimum(batch_size, int(np.round(x * params_scaled[1])))
        else:
            n_clusters_adjusted = int(np.floor(x))
            batch_size_adjusted = int(np.floor(x * params_scaled[1]))
        return n_clusters_adjusted, batch_size_adjusted

    def fit(self, data, validation_data, hp, mem_max, n_clusters=8, n_iters=100, metric='em', adjust_params=True):

        # init min and max values, centroids, and their weights

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        C, W = None, None

        # the main clustering loop

        ntr = data[1].shape[0]
        inp_shape = data[0].shape[1:]
        l = n_clusters

        if type(hp) == list:
            n_clusters = int(np.round(hp[0]))
            batch_size = int(np.round(hp[1] * n_clusters))
        else:
            batch_size = int(np.round(hp * n_clusters))

        n_clusters_, batch_size_ = self._adjust_params(mem_max, inp_shape, n_clusters, batch_size)
        if not adjust_params:
            assert n_clusters == n_clusters_ and batch_size == batch_size_
            #print('n clusters before:', n_clusters, 'batch size before:', batch_size)
            #n_clusters, batch_size = self._adjust_params(mem_max, inp_shape, n_clusters, batch_size)
            #print('n clusters after:', n_clusters, 'batch size after:', batch_size)
        n_clusters, batch_size = n_clusters_, batch_size_

        if n_clusters == 0 or batch_size == 0:
            raise NotImplemented

        batch_size = np.minimum(ntr, batch_size)

        #print(n_clusters, batch_size)

        for i in range(0, ntr - batch_size + 1, batch_size):

            #print(i, '/', ntr - batch_size + 1)

            # take a batch

            idx = np.arange(i, i + batch_size)
            B = data[0][idx, :]
            B_ = (B - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + 1e-10)

            # pick initial centroids

            if C is None:
                C = B[np.random.choice(range(B.shape[0]), n_clusters, replace=False), :]
                C_ = (C - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + 1e-10)
                D = np.sum((B_[:, None, :] - C_[None, :, :]) ** 2, axis=-1)
                #D = np.zeros((B.shape[0], C.shape[0]))
                #for j in range(B.shape[0]):
                #    for k in range(C.shape[0]):
                #        D[j, k] = np.sum(((B[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
                min_dist = np.zeros(D.shape)
                min_dist[range(D.shape[0]), np.argmin(D, axis=1)] = 1
                W = np.zeros(n_clusters)

            # select candidates

            C_ = (C - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + 1e-10)
            D = np.sum((B_[:, None, :] - C_[None, :, :]) ** 2, axis=-1)
            #D = np.zeros((B.shape[0], C.shape[0]))
            #for j in range(B.shape[0]):
            #    for k in range(C.shape[0]):
            #        D[j, k] = np.sum(((B[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
            cost = np.sum(np.min(D, axis=1))
            p = np.min(D, axis=1) / (cost + 1e-10)
            C = np.r_[C, B[np.random.choice(range(len(p)), np.minimum(l, len(p)), p=(p + 1e-10) / np.sum(p + 1e-10), replace=False), :]]

            # assign data to the centroids

            C_ = (C - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + 1e-10)
            D = np.sum((B_[:, None, :] - C_[None, :, :]) ** 2, axis=-1)
            #D = np.zeros((B.shape[0], C.shape[0]))
            #for j in range(B.shape[0]):
            #    for k in range(C.shape[0]):
            #        D[j, k] = np.sum(((B[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
            min_dist = np.zeros(D.shape)
            min_dist[range(D.shape[0]), np.argmin(D, axis=1)] = 1
            count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(C.shape[0])])
            for i in range(len(W)):
                count[i] += W[i]

            # weighted k-means clustering

            centroids = C[:n_clusters, :]
            centroids_ = (centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + 1e-10)

            for i in range(n_iters):

                D = np.sum((C_[:, None, :] - centroids_[None, :, :]) ** 2, axis=-1)
                #D = np.zeros((C.shape[0], centroids.shape[0]))
                #for j in range(C.shape[0]):
                #    for k in range(centroids.shape[0]):
                #        D[j, k] = np.sum(((C[j, :] - self.xmin) / (self.xmax - self.xmin + 1e-10) - (centroids[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)

                cl_labels = np.argmin(D, axis=1)

                centroids_new = []
                W_new = []

                for j in range(n_clusters):
                    idx = np.where(cl_labels == j)[0]
                    if len(idx) > 0:
                        centroids_new.append(np.sum(count[idx, None] * C[idx, :], axis=0) / (np.sum(count[idx] + 1e-10)))
                        W_new.append(np.sum(count[idx]))
                    else:
                        pass

                if np.array_equal(centroids, centroids_new):
                    break

                centroids = np.vstack(centroids_new)
                self.weights = np.hstack(W_new)

            self.centroids = np.array(centroids)

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class CS(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(CS, self).__init__()

    def _adjust_params(self, kb_max, inp_shape, n_clusters, n_micro_clusters, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        k = n_micro_clusters / n_clusters
        x = param_size_total / (2 * np.prod(inp_shape) * k + k + 2 * np.prod(inp_shape))
        n_clusters_adjusted = np.minimum(int(np.round(x)), n_clusters) if only_if_bigger else int(np.round(x))
        n_micro_clusters_adjusted = np.minimum(int(np.round(n_clusters_adjusted * k)), n_micro_clusters) if only_if_bigger else int(np.round(n_clusters_adjusted * k))
        return n_clusters_adjusted, n_micro_clusters_adjusted

    def fit(self, data, validation_data, hp, mem_max, n_clusters=8, micro_cluster_radius_alpha=3, n_iters=40, eps=1e-10, metric='em', adjust_params=True):

        ntr = data[0].shape[0]

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        C, R = [], []
        inp_shape = data[0].shape[1:]

        if type(hp) == list:
            n_clusters = int(np.round(hp[0]))
            n_micro_clusters = int(np.round(hp[1] * n_clusters))
        else:
            n_micro_clusters = int(np.round(hp * n_clusters))

        n_clusters_, n_micro_clusters_ = self._adjust_params(mem_max, inp_shape, n_clusters, n_micro_clusters)
        if not adjust_params:
            assert n_clusters == n_clusters_ and n_micro_clusters == n_micro_clusters_
            #print('n clusters before:', n_clusters, 'n micro clusters before:', n_micro_clusters)
            #n_clusters, n_micro_clusters = self._adjust_params(mem_max, inp_shape, n_clusters, n_micro_clusters)
            #print('n clusters after:', n_clusters, 'n micro clusters after:', n_micro_clusters)
        n_clusters, n_micro_clusters = n_clusters_, n_micro_clusters_

        if n_clusters == 0 or n_micro_clusters == 0:
            raise NotImplemented

        # the main clustering loop

        for xi in range(ntr):

            #print(f'{xi} / {ntr}')

            # take a sample

            x = data[0][xi, :].copy()

            # create initial micro-clsuter

            if len(C) < 2:
                C.append([1, x, x ** 2])
                R.append(0)

            # add sample to the existing framework

            else:

                # update the minimal distance between micro-clusters

                mc_centroids = np.vstack([item[1] / item[0] for item in C])
                mc_centroids_std = (mc_centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)

                D = np.sum((mc_centroids_std[:, None, :] - mc_centroids_std[None, :, :]) ** 2, axis=-1)
                D_triu = D[np.triu_indices(len(C), k=1)]
                #for i in range(len(C)):
                #    for j in range(len(C)):
                #        if i < j:
                #            D[i, j] = np.sqrt(np.sum(((C[i][1] / C[i][0] - self.xmin) / (self.xmax - self.xmin + eps) - (C[j][1] / C[j][0] - self.xmin) / (self.xmax - self.xmin + eps)) ** 2))
                #        else:
                #            D[i, j] = np.inf
                #cl_dist_min = np.min(D)
                cl_dist_min = np.min(D_triu)
                i_dmin, j_dmin = np.where(D == cl_dist_min)
                i_dmin = i_dmin[0]
                j_dmin = j_dmin[0]

                # update micro-cluster radiuses

                for i in range(len(C)):
                    if C[i][0] < 2:
                        R[i] = cl_dist_min
                    else:
                        ls_ = (C[i][1] - C[i][0] * self.xmin) / (self.xmax - self.xmin + eps)
                        ss_ = (C[i][2] - 2 * C[i][1] * self.xmin + C[i][0] * self.xmin ** 2) / ((self.xmax - self.xmin) ** 2 + eps)
                        R[i] = micro_cluster_radius_alpha * np.mean(np.sqrt(np.clip(ss_ / C[i][0] - (ls_ / C[i][0]) ** 2, 0, np.inf)))
                        if R[i] == 0:
                            R[i] = cl_dist_min

                # calculate distances from the sample to the micro-clusters

                #D = np.zeros(len(C))
                #for i in range(len(C)):
                #    D[i] = np.sqrt(np.sum(((x - self.xmin) / (self.xmax - self.xmin + 1e-10) - (C[i][1] / C[i][0] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2))
                x_std = (x - self.xmin) / (self.xmax - self.xmin + 1e-10)
                D = np.sum((mc_centroids_std - x_std[None, :]) ** 2, axis=-1)
                k = np.argmin(D)

                if D[k] <= R[k]:

                    # add sample to the existing micro-cluster

                    C[k][0] += 1
                    C[k][1] += x
                    C[k][2] += x ** 2

                else:

                    # merge the closest clusters

                    if len(C) >= n_micro_clusters:
                        C[i_dmin][0] += C[j_dmin][0]
                        C[i_dmin][1] += C[j_dmin][1]
                        C[i_dmin][2] += C[j_dmin][2]
                        C[j_dmin][0] = 1
                        C[j_dmin][1] = x
                        C[j_dmin][2] = x ** 2
                        # print(f'Micro-clusters {i_dmin} and {j_dmin} have been merged')

                    # create a new cluster

                    else:
                        C.append([1, x, x ** 2])
                        R.append(0)

        C = np.vstack([c[1] / c[0] for c in C])

        # weighted k-means clustering

        count = np.hstack([c[0] for c in C])
        centroids = C[np.random.choice(range(C.shape[0]), np.minimum(n_clusters, C.shape[0]), replace=False), :]

        for i in range(n_iters):

            D = np.zeros((C.shape[0], centroids.shape[0]))
            for j in range(C.shape[0]):
                for k in range(centroids.shape[0]):
                    D[j, k] = np.sum(((C[j, :] - self.xmin) / (self.xmax - self.xmin + eps) - (centroids[k, :] - self.xmin) / (self.xmax - self.xmin + 1e-10)) ** 2)
            cl_labels = np.argmin(D, axis=1)

            centroids_new = []
            W_new = []

            for j in range(n_clusters):
                idx = np.where(cl_labels == j)[0]
                if len(idx) > 0:
                    centroids_new.append(np.sum(count[idx, None] * C[idx, :], axis=0) / (np.sum(count[idx] + eps)))
                    W_new.append(np.sum(count[idx]))
                else:
                    pass

            if np.array_equal(centroids, centroids_new):
                break

            centroids = np.vstack(centroids_new)
            self.weights = np.hstack(W_new)

        self.centroids = np.array(centroids)

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class WAP(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(WAP, self).__init__()

    def _find_exemplars(self, exemplers, exempler_stats, reservoir, preference_alpha=1, n=200, lmbd=0.5, same_labels_thr=5):

        def calculate_r(a, s, k):
            return s[k] - np.max(np.hstack([s[:k], s[k + 1:]]) + np.hstack([a[:k], a[k + 1:]]))

        def calculate_a(r, k, i):
            vec = np.maximum(np.hstack([r[:i], r[i + 1:]]), 0)
            if k == i:
                result = np.sum(vec)
            else:
                result = min(0, r[k] + np.sum(vec))
            return result

        def update_R(X, A, S, R):
            for i in range(X.shape[0]):
                for k in range(X.shape[0]):
                    R[i, k] = calculate_r(A[i, :], S[i, :], k)
            return R

        def update_A(X, R, A):
            for i in range(X.shape[0]):
                for k in range(X.shape[0]):
                    A[i, k] = calculate_a(R[:, k], k, i)
            return A

        if len(reservoir) > 0:
            data = np.vstack([exemplers, reservoir])
        else:
            data = exemplers

        n_exemplers = exemplers.shape[0]

        n_anomalies = len(reservoir)
        D = np.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=-1)
        exempler_counts = [item[0] for item in exempler_stats]
        exempler_distortations = [item[1] for item in exempler_stats]
        S = np.zeros((n_exemplers + n_anomalies, n_exemplers + n_anomalies))
        S[: n_exemplers, :] = -D[: n_exemplers, :] * np.array(exempler_counts)[:, None]
        S[n_exemplers :, :] = -D[n_exemplers :, :]
        preference = np.median(S) * preference_alpha
        #S[np.arange(len(data)), np.arange(len(data))] = np.min(S)
        S[np.arange(n_exemplers), np.arange(n_exemplers)] = preference + np.array(exempler_distortations)
        S[n_exemplers + np.arange(n_anomalies), n_exemplers + np.arange(n_anomalies)] = preference

        A = np.zeros((data.shape[0], data.shape[0]))
        R = np.zeros((data.shape[0], data.shape[0]))

        same_labels_count = 0

        for i in range(n):

            R_ = R.copy()
            A_ = A.copy()
            labels_old = np.argmax(A + R, 1)

            R = update_R(data, A, S, R)
            A = update_A(data, R, A)

            if np.all(A == A_) and np.all(R == R_):
                break

            if i != 0:
                R = lmbd * R_ + (1 - lmbd) * R
                A = lmbd * A_ + (1 - lmbd) * A

            labels = np.argmax(A + R, 1)

            if np.all(labels == labels_old):
                same_labels_count += 1
            else:
                same_labels_count == 0

            if same_labels_count >= same_labels_thr:
                #print('Break at', i)
                break

        exemplar_ids = np.unique(labels)

        return data[exemplar_ids, :]

    def _adjust_params(self, kb_max, inp_shape, n_clusters, batch_size, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        map_size = np.array([n_clusters, batch_size])
        map_size_scaled = map_size / map_size[0]
        a = np.prod(inp_shape)
        k = map_size_scaled[1]
        s = 2 * (k + 1) ** 2
        x = np.sqrt((4 * s * param_size_total + a ** 2 * (k + 1) ** 2) / (4 * s ** 2)) - (a * (k + 1)) / (2 * s)
        # map_size_adjusted = (int(np.floor(x)), int(np.floor(x * map_size_scaled[1])))
        if only_if_bigger:
            map_size_adjusted = [
                np.minimum(map_size[0], int(np.floor(x))),
                np.minimum(map_size[1], int(np.floor(x * map_size_scaled[1])))
            ]
        else:
            map_size_adjusted = [int(np.floor(x)), int(np.floor(x * map_size_scaled[1]))]
        return map_size_adjusted[0], map_size_adjusted[1]

    def fit(self, data, validation_data, hp=None, mem_max=None, std=True, n_clusters=8, eps=1e-10, cluster_distortion_alpha=3, metric='em', adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        if std:
            X = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        else:
            X = data[0]

        reservoir = []
        inp_shape = data[0].shape[1:]

        if type(hp) == list:
            n_clusters = int(np.round(hp[0]))
            reservoir_thr = int(np.round(hp[1] * n_clusters))
        else:
            reservoir_thr = int(np.round(hp * n_clusters))

        n_clusters_, reservoir_thr_ = self._adjust_params(mem_max, inp_shape, n_clusters, reservoir_thr)
        if not adjust_params:
            assert n_clusters == n_clusters_ and reservoir_thr == reservoir_thr_
            #print('n clusters before:', n_clusters, 'reservoir_thr before:', reservoir_thr)
            #n_clusters, reservoir_thr = self._adjust_params(mem_max, inp_shape, n_clusters, reservoir_thr)
            #print('n clusters after:', n_clusters, 'reservoir_thr after:', reservoir_thr)
        n_clusters, reservoir_thr = n_clusters_, reservoir_thr_

        # first AP

        batch = X[:reservoir_thr, :]

        #af = AffinityPropagation()
        #af_ = af.fit(batch)
        #C = af_.cluster_centers_

        Cs = [[1, 0] for _ in range(reservoir_thr)]
        preference_alpha = 1
        n_clusters_ = np.inf
        C = self._find_exemplars(batch, Cs, reservoir, preference_alpha)
        n_clusters_ = C.shape[0]
        #print('n clusters init', n_clusters_)

        for i in range(10):
            C = self._find_exemplars(batch, Cs, reservoir, preference_alpha)
            n_clusters_ = C.shape[0]
            if n_clusters_ <= n_clusters:
                break
            preference_alpha *= 2

        if n_clusters_ > n_clusters:
            raise NotImplemented

        #D = np.zeros((batch.shape[0], n_clusters))
        #for j in range(batch.shape[0]):
        #    for k in range(n_clusters):
        #        D[j, k] = np.sum((batch[j, :] - C[k, :]) ** 2)
        D = np.sum((batch[:, None, :] - C[None, :, :]) ** 2, axis=-1) + eps

        cl_labels = np.argmin(D, axis=1)
        Dmin = np.min(D, axis=1)
        Cs = []
        Ctrue = []
        for i in range(n_clusters_):
            idx = np.where(cl_labels == i)[0]
            if len(idx) > 0:
                Cs.append([len(idx), np.sum(Dmin[idx])])
                Ctrue.append(C[i, :])
        C = np.vstack(Ctrue)

        # calculate distortion threshold

        distortion_thr = cluster_distortion_alpha * np.mean([item[1] / item[0] for item in Cs])

        # the main clustering loop

        ntr = data[1].shape[0] - reservoir_thr
        for xi in range(ntr):

            # take a sample

            x = X[xi + reservoir_thr, :].copy()

            # calculate distances from the sample to the micro-clusters

            D = np.zeros(len(C))
            for i in range(len(C)):
                D[i] = np.sum((x - C[i, :]) ** 2)
            k = np.argmin(D)

            #print(D[k], distortion_thr)

            if D[k] <= distortion_thr:

                #print('Add to existing')

                # add sample to the existing micro-cluster

                Cs[k][0] += 1
                Cs[k][1] += D[k]

            else:

                #print('Add to reservoir')

                reservoir.append(x)

            # check whether do we need an update

            current_distortion = np.max([item[1] / item[0] for item in Cs])

            #print('current distortion:', current_distortion)

            if len(reservoir) >= reservoir_thr or current_distortion >= distortion_thr:

                #print('recalculating')

                #C_new = self._find_exemplars(C, Cs, reservoir)
                #n_clusters = C_new.shape[0]

                preference_alpha = 1
                for i in range(10):
                    C_new = self._find_exemplars(C, Cs, reservoir, preference_alpha)
                    n_clusters_ = C_new.shape[0]
                    if n_clusters_ <= n_clusters:
                        break
                    preference_alpha *= 2

                if n_clusters_ > n_clusters:
                    raise NotImplemented

                if len(reservoir) > 0:
                    batch = np.vstack([C, reservoir])
                else:
                    batch = C.copy()
                #D = np.zeros((batch.shape[0], n_clusters))
                #for j in range(batch.shape[0]):
                #    for k in range(n_clusters):
                #        D[j, k] = np.sum((batch[j, :] - C_new[k, :]) ** 2)
                D = np.sum((batch[:, None, :] - C_new[None, :, :]) ** 2, axis=-1) + eps

                cl_labels = np.argmin(D, axis=1)
                counts = np.hstack([[item[0] for item in Cs], np.ones(len(reservoir))])
                distortions = np.hstack([[item[1] for item in Cs], np.zeros(len(reservoir))])
                Cs = []
                Ctrue = []
                for i in range(n_clusters_):
                    idx = np.where(cl_labels == i)[0]
                    if len(idx) > 0:
                        Cs.append([np.sum(counts[idx]), np.sum(distortions[idx])])
                        Ctrue.append(C_new[i, :])
                C_new = np.vstack(Ctrue)

                distortion_thr = cluster_distortion_alpha * np.mean([item[1] / item[0] for item in Cs])
                reservoir = []
                C = C_new.copy()

        #print('N clusters:', C.shape[0])

        W = [item[0] for item in Cs]

        self.weights = np.hstack(W)
        self.centroids = np.array(C) * (self.xmax[None, :] - self.xmin[None, :] + eps) + self.xmin[None, :]

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class WCM(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(WCM, self).__init__()

    def _calculate_distances(self, data, m=2, eps=1e-10):
        E_va_ = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_va = np.linalg.norm(E_va_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_va = np.argmin(D_va, axis=1)
        dists_va = np.min(D_va, axis=1)

        #D = np.sqrt(np.sum((E_va_[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)) + eps
        #D = (D ** (2 / (m - 1)))
        #U = 1 / (D * np.sum(1 / D, 0))
        #R = np.dot(U ** m, self.centroids) / (np.dot(U ** m, np.ones((self.centroids.shape[0], self.centroids.shape[1]))))
        #dists_va = np.sqrt(np.sum((E_va_ - R) ** 2, axis=1))

        self.radiuses = np.zeros((C_.shape[0], 3))
        for k in range(C_.shape[0]):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                self.radiuses[k, 0] = np.mean(dists_va[idx])
                self.radiuses[k, 1] = np.std(dists_va[idx])
                self.radiuses[k, 2] = np.max(dists_va[idx])
            else:
                self.radiuses[k, 0] = 0
                self.radiuses[k, 1] = 0
                self.radiuses[k, 2] = 0

    def predict(self, data, alpha, m=2, eps=1e-10, standardize=True):
        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        if standardize:
            E_te_ = (data - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        else:
            E_te_ = data
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_te = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_te = np.argmin(D_te, axis=1)
        dists_te = np.min(D_te, axis=1)

        #D = np.sqrt(np.sum((E_te_[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)) + eps
        #D = (D ** (2 / (m - 1)))
        #U = 1 / (D * np.sum(1 / D, 0))
        #R = np.dot(U ** m, self.centroids) / (np.dot(U ** m, np.ones((self.centroids.shape[0], self.centroids.shape[1]))))
        #dists_te = np.sqrt(np.sum((E_te_ - R) ** 2, axis=1))

        nte = E_te_.shape[0]
        pred_thrs = radiuses[cl_labels_te]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + 1e-10)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _get_centroids(self, new_points, centroids=None, centroid_weigths=None, m=2, n_centroids=4, n_iter=100, tol=1e-5, eps=1e-10):
        n_points = new_points.shape[0]
        n_features = new_points.shape[1]
        if centroids is None:
            U = np.random.rand(n_centroids, n_points)
            U = U / np.sum(U, 0)
            X = new_points
            W = np.ones(n_points)
        else:
            X = np.vstack([centroids, new_points])
            D = np.sqrt(np.sum((centroids[:, None, :] - X[None, :, :]) ** 2, axis=-1)) + eps
            D = (D ** (2 / (m - 1)))
            U = 1 / (D * np.sum(1 / D, 0))
            W = np.hstack([centroid_weigths, np.ones(n_points)])

        #if centroids is not None:
        #    print('w',W)
        #    print('u', U)
        #    print('x',X)

        for i in range(n_iter):
            U_old = U.copy()
            V = np.dot(W[None, :] * (U ** m), X) / (np.dot(W[None, :] * (U ** m), np.ones((X.shape[0], n_features))))
            #V = np.dot(W[:, None] * (U ** m), X) / np.dot(W[:, None] * (U ** m), np.ones((n_points, n_features)))
            D = np.sqrt(np.sum((V[:, None, :] - X[None, :, :]) ** 2, axis=-1)) + eps
            D = (D ** (2 / (m - 1)))
            U = 1 / (D * np.sum(1 / D, 0))
            if np.linalg.norm(U - U_old) < tol:
                #print('break at', i)
                break
            else:
                pass
                #print(np.linalg.norm(U - U_old))

        #W = np.sum(W[:, None] * U, 1)
        #print(U.shape, np.sum(U, 0), W.shape)
        W = np.sum(W[None, :] * U, 1)

        return V, W

    def _adjust_params(self, kb_max, inp_shape, n_clusters, batch_size, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        map_size = np.array([n_clusters, batch_size])
        map_size_scaled = map_size / map_size[0]
        a = np.prod(inp_shape)
        k = map_size_scaled[1]
        s = 2 * (k + 1)**2
        x = np.sqrt((4 * s * param_size_total + a**2 * (k + 1)**2) / (4 * s**2)) - (a * (k + 1)) / (2 * s)
        #map_size_adjusted = (int(np.floor(x)), int(np.floor(x * map_size_scaled[1])))
        if only_if_bigger:
            map_size_adjusted = [
                np.minimum(map_size[0], int(np.floor(x))),
                np.minimum(map_size[1], int(np.floor(x * map_size_scaled[1])))
            ]
        else:
            map_size_adjusted = [int(np.floor(x)), int(np.floor(x * map_size_scaled[1]))]
        return map_size_adjusted[0], map_size_adjusted[1]

    def fit(self, data, validation_data, hp=None, mem_max=None, std=True, n_clusters=4, eps=1e-10, metric='em', adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        if std:
            X = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        else:
            X = data[0]

        inp_shape = data[0].shape[1:]

        #batch_size = n_clusters * hp
        if type(hp) == list:
            n_clusters = int(np.round(hp[0]))
            batch_size = int(np.round(hp[1] * n_clusters))
        else:
            batch_size = int(np.round(hp * n_clusters))

        n_clusters_, batch_size_ = self._adjust_params(mem_max, inp_shape, n_clusters, batch_size)
        if not adjust_params:
            assert n_clusters_ == n_clusters and batch_size_ == batch_size
            #print('n clusters before:', n_clusters, 'batch size before:', batch_size)
            #n_clusters, batch_size = self._adjust_params(mem_max, inp_shape, n_clusters, batch_size)
            #print('n clusters after:', n_clusters, 'batch size after:', batch_size)
        n_clusters, batch_size = n_clusters_, batch_size_

        # first AP

        batch = X[:batch_size, :]

        C, W = self._get_centroids(batch, n_centroids=n_clusters)

        # the main clustering loop

        ntr = (data[1].shape[0] - batch_size) // batch_size
        for xi in range(ntr):
            batch = X[(xi + 1) * batch_size : (xi + 2) * batch_size, :].copy()
            C, W = self._get_centroids(new_points=batch, centroids=C, centroid_weigths=W, n_centroids=n_clusters)

        self.weights = np.hstack(W)
        self.centroids = np.array(C) * (self.xmax[None, :] - self.xmin[None, :] + eps) + self.xmin[None, :]

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class GNG(CentroidClusteringAnomalyDetector):

    def __init__(self):
        super(GNG, self).__init__()

    def _calculate_distances(self, data, eb=0.1, en=0.02, eps=1e-10):
        E_va_ = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_va = np.linalg.norm(E_va_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_va = np.argmin(D_va, axis=1)
        dists_va = eb * np.min(D_va, axis=1)
        for i in range(len(dists_va)):
            bmu_1_idx = cl_labels_va[i]
            for e in self.edges:
                if e[0] == bmu_1_idx:
                    dists_va[i] += en * D_va[i, e[1]]
                elif e[1] == bmu_1_idx:
                    dists_va[i] += en * D_va[i, e[0]]

        self.radiuses = np.zeros((C_.shape[0], 3))
        for k in range(C_.shape[0]):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                self.radiuses[k, 0] = np.mean(dists_va[idx])
                self.radiuses[k, 1] = np.std(dists_va[idx])
                self.radiuses[k, 2] = np.max(dists_va[idx])
            else:
                self.radiuses[k, 0] = 0
                self.radiuses[k, 1] = 0
                self.radiuses[k, 2] = 0

    def predict(self, data, alpha, eb=0.1, en=0.02, eps=1e-10, standardize=True):
        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        if standardize:
            E_te_ = (data - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        else:
            E_te_ = data
        C_ = (self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        D_te = np.linalg.norm(E_te_[:, None, :] - C_[None, :, :], axis=-1)
        cl_labels_te = np.argmin(D_te, axis=1)
        dists_te = eb * np.min(D_te, axis=1)
        for i in range(len(dists_te)):
            bmu_1_idx = cl_labels_te[i]
            for e in self.edges:
                if e[0] == bmu_1_idx:
                    dists_te[i] += en * D_te[i, e[1]]
                elif e[1] == bmu_1_idx:
                    dists_te[i] += en * D_te[i, e[0]]
        nte = E_te_.shape[0]
        pred_thrs = radiuses[cl_labels_te]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + 1e-10)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _adjust_params(self, kb_max, inp_shape, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        a = np.prod(inp_shape)
        x = np.sqrt(2 * param_size_total + ((2 * a - 3) ** 2) / 36) - (2 * a - 3) / 6
        n_nodes_thr = int(np.floor(x))
        return n_nodes_thr

    def fit(self, data, validation_data, hp=None, mem_max=None, std=True, eb=0.1, en=0.02, a_max=5, lmb=100, alpha=0.5, d=0.0005, eps=1e-10, metric='em', adjust_params=True):

        if hp is not None:
            a_max = hp

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]

        if std:
            X = (data[0] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps)
        else:
            X = data[0]

        if adjust_params:
            n_nodes_thr = self._adjust_params(mem_max, inp_shape)
            #print('n nodes thr:', n_nodes_thr)

        # first edge

        batch = X[:2, :]
        N = batch
        D = [0, 0]
        W = [1, 1]
        E = [[0, 1, 0]]

        # the main clustering loop

        ntr = (data[1].shape[0] - 2)
        for xi in range(ntr):
            x = X[xi + 2, :].copy()

            dists = np.sum((N - x[None, :]) ** 2, axis=-1)

            idx_sorted = np.argsort(dists)
            bmu_1_idx = idx_sorted[-1]
            bmu_2_idx = idx_sorted[-2]

            bmu_1 = N[bmu_1_idx, :]
            bmu_2 = N[bmu_2_idx, :]

            n_list = []
            bmu_1_bmu_2_idx = None
            for i, e in enumerate(E):
                if e[0] == bmu_1_idx:
                    e[2] += 1
                    n_list.append(e[1])
                    if bmu_1_bmu_2_idx is None and e[1] == bmu_2_idx:
                        bmu_1_bmu_2_idx = i
                elif e[1] == bmu_1_idx:
                    e[2] += 1
                    n_list.append(e[0])
                    if bmu_1_bmu_2_idx is None and e[0] == bmu_2_idx:
                        bmu_1_bmu_2_idx = i

            D[bmu_1_idx] += dists[bmu_1_idx]
            W[bmu_1_idx] += 1
            N[bmu_1_idx, :] = N[bmu_1_idx, :] + eb * (x - N[bmu_1_idx, :])

            for i in n_list:
                N[i, :] = N[i, :] + en * (x - N[i, :])

            if bmu_1_bmu_2_idx is None:
                E.append([bmu_1_idx, bmu_2_idx, 0])
            else:
                E[bmu_1_bmu_2_idx][2] = 0

            E_new = []
            N_idx_list = []
            for e in E:
                if e[2] < a_max:
                    E_new.append(e)
                    if e[0] not in N_idx_list:
                        N_idx_list.append(e[0])
                    if e[1] not in N_idx_list:
                        N_idx_list.append(e[1])
                else:
                    #print('removed edge:', e)
                    pass

            E = E_new.copy()

            potentially_removed_nodes = [i for i in np.arange(len(D)) if i not in N_idx_list]
            if len(potentially_removed_nodes) > 0:
                #print('removed nodes:', potentially_removed_nodes)
                nodes_to_remove = np.unique(potentially_removed_nodes)
                for i, e in enumerate(E):
                    idx = np.where(nodes_to_remove < e[0])[0]
                    E[i][0] -= len(idx)
                    idx = np.where(nodes_to_remove < e[1])[0]
                    E[i][1] -= len(idx)

            N = N[N_idx_list, :]
            W = [W[i] for i in N_idx_list]
            D = [D[i] for i in N_idx_list]

            # insert a new node

            if xi > 0 and xi % lmb == 0 and N.shape[0] < n_nodes_thr:
                i1 = np.argmax(D)
                n_list = []
                for i, e in enumerate(E):
                    if e[0] == i1:
                        n_list.append(e[1])
                    elif e[1] == i1:
                        n_list.append(e[0])
                i2 = n_list[np.argmax([D[j] for j in n_list])]

                for i, e in enumerate(E):
                    if (e[0] == i1 and e[1] == i2) or (e[1] == i1 and e[0] == i2):
                        break

                del E[i]

                new_n = 0.5 * (N[i1, :] + N[i2, :])
                D[i1] *= alpha
                D[i2] *= alpha
                new_d = D[i1]

                N = np.vstack([N, new_n.reshape(1, -1)])
                D.append(new_d)
                W.append(0.5 * (W[i1] + W[i2]))
                i = len(D) - 1
                E.append([i1, i, 0])
                E.append([i2, i, 0])

                #print('e:',E)
                #print('d:', D)

            #print(xi, len(N), len(D), len(E))

            D = [x * (1 - d) for x in D]

        #print('n nodes:', N.shape[0])

        self.weights = np.hstack(W)
        self.centroids = np.array(N) * (self.xmax[None, :] - self.xmin[None, :] + eps) + self.xmin[None, :]
        self.edges = E.copy()

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class Svdd(tf.keras.models.Model):

    def __init__(self, preprocessor, nu=1e-3):
        super(Svdd, self).__init__()
        self.nu = nu
        self.preprocessor = preprocessor
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.built = False

    def build(self, input_shape, X):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))

        #self.c = tf.reduce_mean(self.preprocessor(X), 0)
        self.c = self.add_weight(shape=[self.preprocessor.output_shape[1]], initializer='glorot_uniform', name='c', trainable=True)
        #self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R', trainable=False)
        self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R', trainable=True)
        self.built = True

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.preprocessor(x)
        dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        scores = dists - self.R ** 2
        return scores

    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            x = self.preprocessor(inputs)
            dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
            scores = dists - self.R ** 2
            penalty = tf.maximum(scores, tf.zeros_like(scores))
            loss = self.R ** 2 + (1 / self.nu) * penalty

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        R_new = tf.sort(tf.math.sqrt(dists))[tf.cast((1 - self.nu) * tf.math.reduce_sum(tf.ones_like(dists)), tf.int32)]
        self.R.assign(R_new)

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):
        if len(data) == 2:
            inputs, outputs = data
        else:
            inputs, outputs = data[0]
        x = self.preprocessor(inputs)
        dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        scores = dists - self.R ** 2
        penalty = tf.maximum(scores, tf.zeros_like(scores))
        loss = self.R ** 2 + (1 / self.nu) * penalty
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }


class DSVDD(AnomalyDetector):

    def __init__(self):
        super(DSVDD, self).__init__()

    def _calculate_distances(self, data, eps=1e-10):
        preds = self.model.predict(data[0], verbose=False)
        p = np.clip(preds, 0, np.inf)
        self.radiuses = np.array([np.mean(p), np.std(p), np.max(p)]).reshape(1, -1)

    def plot(self, data, fig_dir=FIG_DIR, labels=['Normal', 'Defective'], prefix=None, title=None, eps=1e-10, n_samples=2000):

        nc = self.centroids.shape[0]
        print('Number of centroids:', nc)

        u_labels = np.unique(data[1][:n_samples])
        for u_label in u_labels:
            print(u_label, len(np.where(data[1][:n_samples] == u_label)[0]))

        X_mapped = self.model.preprocessor(data[0][:n_samples, :])

        print(X_mapped.shape)

        X_plot = np.vstack([
            X_mapped,
            #(self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
            self.centroids,
        ])

        #X_plot = np.vstack([
        #    data[0][:n_samples, :],
        #    self.centroids,
        #])

        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        X_tsne = tsne.fit_transform(X_plot)
        pp.style.use('default')
        pp.figure(figsize=(5, 3))
        colormap = 'jet'
        scatter_points = pp.scatter(X_tsne[:-nc, 0], X_tsne[:-nc, 1], c=data[1][:n_samples], s=5, cmap=colormap, marker='x', linewidth=0.5)
        scatter_centroids = pp.scatter(X_tsne[-nc:, 0], X_tsne[-nc:, 1], c=np.zeros(nc), s=self.weights / np.sum(self.weights) * 1000, cmap=colormap, edgecolor='black', linewidth=1.5);
        #pp.xlabel('t-SNE feature 1') #, fontsize=10)
        #pp.ylabel('t-SNE feature 2') #, fontsize=10)
        pp.axis('off')
        if title is not None:
            pp.title(title)
        pp.legend(
            scatter_points.legend_elements()[0] + scatter_centroids.legend_elements()[0],
            [label.capitalize() for label in labels] + ['Centre of the model'],
            loc=0
        )
        fname = f'{prefix}_' if prefix is not None else ''
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(fig_dir, f'{fname}tsne.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)
        dists_te = self.model.predict(E_te_, verbose=False)
        nte = E_te_.shape[0]
        pred_thrs = radiuses[0]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _set_radiuses(self, data, metric='em', alpha=3, n_generated=100000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        #print(f'Alpha = {alpha}')
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def _adjust_params(self, kb_max, inp_shape, layers, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        layers_scaled = np.array(layers) / layers[0]
        s = np.dot(layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape)
        x = np.sqrt((a ** 2 + 4 * s * param_size_total) / (4 * (s ** 2))) - a / (2 * s)
        layers_adjusted = []
        for i, scale in enumerate(layers_scaled):
            if only_if_bigger:
                n_units = np.minimum(np.floor(x * scale), layers[i])
            else:
                n_units = np.floor(x * scale)
            layers_adjusted.append(n_units)
        return layers_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, epochs=1,
            batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            if len(hp) == 2:
                n_layers = int(np.round(hp[0]))
                n_units = int(np.round(hp[1]))
                n_outputs = np.prod(inp_shape)
            elif len(hp) == 3:
                n_layers = int(np.round(hp[0]))
                n_units = int(np.round(hp[1]))
                n_outputs = int(np.round(hp[2]))
            else:
                raise NotImplemented
        else:
            n_layers = 1
            n_units = int(np.round(hp))
            n_outputs = np.prod(inp_shape)

        encoder_units = [n_units for _ in range(n_layers)]
        encoder_units.append(n_outputs)

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)
        self.encoder = tf.keras.models.Model(inputs, encoded)

        self.model = Svdd(preprocessor=self.encoder)
        self.model.build(input_shape=(None, *inp_shape), X=data[0])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self.centroids = self.model.c.numpy().reshape(1, -1)
        self.weights = [data[0].shape[0]]

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class SomLayer(tf.keras.layers.Layer):

    def __init__(self, map_size, prototypes=None, **kwargs):
        if 'input_shape' not in kwargs and 'latent_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('latent_dim'),)
        super(SomLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.nprototypes = np.prod(map_size)
        self.initial_prototypes = prototypes
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))
        self.prototypes = self.add_weight(shape=(self.nprototypes, *input_dims), initializer='glorot_uniform', name='prototypes')
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes.reshape(1, self.nprototypes, *input_dims))
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        #d = tf.reduce_mean(tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=-1), axis=-1)
        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=-1)
        return d

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.nprototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super(SomLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def som_loss(weights, distances):
    return tf.reduce_mean(tf.reduce_sum(weights * distances, axis=1))


def ae_loss(inputs, outputs):
    return tf.reduce_mean(tf.reduce_sum((inputs - outputs) ** 2, axis=1))


class Som(tf.keras.models.Model):

    def __init__(self, encoder, decoder=None, map_size=(8, 8), T_min=0.1, T_max=10.0, niterations=10000, nnn=1, ae_loss_weight=0.0, prototypes=None):
        super(Som, self).__init__()
        self.map_size = map_size
        self.nprototypes = np.prod(map_size)
        ranges = [np.arange(m) for m in map_size]
        mg = np.meshgrid(*ranges, indexing='ij')
        self.prototype_coordinates = tf.convert_to_tensor(np.array([item.flatten() for item in mg]).T)
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else tf.keras.layers.Lambda(lambda x: x)
        self.som_layer = SomLayer(map_size, name='som_layer', prototypes=prototypes)
        self.T_min = T_min
        self.T_max = T_max
        self.niterations = niterations
        self.current_iteration = 0
        self.total_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.nnn = nnn
        self.ae_loss_weight = ae_loss_weight

    @property
    def prototypes(self):
        return self.som_layer.get_weights()[0]

    def call(self, x):
        x = self.encoder(x)
        x = self.som_layer(x)

        y_pred = tf.math.argmin(x, axis=1)
        weights = self.neighborhood_function(self.map_dist(y_pred), self.T_max)

        #s = tf.sort(x, axis=1)
        #spl = tf.split(s, [self.nnn, self.nprototypes - self.nnn], axis=1)
        #return tf.reduce_mean(spl[0], axis=1)
        #return x

        return y_pred, tf.reduce_sum(weights * x, axis=1)

    def map_dist(self, y_pred):
        labels = tf.gather(self.prototype_coordinates, y_pred)
        mh = tf.reduce_sum(tf.math.abs(tf.expand_dims(labels, 1) - tf.expand_dims(self.prototype_coordinates, 0)), axis=-1)
        return tf.cast(mh, tf.float32)

    def neighborhood_function(self, d, T):
        return tf.math.exp(-(d ** 2) / (T ** 2))

    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:

            # Compute cluster assignments for batches

            encoded = self.encoder(inputs)
            d = self.som_layer(encoded)
            decoded = self.decoder(encoded)
            y_pred = tf.math.argmin(d, axis=1)

            # Update temperature parameter

            self.current_iteration += 1
            if self.current_iteration > self.niterations:
                self.current_iteration = self.niterations
            self.T = self.T_max * (self.T_min / self.T_max) ** (self.current_iteration / (self.niterations - 1))

            # Compute topographic weights batches

            w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)

            # calculate loss

            loss = som_loss(w_batch, d) + self.ae_loss_weight * ae_loss(inputs, decoded)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {
            "loss": self.total_loss_tracker.result()
        }

    def test_step(self, data):
        inputs, outputs = data
        encoded = self.encoder(inputs)
        d = self.som_layer(encoded)
        decoded = self.decoder(encoded)
        y_pred = tf.math.argmin(d, axis=1)
        w_batch = self.neighborhood_function(self.map_dist(y_pred), self.T)

        loss = som_loss(w_batch, d) + self.ae_loss_weight * ae_loss(inputs, decoded)

        self.total_loss_tracker.update_state(loss)
        return {
            "loss": self.total_loss_tracker.result()
        }


class SOM(AnomalyDetector):

    def __init__(self):
        super(SOM, self).__init__()

    def _calculate_distances(self, data):

        #preds = self.model.predict(data[0])
        #cl_labels_va = np.argmin(preds, axis=1)
        #dists_va = np.min(preds, axis=1)
        #n_clusters = preds.shape[1]
        n_clusters = self.model.nprototypes

        cl_labels_va, dists_va = self.model.predict(data[0], verbose=False)

        self.radiuses = np.zeros((n_clusters, 3))
        for k in range(n_clusters):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                self.radiuses[k, 0] = np.mean(dists_va[idx])
                self.radiuses[k, 1] = np.std(dists_va[idx])
                self.radiuses[k, 2] = np.max(dists_va[idx])
            else:
                self.radiuses[k, 0] = 0
                self.radiuses[k, 1] = 0
                self.radiuses[k, 2] = 0

    def plot(self, data, fig_dir=FIG_DIR, labels=['Normal', 'Defective'], prefix=None, title=None, eps=1e-10, n_samples=2000):

        nc = self.centroids.shape[0]
        print('Number of centroids:', nc)

        u_labels = np.unique(data[1][:n_samples])
        for u_label in u_labels:
            print(u_label, len(np.where(data[1][:n_samples] == u_label)[0]))

        X_plot = np.vstack([
            (data[0][:n_samples, :] - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
            #(self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
            self.centroids,
        ])

        #X_plot = np.vstack([
        #    data[0][:n_samples, :],
        #    self.centroids,
        #])

        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        X_tsne = tsne.fit_transform(X_plot)
        pp.style.use('default')
        pp.figure(figsize=(5, 3))
        colormap = 'jet'
        scatter_points = pp.scatter(X_tsne[:-nc, 0], X_tsne[:-nc, 1], c=data[1][:n_samples], s=5, cmap=colormap, marker='x', linewidth=0.5)
        scatter_centroids = pp.scatter(X_tsne[-nc:, 0], X_tsne[-nc:, 1], c=np.zeros(nc), s=self.weights / np.sum(self.weights) * 1000, cmap=colormap, edgecolor='black', linewidth=1.5);
        #pp.xlabel('t-SNE feature 1') #, fontsize=10)
        #pp.ylabel('t-SNE feature 2') #, fontsize=10)
        pp.axis('off')
        if title is not None:
            pp.title(title)
        pp.legend(
            scatter_points.legend_elements()[0] + scatter_centroids.legend_elements()[0],
            [label.capitalize() for label in labels] + ['Prototype vectors'],
            loc=0
        )
        fname = f'{prefix}_' if prefix is not None else ''
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(fig_dir, f'{fname}tsne.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

        self.plot_(data, fig_dir, prefix=prefix, title=title)

    def plot_(self, data, fig_dir, prefix=None, title=None):

        map_size = self.model.map_size
        print(map_size)
        som_layer = [layer for layer in self.model.layers if 'SomLayer' in layer.__class__.__name__]
        assert len(som_layer) == 1
        som_layer = som_layer[0]
        prototype_weights = np.array(som_layer.weights)[0]
        prototype_coors = np.array(self.model.prototype_coordinates)

        u_matrix = np.zeros(map_size)
        p_matrix = np.zeros(map_size)
        ustar_matrix = np.zeros(map_size)

        X = np.array(data[0], dtype=np.float32)
        D_x = som_layer(X)
        D_p = np.sqrt(np.sum((prototype_weights[:, None, :] - prototype_weights[None, :, :]) ** 2, axis=-1))

        print(prototype_weights.shape, D_p.shape)

        labels = prototype_coors[np.argmin(D_x, 1), :]
        mh_x = np.sum(np.abs(labels[:, None] - prototype_coors[None, :]), axis=-1)

        mh_p = np.sum(np.abs(prototype_coors[:, None] - prototype_coors[None, :]), axis=-1)

        mh_thr = 1
        r = np.median(D_x)

        for k in range(prototype_coors.shape[0]):

            i, j = prototype_coors[k, :]
            n_idx = np.array(np.where(mh_p[k, :] <= mh_thr)[0], dtype=int)
            d_p = np.array(D_p[k, :])
            u_matrix[i, j] = np.mean(d_p[n_idx])

            d_x = np.array(D_x[:, k])
            idx = np.where(d_x < r)[0]
            p_matrix[i, j] = len(idx)

        for k in range(prototype_coors.shape[0]):

            i, j = prototype_coors[k, :]
            idx = np.where(p_matrix > p_matrix[i, j])[0]
            ustar_matrix[i, j] = u_matrix[i, j] * len(idx) / np.prod(map_size)

        #matrix = p_matrix
        matrix = ustar_matrix

        print('Matrix shape:', matrix.shape)

        pp.style.use('default')
        fig_size = (5, 3)

        fig, ax = pp.subplots()

        x = np.arange(map_size[0])
        xmin = np.min(x)
        xmax = np.max(x)

        y = np.arange(map_size[1])
        ymin = np.min(y)
        ymax = np.max(y)

        fig.set_size_inches(*fig_size)

        xi, yi = np.linspace(xmin, xmax, xmax), np.linspace(ymin, ymax, ymax)
        xi, yi = np.meshgrid(xi, yi)

        rbf = interpolate.Rbf(prototype_coors[:, 0], prototype_coors[:, 1], matrix.reshape(1, -1), function='linear')
        zi = rbf(xi, yi)

        normalize = Normalize(vmin=np.min(zi), vmax=np.max(zi))

        pp.scatter(xi, yi, c=zi, cmap='Wistia', norm=normalize, marker='s')

        #pp.scatter(xi, yi, c=zi, cmap='seismic', norm=normalize)

        normal_idx = np.where(data[1] == 0)[0]
        anomaly_idx = np.where(data[1] > 0)[0]

        pp.scatter(labels[:, 0], labels[:, 1], c=data[1], marker='o', cmap='jet')
        #pp.scatter(labels[normal_idx, 0], labels[normal_idx, 1], c='blue', marker='o')
        #pp.scatter(labels[anomaly_idx, 0], labels[anomaly_idx, 1], c='red', marker='o')

        pp.xlim([xmin, xmax])
        pp.ylim([ymin, ymax])

        pp.axis('off')

        #pp.colorbar()

        #scatter_points = pp.scatter(X_tsne[:-nc, 0], X_tsne[:-nc, 1], c=data[1][:n_samples], s=5, cmap=colormap, marker='x', linewidth=0.5)
        #scatter_centroids = pp.scatter(X_tsne[-nc:, 0], X_tsne[-nc:, 1], c=np.zeros(nc), s=self.weights / np.sum(self.weights) * 1000, cmap=colormap, edgecolor='black', linewidth=1.5);
        # pp.xlabel('t-SNE feature 1') #, fontsize=10)
        # pp.ylabel('t-SNE feature 2') #, fontsize=10)

        pp.axis('off')
        if title is not None:
            pp.title(title)

        #pp.legend(
        #    scatter_points.legend_elements()[0] + scatter_centroids.legend_elements()[0],
        #    [label.capitalize() for label in labels] + ['Cluster centroids'],
        #    loc=0
        #)

        fname = f'{prefix}_' if prefix is not None else ''
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(fig_dir, f'{fname}map.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)

        #D_te = self.model(E_te_).numpy()
        #dists_te = np.min(D_te, axis=1)
        #cl_labels_te = np.argmin(D_te, axis=1)

        cl_labels_te, dists_te = self.model.predict(E_te_, verbose=False)
        #cl_labels_te = cl_labels_te.numpy()
        #dists_te = dists_te.numpy()

        nte = E_te_.shape[0]
        #pred_thrs = radiuses[0]
        pred_thrs = radiuses[cl_labels_te]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def __set_radiuses(self, data, metric='em', alpha=3, n_generated=100000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def _adjust_params(self, kb_max, inp_shape, map_size, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        map_size = np.array(map_size)
        map_size_scaled = map_size / map_size[0]
        x = np.sqrt(param_size_total / (np.prod(inp_shape) * map_size_scaled[1]))
        if only_if_bigger:
            map_size_adjusted = [
                np.minimum(map_size[0], int(np.floor(x))),
                np.minimum(map_size[1], int(np.floor(x * map_size_scaled[1])))
            ]
        else:
            map_size_adjusted = [int(np.floor(x)), int(np.floor(x * map_size_scaled[1]))]
        return map_size_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, epochs=1,
            batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            map_size0 = int(np.round(hp[0]))
            map_size1 = int(np.round(hp[1]))
        else:
            map_size0 = int(np.round(hp))
            map_size1 = int(np.round(hp))

        map_size = [map_size0, map_size1]

        map_size_ = self._adjust_params(mem_max, inp_shape, map_size)
        if not adjust_params:
            assert map_size_ == map_size
            #print('map size before:', map_size)
            #map_size = self._adjust_params(mem_max, inp_shape, map_size)
            #print('map size after:', map_size)
        map_size = map_size_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            encoded = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            encoded = inputs

        self.encoder = tf.keras.models.Model(inputs, encoded)

        n_prototypes = np.prod(map_size)
        initial_prototypes = self.encoder(data[0][np.random.choice(data[0].shape[0], n_prototypes, replace=False), :]).numpy()

        self.model = Som(encoder=self.encoder, map_size=map_size, nnn=1, T_max=1, ae_loss_weight=0, niterations=data[0].shape[0], prototypes=initial_prototypes)
        self.model.build(input_shape=(None, *inp_shape))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        som_layer = [layer for layer in self.model.layers if 'SomLayer' in layer.__class__.__name__]
        assert len(som_layer) == 1
        som_layer = som_layer[0]
        self.centroids = np.array(som_layer.weights)[0]

        X = np.array(data[0], dtype=np.float32)
        bmu_idx = np.argmin(som_layer(X), 1)
        self.weights = np.zeros(len(self.centroids))
        for i in range(len(self.centroids)):
            self.weights[i] = len(np.where(bmu_idx == i)[0])

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class DSOM(SOM):

    def __init__(self):
        super(DSOM, self).__init__()

    def _adjust_params(self, kb_max, inp_shape, layers, map_size, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        layers_and_map = layers + [map_size[0] * map_size[1]]
        layers_scaled = np.array(layers_and_map) / layers_and_map[0]
        coeff = np.hstack([np.ones(len(layers) - 1) * 2, 1])
        s = np.dot(coeff * layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape) * 2
        x = np.sqrt((a ** 2 + s * param_size_total)) / s - a / s
        layers_adjusted = []

        for i, scale in enumerate(layers_scaled[:-1]):
            if only_if_bigger:
                n_units = np.minimum(int(np.floor(x * scale)), layers[i])
            else:
                n_units = int(np.floor(x * scale))
            layers_adjusted.append(n_units)
        if only_if_bigger:
            map_w = np.minimum(np.floor(x * layers_scaled[-1]), map_size[0] * map_size[1])
        else:
            map_w = np.floor(x * layers_scaled[-1])
        map_w_adjusted = int(np.floor(np.sqrt(map_w)))
        map_h_adjusted = int(np.floor(map_w / map_w_adjusted))
        return layers_adjusted, [map_w_adjusted, map_h_adjusted]

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, ae_loss_weight=0.001, map_size=[8, 8], encoder_units=[32],
            dropout=0.0, epochs=1, batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            encoder_units = [int(np.round(hp[0]))]
            map_size = [int(np.round(hp[1])), int(np.round(hp[1]))]
        else:
            map_size = [int(np.round(hp)), int(np.round(hp))]

        encoder_units_, map_size_ = self._adjust_params(mem_max, inp_shape, encoder_units, map_size)
        if not adjust_params:
            assert encoder_units_ == encoder_units and map_size_ == map_size
            #print('encoder units before:', encoder_units, 'map size before:', map_size)
            #encoder_units, map_size = self._adjust_params(mem_max, inp_shape, encoder_units, map_size)
            #print('encoder units after:', encoder_units, 'map size after:', map_size)
        encoder_units, map_size = encoder_units_, map_size_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)

        hidden = encoded

        for units in encoder_units[::-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        decoded = tf.keras.layers.Dense(units=np.prod(inp_shape))(hidden)

        self.encoder = tf.keras.models.Model(inputs, encoded)
        self.decoder = tf.keras.models.Model(encoded, decoded)

        self.model = Som(encoder=self.encoder, decoder=self.decoder, nnn=1, ae_loss_weight=ae_loss_weight, map_size=map_size, niterations=data[0].shape[0])
        self.model.build(input_shape=(None, *inp_shape))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class UAE(AnomalyDetector):

    def __init__(self):
        super(UAE, self).__init__()

    def _calculate_distances(self, data):
        preds = self.model.predict(data[0], verbose=False)
        dists_va = np.sqrt(np.sum((preds - data[0])**2, axis=1))
        self.radiuses = np.zeros((1, 3))
        self.radiuses[0, 0] = np.mean(dists_va)
        self.radiuses[0, 1] = np.std(dists_va)
        self.radiuses[0, 2] = np.max(dists_va)

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)
        D_te = self.model.predict(E_te_, verbose=False)
        dists_te = np.sqrt(np.sum((D_te - E_te_) ** 2, axis=1))
        nte = E_te_.shape[0]
        pred_thrs = radiuses[0]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _adjust_params(self, kb_max, inp_shape, layers, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param / 2
        layers_scaled = np.array(layers) / layers[0]
        s = np.dot(layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape)
        x = np.sqrt((a ** 2 + 4 * s * param_size_total) / (4 * (s ** 2))) - a / (2 * s)
        layers_adjusted = []
        for scale in layers_scaled:
            layers_adjusted.append(np.floor(x * scale))
        return layers_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, encoder_units=[64, 32, 16],
            dropout=0.0, epochs=1, batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            encoder_units = [int(np.round(hp[1])) for _ in range(int(np.round(hp[0])))]
        else:
            encoder_units = [int(np.round(hp[0])) for _ in range(2)]

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)
        hidden = encoded

        for units in encoder_units[:-1][::-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        outputs = tf.keras.layers.Dense(units=np.prod(inp_shape))(hidden)

        self.model = tf.keras.models.Model(inputs, outputs)
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(UAE):

    def __init__(self):
        super(VAE, self).__init__()

    def _adjust_params(self, kb_max, inp_shape, layers, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        layers_scaled = np.array(layers) / layers[0]
        coeff = np.ones(len(layers) - 1) * 2
        coeff[-1] = 3
        s = np.dot(coeff * layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape) * 2
        x = np.sqrt((a ** 2 + 4 * s * param_size_total) / (4 * (s ** 2))) - a / (2 * s)
        layers_adjusted = []
        for scale in layers_scaled:
            layers_adjusted.append(np.floor(x * scale))
        return layers_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, encoder_units=[64, 32, 16],
            dropout=0.0, epochs=1, batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            encoder_units = [int(np.round(hp[1])) for _ in range(int(np.round(hp[0])))]
        else:
            encoder_units = [int(np.round(hp[0])) for _ in range(2)]

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        z_mean = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)
        z_log_var = tf.keras.layers.Dense(units=encoder_units[-1], kernel_initializer=tf.keras.initializers.Zeros(), bias_initializer=tf.keras.initializers.Zeros())(hidden)

        hidden = Sampling()((z_mean, z_log_var))

        for units in encoder_units[:-1][::-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        outputs = tf.keras.layers.Dense(units=np.prod(inp_shape))(hidden)

        self.model = tf.keras.models.Model(inputs, outputs)
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class SAE(UAE):

    def __init__(self):
        super(SAE, self).__init__()

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, encoder_units=[64, 32, 16],
            dropout=0.0, epochs=1, batch_size=1, lr=3e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            encoder_units = [int(np.round(hp[1])) for _ in range(int(np.round(hp[0])))]
        else:
            encoder_units = [int(np.round(hp[0])) for _ in range(2)]

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1], activity_regularizer = tf.keras.regularizers.L1(0.0001))(hidden)
        hidden = encoded

        for units in encoder_units[:1][::-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        outputs = tf.keras.layers.Dense(units=np.prod(inp_shape))(hidden)

        self.model = tf.keras.models.Model(inputs, outputs)
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class DAE(UAE):

    def __init__(self):
        super(DAE, self).__init__()

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, encoder_units=[64, 32, 16], level_of_noise=0.05,
            dropout=0.0, epochs=1, batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            encoder_units = [int(np.round(hp[1])) for _ in range(int(np.round(hp[0])))]
        else:
            encoder_units = [int(np.round(hp[0])) for _ in range(2)]

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        hidden = tf.keras.layers.GaussianNoise(stddev=level_of_noise)(hidden)

        #X_train_noisy = data[0] + level_of_noise * np.random.normal(loc=0.0, scale=1.0, size=data[0].shape)
        #X_train_noisy = np.clip(X_train_noisy, 0., 1.)

        #X_validation_noisy = validation_data[0] + level_of_noise * np.random.normal(loc=0.0, scale=1.0, size=validation_data[0].shape)
        #X_validation_noisy = np.clip(X_validation_noisy, 0., 1.)

        for units in encoder_units[:1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1], activity_regularizer = tf.keras.regularizers.L1(0.0001))(hidden)
        hidden = encoded

        for units in encoder_units[:1][::-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        outputs = tf.keras.layers.Dense(units=np.prod(inp_shape))(hidden)

        self.model = tf.keras.models.Model(inputs, outputs)
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class Bgn(tf.keras.models.Model):

    def __init__(self, generator, encoder, discriminator):
        super(Bgn, self).__init__()

        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator

        self.generator_trainable_variables = []
        for layer in self.generator.layers:
            self.generator_trainable_variables.extend(layer.trainable_variables)

        self.encoder_trainable_variables = []
        for layer in self.encoder.layers:
            self.encoder_trainable_variables.extend(layer.trainable_variables)

        self.discriminator_trainable_variables = []
        for layer in self.discriminator.layers:
            self.discriminator_trainable_variables.extend(layer.trainable_variables)

        # loss trackers

        self.g_loss_tracker = tf.keras.metrics.Mean(name='g_loss')
        self.d_loss_tracker = tf.keras.metrics.Mean(name='d_loss')

    def build(self, input_shape):
        self.discriminator.build(input_shape)
        self.built = True

    def call(self, x):
        z = self.encoder(x)
        x = self.discriminator([z, x])
        score = 1 - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(x), logits=x)
        return score[:, 0]

    def train_step(self, data):

        x_real, z_fake = data
        x_fake = self.generator(z_fake)

        z_fake_enc = self.encoder(x_fake)
        z_real_enc = self.encoder(x_real)

        z = tf.concat([z_fake_enc, z_real_enc], axis=0)
        x = tf.concat([x_fake, x_real], axis=0)
        d_preds = self.discriminator([z, x])
        pred_g, pred_e = tf.split(d_preds, num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(tf.nn.softplus(pred_g)) + tf.reduce_mean(tf.nn.softplus(-pred_e))
        g_loss = tf.reduce_mean(tf.nn.softplus(-pred_g))

        d_gradients = tf.gradients(d_loss, self.discriminator_trainable_variables + self.encoder_trainable_variables)
        e_gradients_d = tf.gradients(d_loss, self.encoder_trainable_variables)

        self.optimizer.apply_gradients(zip(d_gradients, self.discriminator_trainable_variables + self.encoder_trainable_variables))
        #self.optimizer.apply_gradients(zip(e_gradients_d, self.encoder_trainable_variables))

        g_gradients = tf.gradients(g_loss, self.generator_trainable_variables + self.encoder_trainable_variables)
        e_gradients_g = tf.gradients(g_loss, self.encoder_trainable_variables)

        self.optimizer.apply_gradients(zip(g_gradients, self.generator_trainable_variables + self.encoder_trainable_variables))
        #self.optimizer.apply_gradients(zip(e_gradients_g, self.encoder_trainable_variables))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
        }

    def test_step(self, data):
        x_real, z_fake = data
        x_fake = self.generator(z_fake)
        z_real = self.encoder(x_real)
        z = tf.concat([z_fake, z_real], axis=0)
        x = tf.concat([x_fake, x_real], axis=0)
        d_preds = self.discriminator([z, x])
        pred_g, pred_e = tf.split(d_preds, num_or_size_splits=2, axis=0)

        d_loss = tf.reduce_mean(tf.nn.softplus(pred_g)) + tf.reduce_mean(tf.nn.softplus(-pred_e))
        g_loss = tf.reduce_mean(tf.nn.softplus(-pred_g))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
        }


class BGAN(AnomalyDetector):

    def __init__(self):
        super(BGAN, self).__init__()

    def _calculate_distances(self, data, eps=1e-10):
        preds = self.model.predict(data[0])
        p = np.clip(preds, 0, np.inf)
        self.radiuses = np.array([np.mean(p), np.std(p), np.max(p)]).reshape(1, -1)

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)
        dists_te = self.model(E_te_).numpy()
        nte = E_te_.shape[0]
        pred_thrs = radiuses[0]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _set_radiuses(self, data, metric='em', alpha=3, n_generated=100000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        print(f'Alpha = {alpha}')
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, latent_dim=8, generator_units=[32, 32],
            encoder_units=[32, 32], discriminator_units=[32, 32], epochs=1, batch_size=1, lr=3e-6, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]

        gen_inputs = tf.keras.layers.Input(shape=latent_dim)
        hidden = gen_inputs
        for units in generator_units:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
        gen_outputs = tf.keras.layers.Dense(units=np.prod(inp_shape), activation='sigmoid')(hidden)
        self.generator = tf.keras.models.Model(gen_inputs, gen_outputs)

        enc_inputs = tf.keras.layers.Input(shape=inp_shape)
        if std:
            # hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (enc_inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = enc_inputs
        for units in encoder_units:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
        enc_outptus = tf.keras.layers.Dense(units=latent_dim, activation='linear')(hidden)
        self.encoder = tf.keras.models.Model(enc_inputs, enc_outptus)

        dis_z_inputs = tf.keras.layers.Input(shape=(latent_dim,))
        dis_x_inputs = tf.keras.layers.Input(shape=inp_shape)

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            dis_x_inputs = (dis_x_inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            dis_x_inputs = dis_x_inputs
        hidden = tf.concat([dis_z_inputs, dis_x_inputs], axis=1)
        for units in discriminator_units:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
        dis_outptus = tf.keras.layers.Dense(units=1, activation='sigmoid')(hidden)
        self.discriminator = tf.keras.models.Model([dis_z_inputs, dis_x_inputs], dis_outptus)

        self.model = Bgn(generator=self.generator, encoder=self.encoder, discriminator=self.discriminator)
        self.model.build(input_shape=(None, *inp_shape))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        z_train = np.random.uniform(0, 1, (data[0].shape[0], latent_dim))
        z_validatation = np.random.uniform(0, 1, (validation_data[0].shape[0], latent_dim))

        self.model.fit(
            data[0], z_train,
            validation_data=(validation_data[0], z_validatation),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class WcmLayer(tf.keras.layers.Layer):

    def __init__(self, n_clusters, m=2, centroids=None, **kwargs):
        super(WcmLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.m = m
        self.initial_centroids = centroids
        self.centroids = None
        self.built = False

    def build(self, input_shape):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))
        self.centroids = self.add_weight(shape=(self.n_clusters, *input_dims), initializer='glorot_uniform', name='centroids')
        self.centroid_weights = self.add_weight(shape=(self.n_clusters,), initializer='glorot_uniform', name='centroid_weights')
        if self.initial_centroids is not None:
            self.set_weights(self.initial_centroids)
            del self.initial_centroids
        self.built = True

    def call_(self, inputs, eps=1e-10, **kwargs):
        X = tf.concat([self.centroids, inputs], axis=0)
        d_new = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.centroids), axis=-1) + eps
        D = tf.concat([tf.eye(self.n_clusters) + eps, tf.transpose(d_new)], axis=1)
        D = (D ** (2 / (self.m - 1)))
        U = 1 / (D * tf.reduce_sum(1 / D, axis=0))
        W = tf.concat([self.centroid_weights, tf.ones(1)], axis=0)
        V = tf.matmul(tf.expand_dims(W, 0) * (U ** self.m), X) / tf.matmul(tf.expand_dims(W, 0) * (U ** self.m), tf.ones((self.n_clusters + 1, inputs.shape[1])))
        W = tf.reduce_sum(tf.expand_dims(W, 0) * U, axis=1)
        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.centroids), axis=-1)
        #d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - V), axis=-1)
        d = (d ** (2 / (self.m - 1)))
        u = 1 / (d * tf.reduce_sum(1 / d, axis=0))
        r = tf.matmul(u ** self.m, self.centroids) / (tf.matmul(u ** self.m, tf.ones((self.n_clusters, inputs.shape[1]))))
        #r = tf.matmul(u ** self.m, V) / (tf.matmul(u ** self.m, tf.ones((self.n_clusters, inputs.shape[1]))))
        return d, r, V, W

    def call(self, inputs, eps=1e-10, **kwargs):

        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.centroids), axis=-1)
        d = (d ** (2 / (self.m - 1)))
        u = 1 / (d * tf.reduce_sum(1 / d, axis=0))

        J = tf.reduce_sum(u ** self.m * d)

        r = tf.matmul(u ** self.m, self.centroids) / (tf.matmul(u ** self.m, tf.ones((self.n_clusters, inputs.shape[1]))))

        return d, r, J

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(WcmLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Wcm(tf.keras.models.Model):

    def __init__(self, encoder, decoder, n_clusters=8, m=2):
        super(Wcm, self).__init__()
        self.latent_dim = encoder.output_shape[1]
        self.n_clusters = n_clusters
        self.m = m
        self.W = tf.ones(n_clusters)
        self.C = tf.convert_to_tensor(np.random.rand(n_clusters, self.latent_dim), dtype=tf.float32)
        self.encoder = encoder
        self.decoder = decoder
        self.wcm_layer = WcmLayer(n_clusters, m=self.m, name='wcm_layer')
        self.total_loss_tracker = tf.keras.metrics.Mean(name='loss')

    def call(self, x):
        x = self.encoder(x)
        #d, r, V, W = self.wcm_layer(x)
        d, r, J = self.wcm_layer(x)
        y_pred = tf.math.argmin(d, axis=1)
        dists = tf.reduce_min(d, axis=1)
        return y_pred, dists

    def _update_centroids(self, new_points, eps=1e-10):

        n_points = new_points.shape[0]
        n_features = new_points.shape[1]

        #X = tf.concat([self.wcm_layer.weights[0], new_points], axis=0)
        X = tf.concat([self.C, new_points], axis=0)
        d_new = tf.reduce_sum(tf.square(tf.expand_dims(new_points, axis=1) - self.C), axis=-1)
        D = tf.concat([tf.eye(self.n_clusters), tf.transpose(d_new)], axis=1)
        D = (D ** (2 / (self.m - 1)))
        U = 1 / (D * tf.reduce_sum(1 / D, axis=0))
        #W = tf.concat([self.w, tf.ones(n_points)], axis=0)
        W = tf.concat([self.W, tf.ones(n_points)], axis=0)
        self.C = tf.matmul(tf.expand_dims(W, 0) * (U ** self.m), X) / tf.matmul(tf.expand_dims(W, 0) * (U ** self.m), tf.ones((self.n_clusters + n_points, n_features)))
        self.W = tf.reduce_sum(tf.expand_dims(W, 0) * U, axis=1)

        #for i in range(n_iter):
            #U_old = U.copy()
            #V = np.dot(W[None, :] * (U ** m), X) / (np.dot(W[None, :] * (U ** m), np.ones((X.shape[0], n_features))))

            # V = np.dot(W[:, None] * (U ** m), X) / np.dot(W[:, None] * (U ** m), np.ones((n_points, n_features)))

            #D = np.sqrt(np.sum((V[:, None, :] - X[None, :, :]) ** 2, axis=-1)) + eps
            #D = (D ** (2 / (m - 1)))
            #U = 1 / (D * np.sum(1 / D, 0))
            #if np.linalg.norm(U - U_old) < tol:
                # print('break at', i)
            #    break
            #else:
            #    pass
                # print(np.linalg.norm(U - U_old))

        # W = np.sum(W[:, None] * U, 1)
        # print(U.shape, np.sum(U, 0), W.shape)

    def train_step(self, data, n_iter=10):
        inputs, outputs = data

        with tf.GradientTape() as tape:

            # Compute cluster assignments for batches

            encoded = self.encoder(inputs)
            #d, r, V, W = self.wcm_layer(encoded)
            d, r, J = self.wcm_layer(encoded)

            #self._update_centroids(encoded)

            decoded = self.decoder(r)

            # calculate loss

            #loss = ae_loss(inputs, decoded)
            loss = J

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)
        return {
            "loss": self.total_loss_tracker.result()
        }

    def test_step(self, data, m=2):
        inputs, outputs = data
        encoded = self.encoder(inputs)
        #d, r, V, W = self.wcm_layer(encoded)
        d, r, J = self.wcm_layer(encoded)
        decoded = self.decoder(r)

        #loss = ae_loss(inputs, decoded)
        loss = J

        self.total_loss_tracker.update_state(loss)
        return {
            "loss": self.total_loss_tracker.result()
        }


class DWCM(AnomalyDetector):

    def __init__(self):
        super(DWCM, self).__init__()

    def _calculate_distances(self, data):

        #preds = self.model.predict(data[0])
        #cl_labels_va = np.argmin(preds, axis=1)
        #dists_va = np.min(preds, axis=1)
        #n_clusters = preds.shape[1]
        n_clusters = self.model.n_clusters

        cl_labels_va, dists_va = self.model.predict(data[0])

        self.radiuses = np.zeros((n_clusters, 3))
        for k in range(n_clusters):
            idx = np.where(cl_labels_va == k)[0]
            if len(idx) > 0:
                self.radiuses[k, 0] = np.mean(dists_va[idx])
                self.radiuses[k, 1] = np.std(dists_va[idx])
                self.radiuses[k, 2] = np.max(dists_va[idx])
            else:
                self.radiuses[k, 0] = 0
                self.radiuses[k, 1] = 0
                self.radiuses[k, 2] = 0

    def _adjust_params(self, kb_max, inp_shape, layers, n_clusters, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        layers_and_map = layers + [n_clusters]
        layers_scaled = np.array(layers_and_map) / layers_and_map[0]
        coeff = np.hstack([np.ones(len(layers) - 1) * 2, 1])
        s = np.dot(coeff * layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape) * 2
        x = np.sqrt((a ** 2 + s * param_size_total)) / s - a / s
        layers_adjusted = []
        for scale in layers_scaled[:-1]:
            layers_adjusted.append(int(np.floor(x * scale)))
        n_clusters_adjusted = int(np.floor(x * layers_scaled[-1]))
        return layers_adjusted, n_clusters_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, n_clusters=8, encoder_units=[32, 16],
            dropout=0.0, epochs=1, batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        encoder_units_, map_size_ = self._adjust_params(mem_max, inp_shape, encoder_units, n_clusters)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units, 'n clusters before:', n_clusters)
            #encoder_units, map_size = self._adjust_params(mem_max, inp_shape, encoder_units, n_clusters)
            #print('encoder units after:', encoder_units, 'n clusters after:', n_clusters)
        encoder_units = encoder_units_

        if std:
            # hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)

        hidden = encoded

        for units in encoder_units[:-1][::-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            if dropout is not None and dropout > 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

        decoded = tf.keras.layers.Dense(units=np.prod(inp_shape))(hidden)

        self.encoder = tf.keras.models.Model(inputs, encoded)
        self.decoder = tf.keras.models.Model(encoded, decoded)

        self.model = Wcm(encoder=self.encoder, decoder=self.decoder, n_clusters=n_clusters)
        self.model.build(input_shape=(None, *inp_shape))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)

        #D_te = self.model(E_te_).numpy()
        #dists_te = np.min(D_te, axis=1)
        #cl_labels_te = np.argmin(D_te, axis=1)

        cl_labels_te, dists_te = self.model.predict(E_te_)
        #cl_labels_te = cl_labels_te.numpy()
        #dists_te = dists_te.numpy()

        nte = E_te_.shape[0]
        #pred_thrs = radiuses[0]
        pred_thrs = radiuses[cl_labels_te]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs


class Em(tf.keras.models.Model):

    def __init__(self, preprocessor, nu=10, nt=100):
        super(Em, self).__init__()
        self.preprocessor = preprocessor
        self.nu = nu
        self.nt = nt
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.built = False

    def build(self, input_shape, X):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))
        self.c = self.add_weight(shape=[self.preprocessor.output_shape[1], 1], initializer='glorot_uniform', name='c', trainable=True)
        self.built = True

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.preprocessor(x)
        scores = tf.sigmoid(tf.matmul(x, self.c))
        return scores[:, 0]

    def train_step(self, data):

        inputs, randoms = data
        randoms = tf.concat(tf.split(randoms, self.nu, axis=1), axis=0)

        with tf.GradientTape() as tape:
            s_x = self.call(inputs)
            s_u = self.call(randoms)
            loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(s_u * tf.expand_dims(tf.range(0, 1, 1/self.nt), 1) - s_x, 0.0), axis=0))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):

        if len(data) == 2:
            inputs, randoms = data
        else:
            inputs, randoms = data[0]
        randoms = tf.concat(tf.split(randoms, self.nu, axis=1), axis=0)

        s_x = self.call(inputs)
        s_u = self.call(randoms)
        #loss = tf.reduce_mean(tf.maximum(s_u - s_x, 0.0))
        loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(s_u * tf.expand_dims(tf.range(0, 1, 1/self.nt), 1) - s_x, 0.0), axis=1))
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result()
        }


class DEM(AnomalyDetector):

    def __init__(self):
        super(DEM, self).__init__()

    def _calculate_distances(self, data, eps=1e-10):
        preds = self.model.predict(data[0], verbose=False)
        p = np.clip(1 - preds, 0, 1)
        self.radiuses = np.array([np.mean(p), np.std(p), np.max(p)]).reshape(1, -1)

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)
        dists_te = 1 - self.model.predict(E_te_, verbose=False)
        nte = E_te_.shape[0]
        pred_thrs = radiuses[0]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _set_radiuses(self, data, metric='em', alpha=3, n_generated=100000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        #print(f'Alpha = {alpha}')
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def _adjust_params(self, kb_max, inp_shape, layers, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        layers_scaled = np.array(layers) / layers[0]
        s = np.dot(layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape)
        x = np.sqrt((a ** 2 + 4 * s * param_size_total) / (4 * (s ** 2))) - a / (2 * s)
        layers_adjusted = []
        for i, scale in enumerate(layers_scaled):
            if only_if_bigger:
                n_units = np.minimum(np.floor(x * scale), layers[i])
            else:
                n_units = np.floor(x * scale)
            layers_adjusted.append(n_units)
        return layers_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, epochs=1, nu=100,
            batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            n_layers = int(np.round(hp[0]))
            n_units = int(np.round(hp[1]))
        else:
            n_layers = 2
            n_units = int(np.round(hp))

        encoder_units = [n_units for _ in range(n_layers)]

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)
        self.encoder = tf.keras.models.Model(inputs, encoded)

        self.model = Em(preprocessor=self.encoder, nu=nu)
        self.model.build(input_shape=(None, *inp_shape), X=data[0])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        z_train = np.hstack([np.random.uniform(np.min(data[0], 0), np.max(data[0], 0), data[0].shape) for _ in range(nu)])
        z_val = np.hstack([np.random.uniform(np.min(data[0], 0), np.max(data[0], 0), validation_data[0].shape) for _ in range(nu)])

        self.model.fit(
            data[0], z_train,
            validation_data=(validation_data[0], z_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class Svdd_em(tf.keras.models.Model):

    def __init__(self, preprocessor, nu=1e-4, n_u=100, n_t=100):
        super(Svdd_em, self).__init__()
        self.n_u = n_u
        self.n_t = n_t
        self.nu = nu
        self.preprocessor = preprocessor
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.built = False

    def build(self, input_shape, X):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))

        #self.c = tf.reduce_mean(self.preprocessor(X), 0)

        self.c = self.add_weight(shape=[self.preprocessor.output_shape[1]], initializer='glorot_uniform', name='c', trainable=True)

        self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R', trainable=False)
        #self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R', trainable=True)

        self.built = True

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.preprocessor(x)
        dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        scores = dists - self.R ** 2
        return scores

    def train_step(self, data):
        inputs, randoms = data
        randoms = tf.concat(tf.split(randoms, self.n_u, axis=1), axis=0)
        with tf.GradientTape() as tape:
            x = self.preprocessor(inputs)
            d_x = tf.reduce_sum(tf.square(x - self.c), axis=-1)
            s_x = d_x - self.R ** 2
            p_x = tf.maximum(s_x, tf.zeros_like(s_x))
            u = self.preprocessor(randoms)
            d_u = tf.reduce_sum(tf.square(u - self.c), axis=-1)
            s_u = d_u - self.R ** 2
            p_u = tf.maximum(s_u, tf.zeros_like(s_u))
            loss = -tf.reduce_mean(tf.reduce_sum(tf.maximum(d_u * tf.expand_dims(tf.range(0, 1, 1 / self.n_t), 1) - d_x, 0.0), axis=0))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):
        inputs, randoms = data
        randoms = tf.concat(tf.split(randoms, self.n_u, axis=1), axis=0)
        x = self.preprocessor(inputs)
        d_x = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        s_x = d_x - self.R ** 2
        p_x = tf.maximum(s_x, tf.zeros_like(s_x))
        u = self.preprocessor(randoms)
        d_u = tf.reduce_sum(tf.square(u - self.c), axis=-1)
        s_u = d_u - self.R ** 2
        p_u = tf.maximum(s_u, tf.zeros_like(s_u))
        loss = -tf.reduce_mean(tf.reduce_sum(tf.maximum(d_u * tf.expand_dims(tf.range(0, 1, 1 / self.n_t), 1) - d_x, 0.0), axis=0))
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }


class DSVDDEM(AnomalyDetector):

    def __init__(self):
        super(DSVDDEM, self).__init__()

    def _calculate_distances(self, data, eps=1e-10):
        preds = self.model.predict(data[0], verbose=False)
        p = np.clip(preds, 0, np.inf)
        self.radiuses = np.array([np.mean(p), np.std(p), np.max(p)]).reshape(1, -1)

    def plot(self, data, fig_dir=FIG_DIR, labels=['Normal', 'Defective'], prefix=None, title=None, eps=1e-10, n_samples=2000):

        nc = self.centroids.shape[0]
        print('Number of centroids:', nc)

        u_labels = np.unique(data[1][:n_samples])
        for u_label in u_labels:
            print(u_label, len(np.where(data[1][:n_samples] == u_label)[0]))

        X_mapped = self.model.preprocessor(data[0][:n_samples, :])

        print(X_mapped.shape)

        X_plot = np.vstack([
            X_mapped,
            #(self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
            self.centroids,
        ])

        #X_plot = np.vstack([
        #    data[0][:n_samples, :],
        #    self.centroids,
        #])

        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        X_tsne = tsne.fit_transform(X_plot)
        pp.style.use('default')
        pp.figure(figsize=(5, 3))
        colormap = 'jet'
        scatter_points = pp.scatter(X_tsne[:-nc, 0], X_tsne[:-nc, 1], c=data[1][:n_samples], s=5, cmap=colormap, marker='x', linewidth=0.5)
        scatter_centroids = pp.scatter(X_tsne[-nc:, 0], X_tsne[-nc:, 1], c=np.zeros(nc), s=self.weights / np.sum(self.weights) * 1000, cmap=colormap, edgecolor='black', linewidth=1.5);
        #pp.xlabel('t-SNE feature 1') #, fontsize=10)
        #pp.ylabel('t-SNE feature 2') #, fontsize=10)
        pp.axis('off')
        if title is not None:
            pp.title(title)
        pp.legend(
            scatter_points.legend_elements()[0] + scatter_centroids.legend_elements()[0],
            [label.capitalize() for label in labels] + ['Centre of the model'],
            loc=0
        )
        fname = f'{prefix}_' if prefix is not None else ''
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(fig_dir, f'{fname}tsne.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)
        dists_te = self.model.predict(E_te_, verbose=False)
        nte = E_te_.shape[0]
        pred_thrs = radiuses[0]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _set_radiuses(self, data, metric='em', alpha=3, n_generated=100000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        #print(f'Alpha = {alpha}')
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def _adjust_params(self, kb_max, inp_shape, layers, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        layers_scaled = np.array(layers) / layers[0]
        s = np.dot(layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape)
        x = np.sqrt((a ** 2 + 4 * s * param_size_total) / (4 * (s ** 2))) - a / (2 * s)
        layers_adjusted = []
        for i, scale in enumerate(layers_scaled):
            if only_if_bigger:
                n_units = np.minimum(np.floor(x * scale), layers[i])
            else:
                n_units = np.floor(x * scale)
            layers_adjusted.append(n_units)
        return layers_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, epochs=1, nu=100,
            batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            n_layers = int(np.round(hp[0]))
            n_units = int(np.round(hp[1]))
        else:
            n_layers = 2
            n_units = int(np.round(hp))

        encoder_units = [n_units for _ in range(n_layers)]

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)
        self.encoder = tf.keras.models.Model(inputs, encoded)

        self.model = Svdd_em(preprocessor=self.encoder, n_u=nu)
        self.model.build(input_shape=(None, *inp_shape), X=data[0])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        z_train = np.hstack([np.random.uniform(np.min(data[0], 0), np.max(data[0], 0), data[0].shape) for _ in range(nu)])
        z_val = np.hstack([np.random.uniform(np.min(data[0], 0), np.max(data[0], 0), validation_data[0].shape) for _ in range(nu)])

        self.model.fit(
            data[0], z_train,
            validation_data=(validation_data[0], z_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self.centroids = self.model.c.numpy().reshape(1, -1)
        self.weights = [data[0].shape[0]]

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


class Svdd2(tf.keras.models.Model):

    def __init__(self, preprocessor, nu=1e-3):
        super(Svdd2, self).__init__()
        self.nu = nu
        self.preprocessor = preprocessor
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.built = False

    def build(self, input_shape, X):
        input_dims = input_shape[1:]
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.float32, shape=(None, *input_dims))

        c = tf.reduce_mean(self.preprocessor(X), 0)
        self.c = self.add_weight(shape=[self.preprocessor.output_shape[1]], initializer='glorot_uniform', name='c', trainable=True)
        self.c.assign(c)
        #self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R', trainable=False)
        self.R = self.add_weight(shape=[], initializer='glorot_uniform', name='R', trainable=True)
        self.built = True

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.preprocessor(x)
        dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        scores = dists - self.R ** 2
        return scores

    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            x = self.preprocessor(inputs)
            dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
            scores = dists - self.R ** 2
            penalty = tf.maximum(scores, tf.zeros_like(scores))
            loss = self.R ** 2 + (1 / self.nu) * penalty

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        #R_new = tf.sort(tf.math.sqrt(dists))[tf.cast((1 - self.nu) * tf.math.reduce_sum(tf.ones_like(dists)), tf.int32)]
        #self.R.assign(R_new)

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):
        if len(data) == 2:
            inputs, outputs = data
        else:
            inputs, outputs = data[0]
        x = self.preprocessor(inputs)
        dists = tf.reduce_sum(tf.square(x - self.c), axis=-1)
        scores = dists - self.R ** 2
        penalty = tf.maximum(scores, tf.zeros_like(scores))
        loss = self.R ** 2 + (1 / self.nu) * penalty
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }


class DSVDD2(AnomalyDetector):

    def __init__(self):
        super(DSVDD2, self).__init__()

    def _calculate_distances(self, data, eps=1e-10):
        preds = self.model.predict(data[0], verbose=False)
        p = np.clip(preds, 0, np.inf)
        self.radiuses = np.array([np.mean(p), np.std(p), np.max(p)]).reshape(1, -1)

    def plot(self, data, fig_dir=FIG_DIR, labels=['Normal', 'Defective'], prefix=None, title=None, eps=1e-10, n_samples=2000):

        nc = self.centroids.shape[0]
        print('Number of centroids:', nc)

        u_labels = np.unique(data[1][:n_samples])
        for u_label in u_labels:
            print(u_label, len(np.where(data[1][:n_samples] == u_label)[0]))

        X_mapped = self.model.preprocessor(data[0][:n_samples, :])

        print(X_mapped.shape)

        X_plot = np.vstack([
            X_mapped,
            #(self.centroids - self.xmin[None, :]) / (self.xmax[None, :] - self.xmin[None, :] + eps),
            self.centroids,
        ])

        #X_plot = np.vstack([
        #    data[0][:n_samples, :],
        #    self.centroids,
        #])

        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        X_tsne = tsne.fit_transform(X_plot)
        pp.style.use('default')
        pp.figure(figsize=(5, 3))
        colormap = 'jet'
        scatter_points = pp.scatter(X_tsne[:-nc, 0], X_tsne[:-nc, 1], c=data[1][:n_samples], s=5, cmap=colormap, marker='x', linewidth=0.5)
        scatter_centroids = pp.scatter(X_tsne[-nc:, 0], X_tsne[-nc:, 1], c=np.zeros(nc), s=self.weights / np.sum(self.weights) * 1000, cmap=colormap, edgecolor='black', linewidth=1.5);
        #pp.xlabel('t-SNE feature 1') #, fontsize=10)
        #pp.ylabel('t-SNE feature 2') #, fontsize=10)
        pp.axis('off')
        if title is not None:
            pp.title(title)
        pp.legend(
            scatter_points.legend_elements()[0] + scatter_centroids.legend_elements()[0],
            [label.capitalize() for label in labels] + ['Centre of the model'],
            loc=0
        )
        fname = f'{prefix}_' if prefix is not None else ''
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(fig_dir, f'{fname}tsne.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

    def predict(self, data, alpha, eps=1e-10):

        radiuses = self.radiuses[:, 0] + alpha * self.radiuses[:, 1]
        #radiuses = self.radiuses[:, 2] + alpha * self.radiuses[:, 1]

        E_te_ = np.array(data, dtype=np.float32)
        dists_te = self.model.predict(E_te_, verbose=False)
        nte = E_te_.shape[0]
        pred_thrs = radiuses[0]
        predictions = np.zeros(nte)
        predictions[np.where(dists_te > pred_thrs)[0]] = 1
        scores = (pred_thrs - dists_te) / (pred_thrs + eps)
        probs = dists_te / (pred_thrs + 1e-10)
        return predictions, scores, probs

    def _set_radiuses(self, data, metric='em', alpha=3, n_generated=100000):
        n_features = data.shape[1]
        volume_support = (np.ones(n_features) - np.zeros(n_features)).prod()
        X_unif = np.random.uniform(self.xmin, self.xmax, size=(n_generated, n_features))
        metric_fun = getattr(self, f'_{metric}')
        alpha = np.maximum(alpha, np.max((self.radiuses[:, 2] - self.radiuses[:, 0]) / (self.radiuses[:, 1] + 1e-10)))
        #print(f'Alpha = {alpha}')
        _, s_X, _ = self.predict(data, alpha)
        assert s_X is not None
        _, s_U, _ = self.predict(X_unif, alpha)
        assert s_U is not None
        metric_val = metric_fun(volume_support, s_U, s_X)[0]
        return alpha, metric_val

    def _adjust_params(self, kb_max, inp_shape, layers, bytes_per_param=4, only_if_bigger=True):
        param_size_total = kb_max * 1024 / bytes_per_param
        layers_scaled = np.array(layers) / layers[0]
        s = np.dot(layers_scaled[1:], layers_scaled[:-1])
        a = np.prod(inp_shape)
        x = np.sqrt((a ** 2 + 4 * s * param_size_total) / (4 * (s ** 2))) - a / (2 * s)
        layers_adjusted = []
        for i, scale in enumerate(layers_scaled):
            if only_if_bigger:
                n_units = np.minimum(np.floor(x * scale), layers[i])
            else:
                n_units = np.floor(x * scale)
            layers_adjusted.append(n_units)
        return layers_adjusted

    def fit(self, data, validation_data, hp, mem_max, metric='em', std=True, epochs=1,
            batch_size=1, lr=1e-3, patience=1, eps=1e-10, verbose=False, adjust_params=True):

        self.xmin = np.min(data[0], 0)
        self.xmax = np.max(data[0], 0)

        inp_shape = data[0].shape[1:]
        inputs = tf.keras.layers.Input(shape=inp_shape)

        if type(hp) == list:
            n_layers = int(np.round(hp[0]))
            n_units = int(np.round(hp[1]))
        else:
            n_layers = 2
            n_units = int(np.round(hp))

        encoder_units = [n_units for _ in range(n_layers)]

        encoder_units_ = self._adjust_params(mem_max, inp_shape, encoder_units)
        if not adjust_params:
            assert encoder_units_ == encoder_units
            #print('encoder units before:', encoder_units)
            #encoder_units = self._adjust_params(mem_max, inp_shape, encoder_units)
            #print('encoder units after:', encoder_units)
        encoder_units = encoder_units_

        if std:
            #hidden = (inputs - np.mean(data[0], 0)[None, :]) / (np.std(data[0], 0)[None, :] + eps)
            hidden = (inputs - np.min(data[0], 0)[None, :]) / (np.max(data[0], 0)[None, :] - np.min(data[0], 0)[None, :] + eps)
        else:
            hidden = inputs

        for units in encoder_units[:-1]:
            hidden = tf.keras.layers.Dense(units=units)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)

        encoded = tf.keras.layers.Dense(units=encoder_units[-1])(hidden)
        self.encoder = tf.keras.models.Model(inputs, encoded)

        self.model = Svdd2(preprocessor=self.encoder)
        self.model.build(input_shape=(None, *inp_shape), X=data[0])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        if verbose:
            self.model.summary()

        self.model.fit(
            data[0], data[1],
            validation_data=(validation_data[0], validation_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
            ],
            verbose=verbose
        )

        self.centroids = self.model.c.numpy().reshape(1, -1)
        self.weights = [data[0].shape[0]]

        self._calculate_distances(validation_data)

        alpha, metric_val = self._set_radiuses(validation_data[0], metric=metric)

        return alpha, metric_val


HYPERPARAMS = {

        'KM': [2, 3, 4, 5],

        'CS': [[2, 4, 6, 8], [1, 2, 3, 4]],  # the number of clusters, the number of micro-clusters to the number of clusters ratio
        'SKM': [[2, 4, 6, 8], [1, 2, 3, 4]],  # the number of clusters, the number of candidate centroids to the number of clusters ratio
        'WAP': [[2, 4, 6, 8], [1, 2, 3, 4]],  # the number of clusters, reservoir size to the number of clusters ratio
        'WCM': [[2, 4, 6, 8], [1, 2, 3, 4]],  # the number of clusters, batch size to the number of clusters ratio

        'SOM': [[4, 6, 8, 10], [4, 6, 8, 10]],  # map width, map height
        'GNG': [10, 20, 30, 40],  # a_max

        'VAE': [[2, 3, 4], [16, 32, 48, 64]],  # n_layers, n_units
        'UAE': [[2, 3, 4], [16, 32, 48, 64]],  # n_layers, n_units
        'SAE': [[2, 3, 4], [16, 32, 48, 64]],  # n_layers, n_units
        'DAE': [[2, 3, 4], [16, 32, 48, 64]],  # n_layers, n_units

        'DSOM': [[16, 32, 48, 64], [4, 6, 8, 10]], # encoder n_units, map width/height
        'DSVDD': [[1, 2, 3], [16, 32, 48, 64]],  # n_layers, n_units
        'DSVDD2': [[2, 3, 4], [16, 32, 48, 64]],  # n_layers, n_units

        'BGAN': [None],
        'DWCM': [None],
        'DEM': [[2, 3, 4], [16, 32, 48, 64]],
        'DSVDDEM': [[2, 3, 4], [16, 32, 48, 64]],  # n_layers, n_units
    }
