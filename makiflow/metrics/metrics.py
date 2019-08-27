from __future__ import absolute_import
from makiflow.metrics.utils import one_hot
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

EPSILON = 1e-9


def binary_dice(predicted, actual):
    num = np.sum(predicted * actual)
    den = np.sum(predicted * predicted) + np.sum(actual)
    return (2 * num + EPSILON) / (den + EPSILON)


def categorical_dice_coeff(predicted, actual, num_classes):
    # Without `flatten` shape is [batch_sz] and [batch_sz, num_classes]
    # With `flatten` shape is [batch_sz, img_w, im_h] and [batch_sz, img_w, im_h, num_classes]
    batch_sz = len(predicted)
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    actual = actual.reshape(batch_sz, -1)
    predicted = predicted.reshape(batch_sz, -1, num_classes)

    class_dices = np.zeros(num_classes)
    for i in range(batch_sz):
        sample_actual = actual[i]
        sample_pred = predicted[i]
        for j in range(num_classes):
            sub_actual = (sample_actual[:] == j).astype(np.int32)
            sub_confs = sample_pred[:, j]
            class_dices[j] += binary_dice(sub_confs, sub_actual)
    return class_dices / batch_sz, class_dices.mean() / batch_sz


def v_dice_coeff(P, L, use_argmax=False, one_hot_labels=False):
    """
    Calculates V-Dice for give predictions and labels.
    WARNING! THIS IMPLIES SEGMENTATION CONTEXT.
    Parameters
    ----------
    P : np.ndarray
        Predictions of a segmentator. Array of shape [batch_sz, W, H, num_classes].
    L : np.ndarray
        Labels for the segmentator. Array of shape [batch_sz, W, H]
    use_argmax : bool
        Converts the segmentator's predictions to one-hot format.
        Example: [0.4, 0.1, 0.5] -> [0., 0., 1.]
    one_hot_labels : bool
        Set to True if the labels (`L`) are already one-hot encoded, i.e. have the same
        shape as `P`.
    """
    # P has shape [batch_sz, W, H, num_classes]
    # L has shape [batch_sz, W, H]
    # RESHAPE TENSORS AND ONE-HOT LABELS
    # P -> [batch_sz, num_samples, num_classes]
    batch_sz = len(P)
    num_samples = P.shape[1] * P.shape[2]
    num_classes = P.shape[-1]
    if use_argmax:
        P = P.argmax(axis=3)
        P = P.reshape(batch_sz * num_samples)
        P = one_hot(P, depth=num_classes)
    P = P.reshape(batch_sz, num_samples, num_classes)

    if not one_hot_labels:
        # L -> [batch_sz*num_samples] -> [batch_sz*num_samples, num_classes] -> [batch_sz, num_samples, num_classes]
        L = L.reshape(batch_sz * num_samples)
        L = one_hot(L, depth=num_classes)
        L = L.reshape(batch_sz, num_samples, num_classes)

    # P has shape [batch_sz, num_samples, num_classes]
    # L has shape [batch_sz, num_samples, num_classes]
    R = P * L
    nums = R.sum(axis=1)

    P2 = P * P
    P2vec = P2.sum(axis=1)
    Lvec = L.sum(axis=1)
    dens = P2vec + Lvec
    dices_b = (2 * nums + EPSILON) / (dens + EPSILON)
    dices = dices_b.mean(axis=0)
    return dices.mean(), dices


def confusion_matrix(
        p, l,
        use_argmax_p=False, use_argmax_l=False, to_flatten=False,
        save_path=None, dpi=200):
    """
    Creates confusion matrix for the given predictions `p` and labels `l`.
    Parameters
    ----------
    p : np.ndarray
        Predictions.
    l : np.ndarray
        Corresponding labels.
    use_argmax_p : bool
        Set to true if prediction aren't sparse, i.e. `p` is an array of shape [..., num_classes].
    use_argmax_l : bool
        Set to true if labels aren't sparse (one-hot encoded), i.e. `l` is an array of shape [..., num_classes].
    to_flatten : bool
        Set to true if `p' and `l` are high-dimensional arrays.
    save_path : str
        Saving path for the confusion matrix picture.
    dpi : int
        Affects the size of the saved confusion matrix picture.
    """
    if use_argmax_p:
        p = p.argmax(axis=-1)

    if use_argmax_l:
        l = l.argmax(axis=-1)

    if to_flatten:
        p = p.reshape(-1)
        l = l.reshape(-1)

    mat = confusion_matrix(l, p)

    if save_path is not None:
        conf_mat = sns.heatmap(mat, annot=True)
        conf_mat.figure.savefig(save_path, dpi=dpi)
    return mat
