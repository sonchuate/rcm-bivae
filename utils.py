import numpy as np
import numbers

def estimate_batches(input_size, batch_size):
    """
    Estimate number of batches give `input_size` and `batch_size`
    """
    return int(np.ceil(input_size / batch_size))

def validate_format(input_format, valid_formats):
    """Check the input format is in list of valid formats
    :raise ValueError if not supported
    """
    if not input_format in valid_formats:
        raise ValueError('{} data format is not in valid formats ({})'.format(input_format, valid_formats))

    return input_format

def get_rng(seed):
    '''Return a RandomState of Numpy.
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    '''
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))

def scale(values, target_min, target_max, source_min=None, source_max=None):
    """Scale the value of a numpy array "values"
    from source_min, source_max into a range [target_min, target_max]

    Parameters
    ----------
    values : Numpy array, required
        Values to be scaled.

    target_min : scalar, required
        Target minimum value.

    target_max : scalar, required
        Target maximum value.

    source_min : scalar, required, default: None
        Source minimum value if desired. If None, it will be the minimum of values.

    source_max : scalar, required, default: None
        Source minimum value if desired. If None, it will be the maximum of values.

    Returns
    -------
    res: Numpy array
        Output values mapped into range [target_min, target_max]
    """
    if source_min is None:
        source_min = np.min(values)
    if source_max is None:
        source_max = np.max(values)
    if source_min == source_max:  # improve this scenario
        source_min = 0.0

    values = (values - source_min) / (source_max - source_min)
    values = values * (target_max - target_min) + target_min
    return values
def clip(values, lower_bound, upper_bound):
    """Perform clipping to enforce values to lie
    in a specific range [lower_bound, upper_bound]

    Parameters
    ----------
    values : Numpy array, required
        Values to be clipped.

    lower_bound : scalar, required
        Lower bound of the output.

    upper_bound : scalar, required
        Upper bound of the output.

    Returns
    -------
    res: Numpy array
        Clipped values in range [lower_bound, upper_bound]
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_PREDICTION_COL = "rate"


def predict(
    model,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL,
):
    """Computes predictions of a recommender model from Cornac on the data.
    Can be used for computing rating metrics like RMSE.

    Args:
        model (cornac.models.Recommender): A recommender model from Cornac
        data (pandas.DataFrame): The data on which to predict
        usercol (str): Name of the user column
        itemcol (str): Name of the item column

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    uid_map = model.train_set.uid_map
    iid_map = model.train_set.iid_map
    predictions = [
        [
            getattr(row, usercol),
            getattr(row, itemcol),
            model.rate(
                user_idx=uid_map.get(getattr(row, usercol), len(uid_map)),
                item_idx=iid_map.get(getattr(row, itemcol), len(iid_map)),
            ),
        ]
        for row in data.itertuples()
    ]
    predictions = pd.DataFrame(data=predictions, columns=[usercol, itemcol, predcol])
    return predictions


def predict_ranking(
    model,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL,
    remove_seen=False,
):
    """Computes predictions of recommender model from Cornac on all users and items in data.
    It can be used for computing ranking metrics like NDCG.

    Args:
        model (cornac.models.Recommender): A recommender model from Cornac
        data (pandas.DataFrame): The data from which to get the users and items
        usercol (str): Name of the user column
        itemcol (str): Name of the item column
        remove_seen (bool): Flag to remove (user, item) pairs seen in the training data

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    users, items, preds = [], [], []
    item = list(model.train_set.iid_map.keys())
    for uid, user_idx in model.train_set.uid_map.items():
        user = [uid] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(model.score(user_idx).tolist())

    all_predictions = pd.DataFrame(
        data={usercol: users, itemcol: items, predcol: preds}
    )

    if remove_seen:
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions
# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from timeit import default_timer


class Timer(object):
    """Timer class.

    `Original code <https://github.com/miguelgfierro/pybase/blob/2298172a13fb4a243754acbc6029a4a2dcf72c20/log_base/timer.py>`_.

    Examples:
        >>> import time
        >>> t = Timer()
        >>> t.start()
        >>> time.sleep(1)
        >>> t.stop()
        >>> t.interval < 1
        True
        >>> with Timer() as t:
        ...   time.sleep(1)
        >>> t.interval < 1
        True
        >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        'Time elapsed 1...'
    """

    def __init__(self):
        self._timer = default_timer
        self._interval = 0
        self.running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "{:0.4f}".format(self.interval)

    def start(self):
        """Start the timer."""
        self.init = self._timer()
        self.running = True

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        try:
            self._interval = self.end - self.init
            self.running = False
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() as t:"
            )

    @property
    def interval(self):
        """Get time interval in seconds.

        Returns:
            float: Seconds.
        """
        if self.running:
            raise ValueError("Timer has not been stopped, please use stop().")
        else:
            return self._interval
