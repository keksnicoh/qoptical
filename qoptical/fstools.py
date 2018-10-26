# -*- coding: utf-8 -*-
""" utils to persist data in the fs
    :author: keksnicoh
    """

import os
import json
import numpy as np


def is_json_persistable(v):
    """ if something can be serialized as a json
        """
    return isinstance(v, (str, int, float)) \
        or (isinstance(v, list) and np.all([is_json_persistable(vv) for vv in v])) \
        or (isinstance(v, dict) and True)


def is_np_persistable(v):
    """ if something is persitable with `np.save()`
    """
    return isinstance(v, np.ndarray)


def load_fs(name, data_keys):
    """ loads several data_key from fs """
    if not os.path.exists(name):
        raise FileExistsError(name)

    meta_file = os.path.join(name, '_meta.json')
    if not os.path.exists(meta_file):
        raise RuntimeError('_meta.json does not exist')

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    data = []
    for key in data_keys:
        if key in meta['np_files']:
            data.append(np.load(os.path.join(name, key + '.npy')))
        elif key in meta['json_files']:
            with open(os.path.join(name, key + '.json'), 'r') as f:
                data.append(json.load(f))

    return data


def persist_fs(name, **data):
    """ persists a list of jsonable or numpy ndarray to
        file system.

        Arguments:
        ----------
        :name: folder name

        :**data: dict of data to persist

        """
    meta = {
        'json_files': [k for k, v in data.items() if is_json_persistable(v)],
        'np_files': [k for k, v in data.items() if is_np_persistable(v)]
    }

    # check if we did not forget something
    len_union = len(set(meta['json_files'] + meta['np_files']))
    len_disj = len(set(meta['json_files'])) + len(set(meta['np_files']))
    assert len_union == len(data), "%d != %d" % (len_union, len(data))
    assert len_union == len_disj

    # make dir (if not exists)
    if os.path.exists(name):
        raise RuntimeError('file {} exists.'.format(name))
    os.makedirs(name)

    # persist
    with open(os.path.join(name, '_meta.json'), 'w') as rmeta:
        json.dump(meta, rmeta)

    for json_file in meta['json_files']:
        with open(os.path.join(name, json_file + '.json'), 'w') as rf:
            try:
                json.dump(data[json_file], rf)
            except TypeError:
                raise ValueError('could no serialize {} ({})'.format(json_file, type(data[json_file])))
    for np_file in meta['np_files']:
        np.save(os.path.join(name, np_file + '.npy'), data[np_file])
