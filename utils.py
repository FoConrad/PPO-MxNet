#!/usr/bin/env python

import datetime
import json
import os
import pickle

#import tensorboard_logger as tlogger

def dump_binary_list(dump_list, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(dump_list, fp)


def try_or_none(dic, key, f):
    try:
        dic[key] = f()
    except:
        pass

def dump_partitioned_graph(cost_model, name='partition_dump'):
    nodes_json = {'nodes':[]}
    for n in cost_model.iter_nodes:
        node_json = {
                'name': n._name, 
                'op': n._op,
                'inputs': [ni._name for ni in n._inputs],
                }
        try_or_none(node_json, 'cost', lambda: cost_model._node_map[n])
        try_or_none(node_json, 'partition', lambda: cost_model._partition_scheme[n].partition)
        try_or_none(node_json, 'placement', lambda: cost_model._placement_scheme[n])
        nodes_json['nodes'].append(node_json)

    with open(name + '.json', 'w') as f:
        f.write(json.dumps(nodes_json, indent=4))


def cost_iteration_figure(cost_series, name, y_axis, scale=1, info=None):
    string = ''
    if info:
        for k in info:
            string += '{}: {}\n'.format(k, info[k])
    try:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot([scale * i for i, _ in enumerate(cost_series)], cost_series)
        plt.legend([string])
        plt.ylabel('{} cost of greedy query'.format(y_axis))
        plt.xlabel('episodes')
        plt.savefig(name)
        plt.close()
    except Exception as e:
        pass


def save_arg_dict(d, base_dir='./', filename='args.txt', log=True):
    def _format_value(v):
        if isinstance(v, float):
            return '%.4f' % v
        elif isinstance(v, int):
            return '%d' % v
        else:
            return '%s' % str(v)

    with open(os.path.join(base_dir, filename), 'w') as f:
        for k, v in d.items():
            f.write('%s\t%s\n' % (k, _format_value(v)))
    if log:
        print('Saved settings to %s' % os.path.join(base_dir, filename))


def mkdir_p(path, log=True):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    if log:
        print('Created directory %s' % path)


def date_filename(base_dir='./', prefix=''):
    dt = datetime.datetime.now()
    return os.path.join(base_dir, '{}{}_{:02d}-{:02d}-{:02d}'.format(
        prefix, dt.date(), dt.hour, dt.minute, dt.second))


def setup_tensorboard(log_directory):
    """Creates a logging directory and configures a tensorboard logger."""
    # NOTE(wellecks) Instead of `tensorboard_logger` module (which only supports
    #                scalar outputs), we could create our own wrapper for
    #                logging alternative outputs (e.g. images) with tensorboard.
    try:
        tlogger.configure(log_directory)
    except ValueError:
        pass
    return tlogger


def log_tensorboard(values_dict, step):
    if tlogger.tensorboard_logger._default_logger is not None:
        for k, v in values_dict.items():
            tlogger.log_value(k, v, step)


def setup(args):
    """Boilerplate setup, returning dict of configured items."""
    if args.base_log_dir is not None:
        log_directory = date_filename(args.base_log_dir, args.expr_name)
        mkdir_p(log_directory)
        save_arg_dict(args.__dict__, base_dir=log_directory)
        tlogger = setup_tensorboard(log_directory)
        return dict(log_directory=log_directory,
                    tlogger=tlogger)
    else:
        print("NOTE: Log persistence and tensorboard disabled; set `--base-log-dir` to enable.")
    return dict()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
