import argparse
import errno
import os
import shutil

from data_utils.utils.files import ifnot_create
from ray import tune


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


def get_dirs(path):
    for node in os.listdir(path, ):
        full_path = os.path.join(path, node)
        if os.path.isdir(full_path):
            yield full_path


def compare_top_experiment(src_dist, dest_dist, metric='eval_acc', mode='max'):
    if not os.path.exists(src_dist):
        raise Exception("Directory {} does not exist".format(src_dist))

    dest_dist = ifnot_create(dest_dist)

    for exp_dir in get_dirs(src_dist):
        exp_name = os.path.basename(os.path.normpath(exp_dir))

        analysis = tune.Analysis(exp_dir)

        best_expdir = analysis.get_best_logdir(metric=metric, mode=mode)
        copyanything(best_expdir, os.path.join(dest_dist, exp_name))


parser = argparse.ArgumentParser(
    description="Runs experiments for Non Negative Training")

parser.add_argument('src', help='source of experiments', type=str)
parser.add_argument('dest', help='destination of analysis', type=str)
parser.add_argument('--metric',
                    '-m',
                    help='metric for sorting',
                    type=str,
                    default='eval_acc')
parser.add_argument('--mode',
                    '-d',
                    help='mode for sorting',
                    type=str,
                    default='max')
parser.add_argument('--top', '-t', help='top experiments', type=int, default=1)

args = parser.parse_args()

if args.top == 1:
    compare_top_experiment(args.src, args.dest, args.metric, args.mode)
else:
    assert False
