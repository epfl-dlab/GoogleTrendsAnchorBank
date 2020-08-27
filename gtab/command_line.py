import argparse
import copy
import glob
import json
import os

from .core import GTAB

dir_path = os.path.dirname(os.path.abspath(__file__))


# --- UTILITY METHODS ---

class GroupedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        group, dest = self.dest.split('.', 2)
        groupspace = getattr(namespace, group, argparse.Namespace())
        setattr(groupspace, dest, values)
        setattr(namespace, group, groupspace)


def _load_dir_cl():
    with open(os.path.join(dir_path, "config", "dir_cl.json"), 'r') as fp:
        dir_cl = json.load(fp)

    if dir_cl['dir_cl'].strip() == "":
        raise Exception("No active directory set! Must call 'init' first!")

    print(dir_cl['active_gtab'])
    return dir_cl['dir_cl'], dir_cl['active_gtab']


# --- "EXPOSED" METHODS ---

def init_dir():
    parser = argparse.ArgumentParser(prog='init_dir')
    parser.add_argument("path", help="Path of the desired directory to be initialized/used.", type=str)
    args = parser.parse_args()
    path = os.path.abspath(args.path)

    t = GTAB(path, from_cli=True)
    with open(os.path.join(dir_path, "config", "dir_cl.json"), 'w') as fp:
        json.dump({"dir_cl": path, "active_gtab": "google_anchorbank_geo=_timeframe=2019-01-01 2020-08-01.tsv"}, fp,
                  indent=4, sort_keys=True)

    print("Directory initialized!")


def print_options():
    dir_cl, _ = _load_dir_cl()
    print(f"Active directory is: {dir_cl}")
    t = GTAB(dir_cl, from_cli=True)
    t.print_options()

    return None


def set_options():
    parser = argparse.ArgumentParser(prog="set_options")
    parser.add_argument("--geo", type=str, dest="pytrends.geo", action=GroupedAction, default=argparse.SUPPRESS)
    parser.add_argument("--timeframe", type=str, dest='pytrends.timeframe', action=GroupedAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--num_anchor_candidates", type=int, dest='gtab.num_anchor_candidates', action=GroupedAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--num_anchors", type=int, dest='gtab.num_anchors', action=GroupedAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, dest='gtab.seed', action=GroupedAction, default=argparse.SUPPRESS)
    parser.add_argument("--sleep", type=float, dest='gtab.sleep', action=GroupedAction, default=argparse.SUPPRESS)
    parser.add_argument("--thresh_offline", type=int, dest='gtab.thresh_offline', action=GroupedAction,
                        default=argparse.SUPPRESS)

    parser.add_argument("--backoff_factor", type=float, dest='conn.backoff_factor', action=GroupedAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--proxies", type=str, dest='conn.proxies', action=GroupedAction, default=argparse.SUPPRESS,
                        nargs="+")
    parser.add_argument("--retries", type=int, dest='conn.retries', action=GroupedAction, default=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, dest='conn.timeout', action=GroupedAction, default=argparse.SUPPRESS,
                        nargs=2)

    args = vars(parser.parse_args())

    dir_cl, _ = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    t.set_options(pytrends_config=vars(args.get('pytrends')) if args.get('pytrends') != None else None,
                  gtab_config=vars(args.get('gtab')) if args.get('gtab') != None else None,
                  conn_config=vars(args.get('conn')) if args.get('conn') != None else None,
                  overwite_file=True)


def set_blacklist():
    parser = argparse.ArgumentParser(prog="set_blacklist")
    parser.add_argument("blacklist", type=str, nargs='+')
    args = parser.parse_args()

    dir_cl, _ = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    t.set_blacklist(args.blacklist, overwrite_file=True)


def set_hitraffic():
    parser = argparse.ArgumentParser(prog="set_hitraffic")
    parser.add_argument("hitraffic", type=str, nargs='+')
    args = parser.parse_args()

    dir_cl, _ = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    t.set_hitraffic(args.hitraffic, overwrite_file=True)


def list_gtabs():
    dir_cl, active_gtab = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    if active_gtab.strip() != "":
        t.set_active_gtab(active_gtab)
    t.list_gtabs()


def rename_gtab():
    parser = argparse.ArgumentParser(prog="rename_gtab")
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    args = parser.parse_args()

    dir_cl, active_gtab = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    if active_gtab.strip() != "":
        t.set_active_gtab(active_gtab)
    t.rename_gtab(args.src, args.dst)


def delete_gtab():
    parser = argparse.ArgumentParser(prog="delete_gtab")
    parser.add_argument("src", type=str)
    args = parser.parse_args()

    dir_cl, active_gtab = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    if active_gtab.strip() != "":
        t.set_active_gtab(active_gtab)
    t.delete_gtab(args.src)


def set_active_gtab():
    parser = argparse.ArgumentParser(prog="set_active_gtab")
    parser.add_argument("src", type=str)
    args = parser.parse_args()

    dir_cl, _ = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    t.set_active_gtab(args.src)

    with open(os.path.join(dir_path, "config", "dir_cl.json"), 'w') as fp:
        json.dump({"dir_cl": dir_cl, "active_gtab": args.src}, fp, indent=4, sort_keys=True)


def create_gtab():
    dir_cl, _ = _load_dir_cl()
    t = GTAB(dir_cl, from_cli=True)
    t.create_anchorbank(verbose=True)


def new_query():
    dir_cl, active_gtab = _load_dir_cl()

    parser = argparse.ArgumentParser(prog="new_query")
    parser.add_argument("kws", type=str, nargs="+")
    parser.add_argument("--results_file", type=str, default="query_results.json")
    args = parser.parse_args()

    t = GTAB(dir_cl, from_cli=True)
    if active_gtab.strip() == "":
        raise Exception("Must use 'gtab-set-active' first to select the active gtab!")
    t.set_active_gtab(active_gtab)

    rez = {}
    for kw in args.kws:
        t_rez = t.new_query(kw)
        rez[kw] = copy.deepcopy(t_rez)

    rez = json.loads(json.dumps(rez))

    print(args.results_file)

    os.makedirs(os.path.join(dir_cl, "query_results"), exist_ok=True)
    with open(os.path.join(dir_cl, "query_results", args.results_file), 'w') as fp:
        json.dump(rez, fp, indent=4)
