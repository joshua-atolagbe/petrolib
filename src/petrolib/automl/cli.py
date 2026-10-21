'''
Command line tool to train AutoGluon models and fill missing log curves

    petrolib-impute wells.csv --targets DT --features GR RT NPHI RHOB DEPTH \
        --well-col WELL --holdout-wells 15_9-F-1A --time-limit 600 --output wells_filled.csv

    petrolib-impute wells.csv --targets DT --well-col WELL --per-well --holdout-fraction 0.2

Also runs as `python -m petrolib.automl ...`

'''
import argparse
from pathlib import Path

from .imputer import LogImputer
from ..io.wells import load_well


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog='petrolib-impute',
                                     description='Train AutoGluon models on log data and fill missing log curves')
    parser.add_argument('input', help='CSV, Excel, LAS or DLIS file with the log data (one or more wells)')
    parser.add_argument('--targets', nargs='+', required=True, help='curves to predict and fill, e.g. DT NPHI')
    parser.add_argument('--features', nargs='+', help='input curves (default: all other numeric columns)')
    parser.add_argument('--well-col', help='column with the well names')
    parser.add_argument('--per-well', action='store_true',
                        help='train one model per well (needs --well-col); other wells use a field-wide model')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='with --per-well, the fewest measured values a well needs for its own model')
    parser.add_argument('--no-fallback', action='store_true',
                        help='with --per-well, do not train the field-wide model for wells without their own')
    parser.add_argument('--holdout-wells', nargs='+', help='wells kept out of training to score a field-wide model')
    parser.add_argument('--holdout-fraction', type=float,
                        help='fraction of each well, in one block mid-well, kept out of training to score the models')
    parser.add_argument('--no-refit', action='store_true',
                        help='after scoring, do not retrain on all the data')
    parser.add_argument('--time-limit', type=float, help='training time limit per target, in seconds')
    parser.add_argument('--presets', default='medium_quality',
                        help='AutoGluon preset: medium_quality, good_quality, high_quality or best_quality')
    parser.add_argument('--model-dir', default='petrolib_imputer', help='folder to save the models in')
    parser.add_argument('--output', help='output CSV (default: <input>_filled.csv)')
    parser.add_argument('--keep-prediction', action='store_true', help='also write <TARGET>_PRED columns')
    parser.add_argument('--standardize', action='store_true',
                        help='rename recognized curves to standard names (GR, RT, NPHI, RHOB, DT, ...) first')
    parser.add_argument('--verbosity', type=int, default=1, help='AutoGluon verbosity, 0 to 4')
    return parser


def main(argv=None):

    args = build_parser().parse_args(argv)
    df = load_well(args.input, standardize=args.standardize)

    imputer = LogImputer(args.targets, features=args.features, well_col=args.well_col, per_well=args.per_well,
                         min_samples=args.min_samples, fallback=not args.no_fallback, path=args.model_dir,
                         presets=args.presets, time_limit=args.time_limit, verbosity=args.verbosity)
    filled = imputer.fit_fill(df, holdout_wells=args.holdout_wells, holdout_fraction=args.holdout_fraction,
                              refit=not args.no_refit, keep_prediction=args.keep_prediction)
    imputer.save()

    output = args.output or str(Path(args.input).with_name(Path(args.input).stem + '_filled.csv'))
    filled.to_csv(output, index=False)

    if imputer.scores_ is not None:
        print('\nScores on the held-out data')
        print(imputer.scores_.to_string(index=False))
    for target in imputer.targets:
        print(f"{target}: {int(filled[f'{target}_IMPUTED'].sum())} values filled")
    print(f'Models saved in {args.model_dir}\nFilled data written to {output}')


if __name__ == '__main__':
    main()
