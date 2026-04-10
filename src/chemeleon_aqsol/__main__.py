from pathlib import Path
import argparse
import pandas as pd
from .inference import CheMeleonAqueous


def parse_args(arguments=None):
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(prog='chemeleon-aqsol', description=(
        'Applies the pretrained model to predict kinetic aqueous solubility on '
        'new compounds (uses chemeleon_aqueous.pt).'
    ))
    subparsers = parser.add_subparsers(dest='subcommand')

    # predict subcommand
    _predict = subparsers.add_parser('predict', help='Predict kinetic aqueous solubility')
    _predict.add_argument('infile', type=Path)
    _predict.add_argument('-c', '--smiles-col', default='SMILES')
    _predict.add_argument('-o', '--outfile', type=Path, default=None)
    _predict.add_argument('-m', '--model', type=Path, default=None)

    args = parser.parse_args(arguments)
    return args


def read_infile(infile: Path) -> pd.DataFrame:
    """Read input file"""
    if infile.suffix == '.csv':
        return pd.read_csv(infile)
    elif infile.suffix == '.tsv':
        return pd.read_csv(infile, sep='\t')
    elif infile.suffix == '.txt' or infile.suffix == '.smi':
        return pd.read_csv(infile, sep='\s+')
    elif infile.suffix == '.xlsx':
        return pd.read_excel(infile)
    raise ValueError(f"Cannot recognize filetype for: '{infile!s}'")


def run_predict(
    infile: Path,
    outfile: Path | None = None,
    model: Path | None = None,
    smiles_column: str = 'SMILES'
):

    if outfile is None:
        outfile = infile.parent / (infile.stem + '_out.csv')

    # Read data
    df = read_infile(infile)
    smiles = df[smiles_column].to_list()

    # Run inference
    chemaqsol = CheMeleonAqueous(model_path=model)
    y_pred = chemaqsol(smiles)

    # Write output
    df['chemeleon-aqsol logS(M)'] = y_pred
    df.to_csv(outfile, index=False)



def main(arguments = None):
    """Main function"""
    # Parse arguments
    args = parse_args(arguments)

    # Run prediction
    if args.subcommand == 'predict':
        run_predict(args.infile, outfile=args.outfile, model=args.model, smiles_column=args.smiles_col)


if __name__ == "__main__":
    main()
