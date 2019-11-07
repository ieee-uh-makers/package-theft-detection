import plac
import pandas as pd
import random


@plac.annotations(
    csvpath=('Path to the sequence description CSV', 'option', 'c', str),
    tvsplit=('Percentage of training data', 'option', float)
)
def main(csvpath: str = 'sequence.csv',
         tvsplit: float = 0.8):

    # Repeatability
    random.setstate(0)

    df = pd.read_csv(csvpath)

    idx = [i for i in range(0, len(df))]
    random.shuffle(idx)

    train_start_idx = 0
    train_end_idx = int(tvsplit*len(idx))
    df.iloc[idx[train_start_idx:train_end_idx]].to_csv('csv/train.csv', index=False)

    if tvsplit != 1.0:
        val_start_idx = train_end_idx
        val_end_idx = len(idx)
        df.iloc[idx[val_start_idx:val_end_idx]].to_csv('csv/val.csv', index=False)


if __name__ == '__init__':
    plac.call(main)
