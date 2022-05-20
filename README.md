# KAOG

K-Associated Optimal Graph implemented using Python.

## Dependencies

All the project dependencies can be found
under [requirements.txt](https://github.com/Anakin86708/kaog/blob/master/requirements.txt).

## Installation

`pip install -e kaog`

## Running

Some examples can be found in the [main](https://github.com/Anakin86708/kaog/tree/master/main) directory.

## Using

The KAOG object only requires a dataset to work, containing also the label for each item. The label column is set as
default to `target`, but can be changed using `ColunaYSingleton().NOME_COLUNA_Y = *NAME*`.
If the dataset contains categorical data, the columns must be specified when creating the KAOG object.

--------
More documentation should be added later.
