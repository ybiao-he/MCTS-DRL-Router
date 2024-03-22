# Generating chip layouts for training

## Requirements

* Python3
* Numpy

## Running

Run by:

```
python3 main.py number_boards folder_to_save_generated_files whether_generating_PCB_for_testing
```

For example:

To generate PCB for testing:
```
python3 main.py 100 ./test_boards True
```

To generate PCB for training:
```
python3 main.py 100 ./training_boards False
```