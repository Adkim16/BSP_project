# BSP_project

Project for "Biomedical Signal Processing", Università degli Studi di Milano

## To run an experiment:
1) Install the requirements
2) Choose the dataset path, the output path, and the index of the sample to be processed in `MMF.py` main
3) *(optional)* choose a start and end index for the charts created by the script
4) Run `MMF.py`

The script will create a folder with the same name as the sample in the dataset, containing all the charts from every step of the pipeline, and a `results.csv` file containing metrics

## To process data from all experiments:
1) Choose the experiments folder, and the output folder in `process_results.py`
2) Run `process_results.py`

The script will create the output folder (if not present), containing the charts for all metrics (accuracy, precision, distance, BCR, NSR, SDR)
