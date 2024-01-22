import pandas as pd
import matplotlib.pyplot as plt
import os
from statistics import mean 
from ecgdetectors import Detectors
import numpy as np

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1, 1.15), loc='upper right')

def get_sparse_annotation(data_length, annotation):
    sync_ann = []

    for i in range(data_length):
        if i in annotation["Sample"].values:
            sync_ann.append(annotation[annotation.Sample == i]["#"].values[0])
        else:
            sync_ann.append(None)

    return sync_ann

def get_annotated_peaks_indexes(annotation):
    peaks = []

    for idx, ann in enumerate(annotation):
        if ann is not None:
            peaks.append(idx)
    
    return peaks

def get_detection_accuracy(a_peaks, d_peaks, tolerance=31):
    correct = 0
    distances = []
    for a_peak in a_peaks:
        for d_peak in d_peaks:
            if d_peak > a_peak+tolerance:
                break
            if d_peak >= a_peak-tolerance and d_peak <= a_peak+tolerance:
                correct += 1
                distances.append(a_peak - d_peak)
                break
    
    avg_distance = mean(distances)
    accuracy = correct / len(a_peaks)
    false_positives = len(d_peaks) - correct
    precision = correct / (correct + false_positives)

    return accuracy, precision, avg_distance

def save_ecg_plot(ecg, output_path, filename, a_peaks=None, d_peaks=None, start=0, end=None):
    fig, ax = plt.subplots()
    ax.plot(ecg[start:end])

    if a_peaks != None:
        for peak in a_peaks:
            if start <= peak <= end:
                ax.plot(peak-start, ecg[peak], "ro", label="Annotated peaks")

    if d_peaks != None:
        for peak in d_peaks:
            if start <= peak <= end:
                ax.axvline(peak-start, color="g", ls=":")
                ax.plot(peak-start, ecg[peak], "go", label="Detected peaks")

    if a_peaks != None and d_peaks != None:
        legend_without_duplicate_labels(ax)

    plt.savefig(output_path + filename)
    print("Chart saved as " + filename)

def generate_gaussian_noise(length):
    # set parameters
    epsilon = np.random.uniform(0.1, 0.5)
    sigma1 = np.random.uniform(0.01, 0.02)
    sigma2 = np.random.uniform(2 * sigma1, 5 * sigma1)
    snr = np.random.uniform(0.5, 1.5)
    
    # generate noise components
    noise1 = sigma1 * np.random.randn(length)
    noise2 = sigma2 * np.random.randn(length)
    
    # combine noise components
    noise = snr * ((1 - epsilon) * noise1 + epsilon * noise2)
    
    return noise

def generate_random_drift(length, slope=0.01, max_slope_change=0.001):
    # generate random slope change sequence
    slope_change_sequence = max_slope_change * np.random.randn(length)
    
    # compute the cumulative sum
    cumulative_slope_change = np.cumsum(slope_change_sequence)
    
    # apply variable slope change to initial slope
    slopes = slope + cumulative_slope_change

    time = np.arange(0, length)

    drift = slopes * time 

    return drift

def erosion(input_signal, strel):
    M = len(strel)
    N = len(input_signal)

    if M % 2 == 0:
        raise ValueError('strel length must be odd')

    hw = M // 2
    input_signal = np.pad(input_signal, (hw, hw), mode='edge')

    output = np.zeros(N)
    for n in range(hw, N + hw):
        output[n - hw] = np.min(input_signal[n - hw: n + hw + 1] - strel)
    
    return output

def dilation(input_signal, strel):
    M = len(strel)
    N = len(input_signal)
    
    if M % 2 == 0:
        raise ValueError('strel length must be odd')

    hw = M // 2
    input_signal = np.pad(input_signal, (hw, hw), mode='edge')

    output = np.zeros(N)
    for n in range(hw, N + hw):
        output[n - hw] = np.max(input_signal[n - hw: n + hw + 1] + strel)
    
    return output

def opening(input_signal, strel):
    eroded_signal = erosion(input_signal, strel)
    opened_signal = dilation(eroded_signal, strel)
    return opened_signal

def closing(input_signal, strel):
    dilated_signal = dilation(input_signal, strel)
    closed_signal = erosion(dilated_signal, strel)
    return closed_signal


if __name__ == "__main__":

    path = "C:/Users/Nico/Documents/UNIVERSITA/Biomedical Signal Processing/mitbih_csv/"
    output_folder = "BSP/"
    fs = 360
    sample_index = 0  # range [0, 47]
    # start and end indexes for charts
    start = 250
    end = 2500

    # read dataset files
    filenames = next(os.walk(path))[2]
    records = []
    annotations = []
    filenames.sort()

    for f in filenames:
        filename, file_extension = os.path.splitext(f)

        if(file_extension == '.csv'):
            records.append(path + filename + file_extension)
        else:
            annotations.append(path + filename + file_extension)

    sample_name = records[sample_index].split('/')[-1].split('.')[0]
    output_path = output_folder + sample_name + "/"

    if sample_name in ["102", "104", "214", "228", "231"]:
        raise ValueError("Data is not available for records 102, 104, 214, 228, and 231.")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"\nFolder \"{output_path}\" has been created.")

    print(f"\n---------- GETTING DATA FOR SAMPLE \"{sample_name}\" ----------\n")
    data = pd.read_csv(records[sample_index])
    annotation = pd.read_table(annotations[sample_index], sep='\s+')
    data["type"] = get_sparse_annotation(len(data), annotation)

    # get ecg and annotated peaks
    ecg = np.array(data["'MLII'"])
    annotated_peaks = get_annotated_peaks_indexes(data["type"])
    print(f"Signal duration: {len(ecg)} ({round(len(ecg) / fs, 2)} seconds)\n")
    if end >= len(ecg):
        raise ValueError("End index for charts is out of bounds for ECG signal.")
    save_ecg_plot(ecg=ecg, start=start, end=end, output_path=output_path, filename="original_ECG.png")
    print("-DONE")

    # normalize ECG signal
    print("\n---------- NORMALIZING ECG ----------\n")
    original_ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
    print("-DONE")

    # adding noise
    print("\n---------- ADDING NOISE TO ECG ----------\n")
    noise = generate_gaussian_noise(len(ecg))
    noise = original_ecg * noise
    ecg = original_ecg + noise
    save_ecg_plot(ecg=noise, start=start, end=end, output_path=output_path, filename="generated_noise.png")
    save_ecg_plot(ecg=ecg, start=start, end=end, output_path=output_path, filename="noisy_ECG.png")
    print("-DONE")

    # adding baseline drift
    print("\n---------- ADDING BASELINE DRIFT TO ECG ----------\n")
    baseline_drift = generate_random_drift(len(ecg))
    baseline_drift = (baseline_drift - np.min(baseline_drift)) / (np.max(baseline_drift) - np.min(baseline_drift))
    ecg = ecg + baseline_drift
    save_ecg_plot(ecg=baseline_drift, start=start, end=end, output_path=output_path, filename="generated_drift.png")
    save_ecg_plot(ecg=ecg, start=start, end=end, output_path=output_path, filename="drifted_ECG.png")
    print("-DONE")

    # detect r-peaks
    print("\n---------- DETECTING R-PEAKS ----------\n")
    detectors = Detectors(fs)
    r_peaks = detectors.matched_filter_detector(ecg)

    # get accuracy and average distance of detected peaks
    tolerance = 31  # threshold to count a detection as correct (plus or minus)
    accuracy, precision, avg_distance = get_detection_accuracy(a_peaks=annotated_peaks, d_peaks=r_peaks, tolerance=tolerance)
    results = pd.Series(index=["before_accuracy", "before_precision", "before_distance", "after_accuracy", "after_precision", "after_distance", "BCR", "NSR", "SDR"], dtype="float64")
    results["before_accuracy"], results["before_precision"], results["before_distance"] = round(accuracy*100, 2), round(precision*100, 2), round(avg_distance, 2)
    print(f"Accuracy: {results['before_accuracy']}%\nPrecision: {results['before_precision']}%\nAverage distance: {results['before_distance']} (from actual peak to detected peak)\nTolerance: {tolerance}\n")

    # plot ECG with annotated and detected peaks
    save_ecg_plot(ecg=ecg, a_peaks=annotated_peaks, d_peaks=r_peaks, start=start, end=end, output_path=output_path, filename="peaks_detected.png")
    print("-DONE")

    # baseline correction
    print("\n---------- PERFORMING BASELINE CORRECTION ----------\n")
    l0 = int(0.2 * fs) + 1
    l1 = int(l0 * 1.5)
    l1 = l1 if l1%2 == 1 else l1+1  # render l1 odd

    bo = np.ones(l0)
    bc = np.ones(l1)

    ecg_opening = opening(ecg, bo)
    detected_baseline_drift = closing(ecg_opening, bc)
    baseline_corrected_ecg = np.subtract(ecg, detected_baseline_drift)
    
    save_ecg_plot(ecg=detected_baseline_drift, start=start, end=end, output_path=output_path, filename="detected_baseline_drift.png")
    save_ecg_plot(ecg=baseline_corrected_ecg, start=start, end=end, output_path=output_path, filename="baseline_corrected_ECG.png")
    print("-DONE")

    # noise suppression
    print("\n---------- PERFORMING NOISE SUPPRESSION ----------\n")
    b1 = np.array([0, 1, 5, 1, 0])
    b2 = np.ones(5)

    noise_closing = erosion(dilation(baseline_corrected_ecg, b1), b2)
    noise_opening = dilation(erosion(baseline_corrected_ecg, b1), b2)
    noise_suppressed_ecg = (noise_closing + noise_opening) / 2

    save_ecg_plot(ecg=noise_suppressed_ecg, start=start, end=end, output_path=output_path, filename="noise_suppressed_ECG.png")
    print("-DONE")

    # detect r-peaks in baseline corrected and normalized ECG
    print("\n---------- DETECTING R-PEAKS AFTER MMF ----------\n")
    r_peaks = detectors.matched_filter_detector(baseline_corrected_ecg)

    # get accuracy and average distance of detected peaks
    tolerance = 31  # threshold to count a detection as correct (plus or minus)
    accuracy, precision, avg_distance = get_detection_accuracy(a_peaks=annotated_peaks, d_peaks=r_peaks, tolerance=tolerance)
    results["after_accuracy"], results["after_precision"], results["after_distance"] = round(accuracy*100, 2), round(precision*100, 2), round(avg_distance, 2)
    print(f"Accuracy: {results['after_accuracy']}%\nPrecision: {results['after_precision']}%\nAverage distance: {results['after_distance']} (from actual peak to detected peak)\nTolerance: {tolerance}\n")

    save_ecg_plot(ecg=noise_suppressed_ecg, a_peaks=annotated_peaks, d_peaks=r_peaks, start=start, end=end, output_path=output_path, filename="peaks_detected_MMF.png")
    print("-DONE")

    print("\n---------- COMPUTING BCR, NSR, and SDR ----------\n")
    # compute baseline correction ratio (BCR)
    sum_detected_drift = sum(abs(detected_baseline_drift))
    sum_baseline_drift = sum(abs(baseline_drift))
    bcr = round(sum_detected_drift / sum_baseline_drift, 2)
    results["BCR"] = bcr

    # compute noise suppression ratio (NSR)
    sum_noise_suppressed = sum(abs(noise_suppressed_ecg))
    sum_noise_added = sum(abs(noise))
    nsr = round(sum_noise_suppressed / sum_noise_added, 2)
    results["NSR"] = nsr

    # compute signal distortion ratio (SDR)
    difference = sum(abs(original_ecg - noise_suppressed_ecg))
    sum_filtered = sum(abs(noise_suppressed_ecg))
    sdr = round(difference / sum_filtered, 2)
    results["SDR"] = sdr
    print(f"BCR: {bcr}\nNSR: {nsr}\nSDR: {sdr}\n")
    print("-DONE")

    # save results
    results.to_csv(output_path + "results.csv", index=True, header=True)
