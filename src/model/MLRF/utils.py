
import numpy as np


def event_detector(PSD, threshold_factor=1.05**-1):
    """
    Detect events in PSD data where signal exceeds a threshold relative to the signal mean
    """
    window_size = 64
    event_ranges = []
    in_event = False
    start_idx = 0

    # Calculate mean of the entire PSD
    signal_mean = np.mean(PSD)

    # Set threshold relative to the mean
    dynamic_threshold = signal_mean * threshold_factor

    i = 0
    while i < len(PSD) - window_size + 1:
        window = PSD[i : i + window_size]
        window_average = round(sum(window) / window_size, 2)

        if window_average > dynamic_threshold and not in_event:
            start_idx = i
            in_event = True
            i += window_size
        elif window_average <= dynamic_threshold and in_event:
            event_ranges.append((start_idx, i - 1))
            in_event = False
            i += window_size
        else:
            i += window_size

    if in_event:
        event_ranges.append((start_idx, len(PSD) - 1))
    elif not event_ranges:  # Only add (0,0) if no events were found
        event_ranges.append((0, 0))

    return event_ranges


def get_chunk_indices(total_size, num_chunks):
    """Split data into roughly equal chunks for multiprocessing"""
    chunk_size = total_size // num_chunks
    indices = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else total_size
        indices.append((start, end))
    return indices
