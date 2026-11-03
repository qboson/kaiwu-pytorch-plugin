# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Spike-waveform dataset utilities for the spike-sorting reproduction.

The synthetic dataset is the one used by Ardelean et al. (2023), "A study of
autoencoders as a feature extraction technique for spike sorting", PLOS ONE,
doi:10.1371/journal.pone.0282810 (Pedreira et al., 2012 simulations).

Each ``simulation_<n>.mat`` stores a 10-minute 24 kHz extracellular recording
under the ``data`` key. ``ground_truth.mat`` stores cell arrays:

* ``spike_first_sample`` - first sample of every spike in the recording,
* ``spike_classes``      - unit label of every spike (0 marks the multi-unit),
* ``su_waveforms``       - the single-unit templates.

The paper extracts 79 samples per spike (316 samples at 96 kHz downsampled to
24 kHz), aligns the positive peak to sample index 39 and optionally z-scores
each waveform.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

SPIKE_LENGTH = 79
PEAK_INDEX = 39


@dataclass
class SimulationSpikes:
    """Spike waveforms and ground-truth labels of one simulation."""

    spikes: np.ndarray  # (n_spikes, 79) float32
    labels: np.ndarray  # (n_spikes,) int64, 0 = multi-unit cluster


def load_simulation_mat(path: str) -> dict:
    """Load a ``simulation_<n>.mat`` file."""
    return loadmat(path)


def load_ground_truth_mat(path: str) -> dict:
    """Load the shared ``ground_truth.mat`` file."""
    return loadmat(path)


def _cell(matrix, index: int) -> np.ndarray:
    """Read row ``index`` of a MATLAB cell array loaded by scipy."""
    return np.asarray(matrix[0][index]).ravel()


def extract_spikes(
    signal: np.ndarray,
    spike_start: np.ndarray,
    spike_length: int = SPIKE_LENGTH,
    align_to_peak: bool = True,
    peak_index: int = PEAK_INDEX,
    zscore: bool = False,
) -> np.ndarray:
    """Cut fixed-length waveforms out of a recording and align their peak.

    Args:
        signal: 1-D extracellular recording.
        spike_start: first sample of every spike in ``signal``.
        spike_length: number of samples per waveform (79 for this dataset).
        align_to_peak: when True, shift every spike so that its maximum falls
            on ``peak_index`` (the paper's ``align_to_peak=2`` mode).
        peak_index: sample index the positive peak is aligned to.
        zscore: optionally normalise every waveform to zero mean and unit
            standard deviation.

    Returns:
        Array of shape ``(n_spikes, spike_length)``.
    """
    starts = np.asarray(spike_start, dtype=np.int64).copy()
    spikes = np.zeros((len(starts), spike_length), dtype=np.float64)
    for row in range(len(starts)):
        spikes[row, :] = signal[starts[row]: starts[row] + spike_length]

    if align_to_peak:
        peaks = np.argmax(spikes, axis=1)
        starts = starts - (peak_index - peaks)
        for row in range(len(starts)):
            spikes[row, :] = signal[starts[row]: starts[row] + spike_length]

    if zscore:
        mean = spikes.mean(axis=1, keepdims=True)
        std = spikes.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        spikes = (spikes - mean) / std

    return spikes.astype(np.float32)


def load_simulation(
    simulation_number: int,
    data_dir: str,
    align_to_peak: bool = True,
    zscore: bool = False,
    max_spikes: int | None = None,
    seed: int = 0,
) -> SimulationSpikes:
    """Load one simulation as ``(spikes, labels)``.

    Args:
        simulation_number: 1-based simulation index (1..95).
        data_dir: directory holding ``simulation_<n>.mat`` and
            ``ground_truth.mat``.
        align_to_peak: align the positive peak to ``PEAK_INDEX``.
        zscore: z-score every waveform.
        max_spikes: optionally subsample spikes (useful for quick runs).
        seed: random seed used for subsampling.

    Returns:
        :class:`SimulationSpikes` with float32 waveforms and int64 labels.
    """
    sim_path = os.path.join(data_dir, f"simulation_{simulation_number}.mat")
    gt_path = os.path.join(data_dir, "ground_truth.mat")
    sim = loadmat(sim_path)
    gt = loadmat(gt_path)

    signal = np.asarray(sim["data"]).ravel()
    labels = _cell(gt["spike_classes"], simulation_number - 1).astype(np.int64)
    starts = _cell(gt["spike_first_sample"], simulation_number - 1).astype(np.int64)

    spikes = extract_spikes(
        signal,
        starts,
        align_to_peak=align_to_peak,
        zscore=zscore,
    )
    if max_spikes is not None and len(spikes) > max_spikes:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(spikes), size=max_spikes, replace=False)
        keep.sort()
        spikes = spikes[keep]
        labels = labels[keep]

    return SimulationSpikes(spikes=spikes, labels=labels)


def load_spike_folder(
    scenarios: list[int],
    base_dir: str,
    align_to_peak: bool = True,
    zscore: bool = False,
) -> dict:
    """Load every requested simulation once and return a mapping."""
    return {
        number: load_simulation(
            number,
            base_dir,
            align_to_peak=align_to_peak,
            zscore=zscore,
        )
        for number in scenarios
    }


# ---------------------------------------------------------------------------
# Real (``in vivo``) recording stored in the TINS "Spike Sorted Dataset" (SSD)
# container. The archive holds a text descriptor plus binary companions:
#
#   ``*.ssd``   - descriptor: unit names, electrodes, spike counts per unit
#   ``*.ssduw`` - float32 waveforms, concatenated unit after unit
#   ``*.ssdst`` - int32 first sample of every spike
#   ``*.ssdec`` - int32 manual-sorting decision codes
#
# The waveform length is inferred from the file size and the total spike
# count (58 samples for this recording).
# ---------------------------------------------------------------------------

REAL_PEAK_INDEX = 29


def parse_ssd_descriptor(path: str) -> dict:
    """Parse the text companion of an SSD recording.

    Args:
        path: path to the ``*.ssd`` descriptor.

    Returns:
        Mapping with ``unit_names``, ``electrodes`` and ``spike_counts``.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle]

    def block(start_marker: str) -> list[str]:
        start = lines.index(start_marker)
        values: list[str] = []
        for line in lines[start + 1:]:
            if not line:
                break
            values.append(line)
        return values

    unit_names = block("List with the names of units:")
    electrodes = block(
        "List with the recording elements (channels, tetrodes, etc) "
        "where each unit originates:"
    )
    counts = [int(value) for value in block("Number of spikes in each unit:")]
    return {
        "unit_names": unit_names,
        "electrodes": electrodes,
        "spike_counts": np.asarray(counts, dtype=np.int64),
    }


def load_realdata(
    ssd_dir: str,
    align_to_peak: bool = True,
    zscore: bool = False,
    max_spikes: int | None = None,
    seed: int = 0,
) -> SimulationSpikes:
    """Load the TINS real recording as ``(spikes, labels)``.

    The unit labels come from the manual sorting shipped with the dataset, so
    they act as a reference partition rather than a synthetic ground truth.

    Args:
        ssd_dir: directory containing the ``*.ssd``/``*.ssduw`` files.
        align_to_peak: align the positive peak to ``REAL_PEAK_INDEX``.
        zscore: z-score every waveform.
        max_spikes: optionally subsample spikes for quick runs.
        seed: random seed used for subsampling.

    Returns:
        :class:`SimulationSpikes` with float32 waveforms and int64 labels.
    """
    descriptor = None
    waveform_file = None
    for name in os.listdir(ssd_dir):
        if name.endswith(".ssd"):
            descriptor = os.path.join(ssd_dir, name)
        elif name.endswith(".ssduw"):
            waveform_file = os.path.join(ssd_dir, name)
    if descriptor is None or waveform_file is None:
        raise FileNotFoundError(f"no .ssd/.ssduw pair found in {ssd_dir}")

    meta = parse_ssd_descriptor(descriptor)
    counts = meta["spike_counts"]
    total = int(counts.sum())
    waveforms = np.fromfile(waveform_file, dtype=np.float32)
    if waveforms.size % total != 0:
        raise ValueError(
            f"waveform file has {waveforms.size} floats for {total} spikes"
        )
    length = waveforms.size // total
    waveforms = waveforms.reshape(total, length)

    labels = np.repeat(np.arange(len(counts), dtype=np.int64), counts)

    if align_to_peak:
        peaks = np.argmax(waveforms, axis=1)
        aligned = np.zeros_like(waveforms)
        for row, peak in enumerate(peaks):
            shift = REAL_PEAK_INDEX - int(peak)
            if shift == 0:
                aligned[row] = waveforms[row]
            elif shift > 0:
                aligned[row, shift:] = waveforms[row, :-shift]
            else:
                aligned[row, :shift] = waveforms[row, -shift:]
        waveforms = aligned

    if zscore:
        mean = waveforms.mean(axis=1, keepdims=True)
        std = waveforms.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        waveforms = (waveforms - mean) / std

    if max_spikes is not None and len(waveforms) > max_spikes:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(waveforms), size=max_spikes, replace=False)
        keep.sort()
        waveforms = waveforms[keep]
        labels = labels[keep]

    return SimulationSpikes(spikes=waveforms.astype(np.float32), labels=labels)
