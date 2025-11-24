import os
import numpy as np
import pandas as pd
from scipy import signal
import pyedflib
import mne
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings('ignore')

class CHBMITCorrectPreprocessor:
    def __init__(self, data_path, target_fs=256, window_size=10, stride=5, 
                 pre_ictal_horizon=300, prediction_horizon=30, artifact_threshold=0.3):
        """
        Correct Preprocessor for CHB-MIT Scalp EEG Database
        """
        self.data_path = data_path
        self.target_fs = target_fs
        self.window_size = window_size
        self.stride = stride
        self.pre_ictal_horizon = pre_ictal_horizon
        self.prediction_horizon = prediction_horizon
        self.artifact_threshold = artifact_threshold
        
        # Standard channels based on CHB-MIT documentation
        self.standard_channels = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ'
        ]
        
    def get_subject_files(self, subject_id):
        """Get all EDF files and annotation files for a subject"""
        subject_path = os.path.join(self.data_path, subject_id)
        edf_files = []
        seizure_files = []
        
        for file in os.listdir(subject_path):
            if file.endswith('.edf'):
                edf_files.append(os.path.join(subject_path, file))
            elif 'summary' in file.lower() and file.endswith('.txt'):
                seizure_files.append(os.path.join(subject_path, file))
            elif file.endswith('.seizures'):
                seizure_files.append(os.path.join(subject_path, file))
        
        return sorted(edf_files), sorted(seizure_files)
    
    def parse_seizure_annotations(self, seizure_files):
        """Parse seizure annotations from summary files"""
        seizure_info = {}
        
        for seizure_file in seizure_files:
            try:
                with open(seizure_file, 'r') as f:
                    content = f.readlines()
                
                current_file = None
                for line in content:
                    line = line.strip()
                    if line.startswith('File Name:'):
                        current_file = line.split('File Name:')[1].strip()
                        seizure_info[current_file] = []
                    elif 'Seizure Start' in line and 'seconds' in line:
                        # Extract times from line
                        parts = line.split()
                        start_time = None
                        end_time = None
                        
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                num = int(part)
                                if start_time is None:
                                    start_time = num
                                elif end_time is None and num > start_time:
                                    end_time = num
                                    break
                        
                        if start_time is not None and end_time is not None:
                            seizure_info[current_file].append({
                                'start': start_time,
                                'end': end_time
                            })
                            
            except Exception as e:
                print(f"Warning: Could not parse {seizure_file}: {e}")
                continue
        
        return seizure_info
    
    def load_and_standardize_edf(self, file_path):
        """Load EDF file with robust channel handling"""
        try:
            # Load with MNE
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            print(f"    Original channels: {len(raw.ch_names)} channels")
            
            # Create a clean mapping
            clean_mapping = {}
            used_names = set()
            
            for orig_ch in raw.ch_names:
                # Clean the channel name
                clean_ch = orig_ch.upper().replace('EEG', '').replace('REF', '').replace('_', '-').strip()
                clean_ch = clean_ch.replace('--', '-').strip('-')
                
                # Map to standard names - handle T8-P8 duplicates
                if 'T8-P8' in clean_ch:
                    if '-0' in orig_ch or 'T8-P8-0' in clean_ch:
                        mapped_ch = 'T8-P8'
                    elif '-1' in orig_ch or 'T8-P8-1' in clean_ch:
                        mapped_ch = 'T8-P8-2'  # Give it a different name
                    else:
                        mapped_ch = 'T8-P8'
                else:
                    # Map other channels
                    mapped_ch = None
                    for std_ch in self.standard_channels:
                        if std_ch in clean_ch or clean_ch in std_ch:
                            mapped_ch = std_ch
                            break
                    
                    if mapped_ch is None:
                        mapped_ch = clean_ch
                
                # Handle duplicates
                base_name = mapped_ch
                suffix = 1
                while mapped_ch in used_names:
                    mapped_ch = f"{base_name}_{suffix}"
                    suffix += 1
                
                clean_mapping[orig_ch] = mapped_ch
                used_names.add(mapped_ch)
            
            # Apply mapping
            raw.rename_channels(clean_mapping)
            
            # Pick only standard channels that exist
            available_standard_chs = []
            for ch in raw.ch_names:
                for std_ch in self.standard_channels:
                    if std_ch in ch:
                        available_standard_chs.append(ch)
                        break
            
            if len(available_standard_chs) < 10:
                print(f"    Using available channels: {len(available_standard_chs)} channels")
            
            if len(available_standard_chs) < 8:
                print(f"    Too few channels ({len(available_standard_chs)}), skipping file")
                return None
            
            raw.pick_channels(available_standard_chs)
            print(f"    Using {len(raw.ch_names)} channels")
            
            return raw
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
    
    def apply_common_average_reference(self, raw):
        """Apply Common Average Reference (CAR)"""
        try:
            raw_car = raw.copy()
            raw_car.set_eeg_reference('average', projection=True)
            raw_car.apply_proj()
            return raw_car
        except Exception as e:
            print(f"CAR failed: {e}, using original data")
            return raw
    
    def apply_filters(self, raw):
        """Apply bandpass and notch filters using MNE"""
        try:
            # Bandpass filter 0.5-45 Hz
            raw.filter(0.5, 45., fir_design='firwin', verbose=False)
            
            # Notch filter 60 Hz
            raw.notch_filter(60., fir_design='firwin', verbose=False)
            
            return raw
        except Exception as e:
            print(f"Filtering failed: {e}")
            return raw
    
    def remove_artifacts_clinical(self, raw):
        """Clinical artifact removal with appropriate thresholds"""
        try:
            data = raw.get_data()
            fs = raw.info['sfreq']
            n_channels, n_samples = data.shape
            
            # CORRECTED: Clinical EEG artifact thresholds (in volts)
            AMPLITUDE_THRESHOLD = 1000e-6   # 1000 µV - normal EEG can reach 300-500 µV
            JUMP_THRESHOLD = 500e-6         # 500 µV between samples
            FLATLINE_THRESHOLD = 1e-6       # 1 µV range for flatline detection
            
            # Create artifact mask
            artifact_mask = np.zeros(n_samples, dtype=bool)
            
            # Process each time sample
            for i in range(n_samples):
                sample_data = data[:, i]
                
                # 1. Amplitude artifact - check if any channel exceeds threshold
                if np.max(np.abs(sample_data)) > AMPLITUDE_THRESHOLD:
                    artifact_mask[i] = True
                    continue
                
                # 2. Flatline detection - check if signal is flat
                if np.max(sample_data) - np.min(sample_data) < FLATLINE_THRESHOLD:
                    artifact_mask[i] = True
                    continue
            
            # 3. Jump detection - check for sudden changes between samples
            if n_samples > 1:
                for ch_idx in range(n_channels):
                    diffs = np.abs(np.diff(data[ch_idx]))
                    jump_positions = np.where(diffs > JUMP_THRESHOLD)[0]
                    # Mark regions around jumps
                    for pos in jump_positions:
                        start = max(0, pos - int(0.05 * fs))  # 50ms before jump
                        end = min(n_samples, pos + int(0.05 * fs))  # 50ms after jump
                        artifact_mask[start:end] = True
            
            artifact_percentage = np.sum(artifact_mask) / n_samples * 100
            if artifact_percentage > 5:  # Only report if significant
                print(f"    Found {artifact_percentage:.1f}% artifact samples")
            else:
                print(f"    Clean signal: {artifact_percentage:.1f}% artifacts")
            
            return raw, artifact_mask
            
        except Exception as e:
            print(f"Artifact removal failed: {e}")
            # Return no artifacts if detection fails
            return raw, np.zeros(raw.get_data().shape[1], dtype=bool)
    
    def resample_to_target_fs(self, raw):
        """Resample to target frequency"""
        if raw.info['sfreq'] != self.target_fs:
            try:
                raw.resample(self.target_fs, verbose=False)
                print(f"    Resampled to {self.target_fs} Hz")
            except Exception as e:
                print(f"Resampling failed: {e}")
        return raw
    
    def create_labeled_windows(self, raw, seizure_times, file_duration, artifact_mask):
        """Create windows with labels"""
        try:
            data = raw.get_data()
            n_channels, n_samples = data.shape
            fs = raw.info['sfreq']
            
            window_samples = int(self.window_size * fs)
            stride_samples = int(self.stride * fs)
            
            windows = []
            detection_labels = []
            prediction_labels = []
            metadata = []
            
            # Skip files with excessive artifacts (>50%)
            artifact_percentage = np.sum(artifact_mask) / n_samples
            if artifact_percentage > 0.5:
                print(f"    Skipping file: {artifact_percentage*100:.1f}% artifacts")
                return np.array([]), np.array([]), np.array([]), []
            
            window_count = 0
            for start_idx in range(0, n_samples - window_samples + 1, stride_samples):
                end_idx = start_idx + window_samples
                timestamp = start_idx / fs
                
                # Skip if window contains too many artifacts
                window_artifact_ratio = np.sum(artifact_mask[start_idx:end_idx]) / window_samples
                if window_artifact_ratio > self.artifact_threshold:
                    continue
                
                # Initialize labels
                det_label = 0  # Detection (ictal)
                pred_label = 0  # Prediction (pre-ictal)
                
                # Check if window contains seizure (ictal)
                for seizure in seizure_times:
                    seizure_start = seizure['start']
                    seizure_end = seizure['end']
                    
                    window_start = timestamp
                    window_end = timestamp + self.window_size
                    
                    if (seizure_start <= window_end and seizure_end >= window_start):
                        det_label = 1
                        pred_label = 1
                        break
                
                # Check if window is pre-ictal
                if det_label == 0:
                    for seizure in seizure_times:
                        seizure_start = seizure['start']
                        pre_ictal_start = max(0, seizure_start - self.pre_ictal_horizon)
                        pre_ictal_end = seizure_start - self.prediction_horizon
                        
                        if pre_ictal_start <= timestamp < pre_ictal_end:
                            pred_label = 1
                            break
                
                # Apply normalization
                window_data = data[:, start_idx:end_idx].astype(np.float32)
                normalized_window = self.z_score_normalize(window_data)
                
                if not np.any(np.isnan(normalized_window)) and not np.any(np.isinf(normalized_window)):
                    windows.append(normalized_window)
                    detection_labels.append(det_label)
                    prediction_labels.append(pred_label)
                    
                    metadata.append({
                        'timestamp': timestamp,
                        'file_duration': file_duration,
                        'is_ictal': det_label,
                        'is_preictal': pred_label,
                        'artifact_ratio': window_artifact_ratio
                    })
                    window_count += 1
            
            print(f"    Created {window_count} valid windows")
            return (np.array(windows), np.array(detection_labels), 
                    np.array(prediction_labels), metadata)
            
        except Exception as e:
            print(f"Window creation failed: {e}")
            return np.array([]), np.array([]), np.array([]), []
    
    def z_score_normalize(self, window_data):
        """Channel-wise z-score normalization"""
        normalized = np.zeros_like(window_data)
        for ch_idx in range(window_data.shape[0]):
            channel_data = window_data[ch_idx]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            if std_val > 0:
                normalized[ch_idx] = (channel_data - mean_val) / std_val
            else:
                normalized[ch_idx] = channel_data - mean_val
        
        return normalized
    
    def preprocess_subject(self, subject_id):
        """Preprocess a single subject"""
        print(f"\n=== Preprocessing subject {subject_id} ===")
        
        edf_files, seizure_files = self.get_subject_files(subject_id)
        seizure_info = self.parse_seizure_annotations(seizure_files)
        
        all_windows = []
        all_detection_labels = []
        all_prediction_labels = []
        all_metadata = []
        
        for edf_file in edf_files:
            file_name = os.path.basename(edf_file)
            print(f"  Processing {file_name}...")
            
            # Load EDF file
            raw = self.load_and_standardize_edf(edf_file)
            if raw is None:
                continue
            
            # Get file duration
            file_duration = raw.n_times / raw.info['sfreq']
            
            # Apply preprocessing pipeline
            raw = self.apply_filters(raw)
            raw = self.apply_common_average_reference(raw)
            
            # CORRECTED: Clinical artifact removal
            raw, artifact_mask = self.remove_artifacts_clinical(raw)
            
            # Resample to target frequency
            raw = self.resample_to_target_fs(raw)
            
            # Get seizure times for this file
            file_seizure_times = seizure_info.get(file_name, [])
            
            # Create labeled windows
            windows, det_labels, pred_labels, metadata = self.create_labeled_windows(
                raw, file_seizure_times, file_duration, artifact_mask
            )
            
            if len(windows) > 0:
                all_windows.append(windows)
                all_detection_labels.append(det_labels)
                all_prediction_labels.append(pred_labels)
                all_metadata.extend(metadata)
        
        # Concatenate all files
        if all_windows:
            all_windows = np.concatenate(all_windows, axis=0)
            all_detection_labels = np.concatenate(all_detection_labels, axis=0)
            all_prediction_labels = np.concatenate(all_prediction_labels, axis=0)
            
            total_windows = len(all_windows)
            ictal_windows = np.sum(all_detection_labels)
            preictal_windows = np.sum(all_prediction_labels)
            preictal_only = preictal_windows - ictal_windows
            interictal_windows = total_windows - ictal_windows - preictal_only
            
            print(f"  ✅ Subject {subject_id} complete:")
            print(f"     Total windows: {total_windows}")
            print(f"     Ictal windows: {ictal_windows}")
            print(f"     Pre-ictal windows: {preictal_windows}")
            print(f"     Interictal windows: {interictal_windows}")
            
            return {
                'windows': all_windows,
                'detection_labels': all_detection_labels,
                'prediction_labels': all_prediction_labels,
                'metadata': all_metadata
            }
        else:
            print(f"  ❌ Subject {subject_id}: No valid windows")
            return None
    
    def preprocess_all_subjects(self, output_dir, subjects_to_process=None):
        """Preprocess all subjects"""
        os.makedirs(output_dir, exist_ok=True)
        
        if subjects_to_process is None:
            subjects_to_process = [d for d in os.listdir(self.data_path) if d.startswith('chb')]
        
        all_subjects_data = {}
        subject_stats = {}
        
        for subject_id in subjects_to_process:
            subject_path = os.path.join(self.data_path, subject_id)
            if not os.path.isdir(subject_path):
                continue
                
            subject_data = self.preprocess_subject(subject_id)
            
            if subject_data is not None:
                all_subjects_data[subject_id] = subject_data
                
                # Save subject data
                output_file = os.path.join(output_dir, f"{subject_id}_processed.npz")
                np.savez_compressed(output_file, 
                                  windows=subject_data['windows'],
                                  detection_labels=subject_data['detection_labels'],
                                  prediction_labels=subject_data['prediction_labels'],
                                  metadata=subject_data['metadata'])
                
                # Calculate statistics
                total_windows = len(subject_data['windows'])
                ictal_windows = np.sum(subject_data['detection_labels'])
                preictal_windows = np.sum(subject_data['prediction_labels'])
                preictal_only = preictal_windows - ictal_windows
                interictal_windows = total_windows - ictal_windows - preictal_only
                
                subject_stats[subject_id] = {
                    'total_windows': total_windows,
                    'seizure_windows': ictal_windows,
                    'preictal_windows': preictal_windows,
                    'preictal_only_windows': preictal_only,
                    'interictal_windows': interictal_windows,
                    'seizure_ratio': ictal_windows / total_windows,
                    'preictal_ratio': preictal_windows / total_windows
                }
                
                print(f"  💾 Saved {output_file}")
        
        # Save statistics
        if subject_stats:
            stats_df = pd.DataFrame.from_dict(subject_stats, orient='index')
            stats_df.to_csv(os.path.join(output_dir, 'preprocessing_statistics.csv'))
            print(f"\n📊 Statistics saved to {os.path.join(output_dir, 'preprocessing_statistics.csv')}")
        
        print(f"\n🎉 Preprocessing completed!")
        print(f"📁 Processed {len(all_subjects_data)} subjects")
        print(f"💾 Data saved to {output_dir}")
        
        return all_subjects_data, subject_stats

# Run the preprocessing
if __name__ == "__main__":
    DATA_PATH = "chb-mit-scalp-eeg-database-1.0.0"
    OUTPUT_DIR = "processed_data_correct"
    
    preprocessor = CHBMITCorrectPreprocessor(
        data_path=DATA_PATH,
        target_fs=256,
        window_size=10,
        stride=5,
        pre_ictal_horizon=300,
        prediction_horizon=30,
        artifact_threshold=0.3
    )
    
    # Test with just one subject first
    print("🧪 Testing with chb01...")
    subject_data = preprocessor.preprocess_subject('chb01')
    
    if subject_data is not None:
        print(f"✅ Successfully processed chb01 with {len(subject_data['windows'])} windows")
        
        # Now process all subjects
        print("\n🔄 Processing all subjects...")
        all_data, stats = preprocessor.preprocess_all_subjects(OUTPUT_DIR)
    else:
        print("❌ Failed to process chb01")