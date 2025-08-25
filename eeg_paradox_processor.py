#!/usr/bin/env python3
"""
EEG Paradox Rapid Reporter - Cuban Normative Database Processor
Enhanced for Clinical QEEG Analysis (Jay Gunkelman & Mark Jones Standards)
Processes Cuban EEG normative data to create comprehensive clinical QEEG database
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set dark mode styling for plots
plt.style.use('dark_background')
sns.set_palette("husl")

class CubanEEGProcessor:
    """Process Cuban EEG normative database and create clinical QEEG tables"""
    
    def __init__(self, data_path="."):
        self.data_path = Path(data_path)
        self.ec_path = self.data_path / "EyesClose"
        self.eo_path = self.data_path / "EyesOpen"
        
        # EEG electrode names (10-20 system) - Clinical standard order
        self.electrodes = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        ]
        
        # Enhanced frequency bands (clinical QEEG standard)
        self.freq_bands = {
            'delta': (0.5, 3.5),
            'theta': (4.0, 7.5),
            'alpha': (8.0, 12.0),
            'beta1': (12.5, 15.5),
            'beta2': (15.5, 18.5),
            'beta3': (18.5, 21.5),
            'beta4': (21.5, 30.0),
            'gamma': (30.0, 44.0)
        }
        
        # Clinical frequency bands (Gunkelman/Jones standard)
        self.clinical_bands = {
            'delta': (0.5, 3.5),
            'theta': (4.0, 7.5),
            'alpha': (8.0, 12.0),
            'beta': (12.5, 30.0),
            'gamma': (30.0, 44.0)
        }
        
        # Frequencies from 0.39 to 19.11 Hz (49 points)
        self.frequencies = np.linspace(0.39, 19.11, 49)
        
        # Clinical significance thresholds
        self.z_thresholds = {
            'normal': (-1.96, 1.96),
            'borderline': (-2.58, -1.96),
            'abnormal': (-3.29, -2.58),
            'severely_abnormal': (-3.29, float('-inf'))
        }
        
        self.subjects_data = {}
        self.normative_data = {}
        
    def load_subject_data(self, file_path):
        """Load individual subject data from .mat file with robust error handling"""
        try:
            data = sio.loadmat(file_path)
            
            # Debug: Print the structure of the loaded data
            print(f"Debug - {file_path.name} keys: {list(data.keys())}")
            for key in data.keys():
                if not key.startswith('__'):
                    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
            
            # Handle different possible variable names and structures
            age = None
            sex = None
            mcross = None
            
            # Try different possible variable names for age
            for age_key in ['age', 'Age', 'AGE']:
                if age_key in data:
                    age_data = data[age_key]
                    if age_data.ndim == 1:
                        age = int(age_data[0])
                    elif age_data.ndim == 2:
                        age = int(age_data[0, 0])
                    else:
                        age = int(age_data.flatten()[0])
                    break
            
            # Try different possible variable names for sex
            for sex_key in ['sex', 'Sex', 'SEX']:
                if sex_key in data:
                    sex_data = data[sex_key]
                    if sex_data.ndim == 1:
                        sex_raw = sex_data[0]
                    elif sex_data.ndim == 2:
                        sex_raw = sex_data[0, 0]
                    else:
                        sex_raw = sex_data.flatten()[0]
                    
                    # Convert sex to numeric (1 for M, 2 for F, 0 for unknown)
                    if sex_raw == 'M' or sex_raw == b'M':
                        sex = 1
                    elif sex_raw == 'F' or sex_raw == b'F':
                        sex = 2
                    else:
                        try:
                            sex = int(sex_raw)  # In case it's already numeric
                        except:
                            sex = 0  # Unknown
                    break
            
            # Try different possible variable names for MCross
            for mcross_key in ['MCross', 'Mcross', 'mcross', 'Mcr']:
                if mcross_key in data:
                    mcross_data = data[mcross_key]
                    if mcross_data.ndim == 3 and mcross_data.shape[0] == 19:
                        mcross = mcross_data
                        break
            
            # Validate that we have all required data
            if age is None or sex is None or mcross is None:
                print(f"Warning: Missing required data in {file_path.name}")
                print(f"  age: {age}, sex: {sex}, mcross shape: {mcross.shape if mcross is not None else None}")
                return None
            
            return {
                'age': age,
                'sex': sex,
                'MCross': mcross
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def average_reference_transform(self, data):
        """Transform data to average reference (equivalent to avr_ref.m)"""
        nd, _, nf = data.shape
        H = np.eye(nd) - np.ones((nd, nd)) / nd
        
        transformed = np.zeros_like(data)
        for w in range(nf):
            R = data[:, :, w]
            transformed[:, :, w] = H @ R @ H.T
        
        return transformed
    
    def global_scale_factor_correction(self, data):
        """Apply Global Scale Factor correction (equivalent to gsf.m)"""
        nd, _, nf = data.shape
        
        sum_log = 0
        n_freq = 0
        
        for w in range(nf):
            tmp = data[:, :, w]
            if np.sum(tmp) != 0:
                n_freq += 1
                diag_vals = np.real(np.diag(tmp))
                positive_indices = diag_vals > 0
                if np.any(positive_indices):
                    sum_log += np.sum(np.log(diag_vals[positive_indices]))
        
        if n_freq > 0:
            factor = np.exp(sum_log / (n_freq * nd))
            return data / factor, factor
        else:
            return data, 1.0
    
    def extract_power_spectra(self, data):
        """Extract power spectra from cross-spectral matrix"""
        # Diagonal elements contain power spectra
        power_spectra = np.real(np.diagonal(data, axis1=0, axis2=1))
        return power_spectra.T  # Shape: (19 electrodes, 49 frequencies)
    
    def calculate_band_powers(self, power_spectra):
        """Calculate power in different frequency bands"""
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find frequency indices within band
            band_indices = np.where((self.frequencies >= low_freq) & 
                                   (self.frequencies <= high_freq))[0]
            
            if len(band_indices) > 0:
                # Average power across frequencies in band
                band_powers[band_name] = np.mean(power_spectra[:, band_indices], axis=1)
            else:
                band_powers[band_name] = np.zeros(19)
        
        return band_powers
    
    def calculate_clinical_metrics(self, power_spectra):
        """Calculate clinical QEEG metrics (Gunkelman/Jones standard)"""
        clinical_metrics = {}
        
        # 1. Peak Alpha Frequency (PAF)
        alpha_range = np.where((self.frequencies >= 8.0) & (self.frequencies <= 12.0))[0]
        if len(alpha_range) > 0:
            alpha_power = power_spectra[:, alpha_range]
            paf_indices = np.argmax(alpha_power, axis=1)
            clinical_metrics['peak_alpha_freq'] = self.frequencies[alpha_range[paf_indices]]
        else:
            clinical_metrics['peak_alpha_freq'] = np.full(19, np.nan)
        
        # 2. Alpha Asymmetry (F3-F4, T3-T4, P3-P4, O1-O2)
        asymmetry_pairs = [('F3', 'F4'), ('T3', 'T4'), ('P3', 'P4'), ('O1', 'O2')]
        for left, right in asymmetry_pairs:
            left_idx = self.electrodes.index(left)
            right_idx = self.electrodes.index(right)
            
            # Alpha power asymmetry
            alpha_power_left = np.mean(power_spectra[left_idx, alpha_range])
            alpha_power_right = np.mean(power_spectra[right_idx, alpha_range])
            
            if alpha_power_right > 0:
                asymmetry = (alpha_power_left - alpha_power_right) / alpha_power_right
            else:
                asymmetry = np.nan
            
            clinical_metrics[f'alpha_asymmetry_{left}_{right}'] = asymmetry
        
        # 3. Theta/Beta Ratio (clinical standard)
        theta_power = np.mean(power_spectra[:, np.where((self.frequencies >= 4.0) & (self.frequencies <= 7.5))[0]], axis=1)
        beta_power = np.mean(power_spectra[:, np.where((self.frequencies >= 12.5) & (self.frequencies <= 30.0))[0]], axis=1)
        
        theta_beta_ratio = np.divide(theta_power, beta_power, out=np.full_like(theta_power, np.nan), where=beta_power != 0)
        clinical_metrics['theta_beta_ratio'] = theta_beta_ratio
        
        # 4. Beta/Alpha Ratio
        alpha_power = np.mean(power_spectra[:, alpha_range], axis=1)
        beta_alpha_ratio = np.divide(beta_power, alpha_power, out=np.full_like(beta_power, np.nan), where=alpha_power != 0)
        clinical_metrics['beta_alpha_ratio'] = beta_alpha_ratio
        
        # 5. Total Power
        clinical_metrics['total_power'] = np.sum(power_spectra, axis=1)
        
        # 6. Relative Power (percentage of total)
        for band_name, (low_freq, high_freq) in self.clinical_bands.items():
            band_indices = np.where((self.frequencies >= low_freq) & (self.frequencies <= high_freq))[0]
            if len(band_indices) > 0:
                band_power = np.mean(power_spectra[:, band_indices], axis=1)
                relative_power = np.divide(band_power, clinical_metrics['total_power'], 
                                         out=np.full_like(band_power, np.nan), 
                                         where=clinical_metrics['total_power'] != 0) * 100
                clinical_metrics[f'relative_{band_name}_power'] = relative_power
        
        return clinical_metrics
    
    def process_all_subjects(self):
        """Process all subjects and create normative database"""
        print("🔍 Processing Cuban EEG Normative Database...")
        
        # Process Eyes Closed data
        print("📁 Processing Eyes Closed data...")
        for file_path in self.ec_path.glob("*.mat"):
            subject_id = file_path.stem
            data = self.load_subject_data(file_path)
            if data:
                # Apply preprocessing
                data['MCross_avr'] = self.average_reference_transform(data['MCross'])
                data['MCross_gsf'], data['gsf_factor'] = self.global_scale_factor_correction(data['MCross_avr'])
                
                # Extract features
                data['power_spectra'] = self.extract_power_spectra(data['MCross_gsf'])
                data['band_powers'] = self.calculate_band_powers(data['power_spectra'])
                data['clinical_metrics'] = self.calculate_clinical_metrics(data['power_spectra'])
                
                self.subjects_data[subject_id] = data
        
        # Process Eyes Open data
        print("📁 Processing Eyes Open data...")
        for file_path in self.eo_path.glob("*.mat"):
            subject_id = file_path.stem
            data = self.load_subject_data(file_path)
            if data:
                # Apply preprocessing
                data['MCross_avr'] = self.average_reference_transform(data['MCross'])
                data['MCross_gsf'], data['gsf_factor'] = self.global_scale_factor_correction(data['MCross_avr'])
                
                # Extract features
                data['power_spectra'] = self.extract_power_spectra(data['MCross_gsf'])
                data['band_powers'] = self.calculate_band_powers(data['power_spectra'])
                data['clinical_metrics'] = self.calculate_clinical_metrics(data['power_spectra'])
                
                self.subjects_data[subject_id] = data
        
        print(f"✅ Processed {len(self.subjects_data)} subjects")
    
    def create_normative_database(self):
        """Create normative database by age groups"""
        if len(self.subjects_data) == 0:
            print("⚠️ No subjects processed, skipping normative database creation")
            return
            
        print(" Creating normative database...")
        
        # Group subjects by age
        age_groups = {}
        for subject_id, data in self.subjects_data.items():
            age = data['age']
            age_group = self._get_age_group(age)
            
            if age_group not in age_groups:
                age_groups[age_group] = []
            age_groups[age_group].append(data)
        
        # Calculate normative statistics for each age group
        for age_group, subjects in age_groups.items():
            if len(subjects) < 3:  # Need at least 3 subjects for statistics
                continue
                
            print(f"📈 Processing age group: {age_group} ({len(subjects)} subjects)")
            
            # Collect all data for this age group
            all_power_spectra = []
            all_band_powers = {band: [] for band in self.freq_bands.keys()}
            all_clinical_metrics = {metric: [] for metric in ['peak_alpha_freq', 'theta_beta_ratio', 'beta_alpha_ratio']}
            
            for subject in subjects:
                all_power_spectra.append(subject['power_spectra'])
                for band in all_band_powers:
                    all_band_powers[band].append(subject['band_powers'][band])
                
                # Clinical metrics
                for metric in all_clinical_metrics:
                    if metric in subject['clinical_metrics']:
                        all_clinical_metrics[metric].append(subject['clinical_metrics'][metric])
            
            # Convert to numpy arrays
            all_power_spectra = np.array(all_power_spectra)  # (n_subjects, 19, 49)
            
            # Calculate normative statistics
            self.normative_data[age_group] = {
                'n_subjects': len(subjects),
                'ages': [s['age'] for s in subjects],
                'sexes': [s['sex'] for s in subjects],
                'power_spectra_mean': np.mean(all_power_spectra, axis=0),
                'power_spectra_std': np.std(all_power_spectra, axis=0),
                'band_powers_mean': {},
                'band_powers_std': {},
                'clinical_metrics_mean': {},
                'clinical_metrics_std': {}
            }
            
            for band in all_band_powers:
                band_data = np.array(all_band_powers[band])  # (n_subjects, 19)
                self.normative_data[age_group]['band_powers_mean'][band] = np.mean(band_data, axis=0)
                self.normative_data[age_group]['band_powers_std'][band] = np.std(band_data, axis=0)
            
            for metric in all_clinical_metrics:
                if all_clinical_metrics[metric]:  # Check if we have data
                    metric_data = np.array(all_clinical_metrics[metric])  # (n_subjects, 19)
                    self.normative_data[age_group]['clinical_metrics_mean'][metric] = np.mean(metric_data, axis=0)
                    self.normative_data[age_group]['clinical_metrics_std'][metric] = np.std(metric_data, axis=0)
        
        print(f"✅ Created normative database for {len(self.normative_data)} age groups")
    
    def _get_age_group(self, age):
        """Group ages into appropriate categories"""
        if age <= 15:
            return f"{age//1*1}-{(age//1+1)*1-0.1}"
        elif age <= 19:
            return f"{age//2*2}-{(age//2+1)*2-0.1}"
        else:
            return f"{age//5*5}-{(age//5+1)*5-0.1}"
    
    def calculate_z_scores(self, subject_data, condition='EC'):
        """Calculate z-scores for a subject relative to age-matched normative data"""
        age = subject_data['age']
        age_group = self._get_age_group(age)
        
        if age_group not in self.normative_data:
            return None
        
        normative = self.normative_data[age_group]
        
        # Calculate z-scores for power spectra
        z_scores_spectra = (subject_data['power_spectra'] - normative['power_spectra_mean']) / normative['power_spectra_std']
        
        # Calculate z-scores for band powers
        z_scores_bands = {}
        for band in self.freq_bands.keys():
            if band in normative['band_powers_mean']:
                z_scores_bands[band] = (subject_data['band_powers'][band] - 
                                       normative['band_powers_mean'][band]) / normative['band_powers_std'][band]
        
        # Calculate z-scores for clinical metrics
        z_scores_clinical = {}
        for metric in ['peak_alpha_freq', 'theta_beta_ratio', 'beta_alpha_ratio']:
            if metric in normative['clinical_metrics_mean']:
                z_scores_clinical[metric] = (subject_data['clinical_metrics'][metric] - 
                                            normative['clinical_metrics_mean'][metric]) / normative['clinical_metrics_std'][metric]
        
        return {
            'age_group': age_group,
            'condition': condition,
            'z_scores_spectra': z_scores_spectra,
            'z_scores_bands': z_scores_bands,
            'z_scores_clinical': z_scores_clinical
        }
    
    def classify_abnormality(self, z_score):
        """Classify z-score into clinical categories"""
        if -1.96 <= z_score <= 1.96:
            return 'Normal'
        elif -2.58 <= z_score < -1.96 or 1.96 < z_score <= 2.58:
            return 'Borderline'
        elif -3.29 <= z_score < -2.58 or 2.58 < z_score <= 3.29:
            return 'Abnormal'
        elif z_score < -3.29 or z_score > 3.29:
            return 'Severely Abnormal'
        else:
            return 'Unknown'
    
    def create_clinical_summary(self, subject_id, subject_data, z_scores):
        """Create clinical summary for a subject"""
        summary = {
            'subject_id': subject_id,
            'age': subject_data['age'],
            'sex': 'Male' if subject_data['sex'] == 1 else 'Female',
            'condition': z_scores['condition'],
            'age_group': z_scores['age_group']
        }
        
        # Add clinical metrics
        for metric in ['peak_alpha_freq', 'theta_beta_ratio', 'beta_alpha_ratio']:
            if metric in z_scores['z_scores_clinical']:
                z_score = np.mean(z_scores['z_scores_clinical'][metric])
                summary[f'{metric}_z_score'] = z_score
                summary[f'{metric}_classification'] = self.classify_abnormality(z_score)
        
        # Add band power summaries
        for band in ['alpha', 'beta', 'theta', 'delta']:
            if band in z_scores['z_scores_bands']:
                z_score = np.mean(z_scores['z_scores_bands'][band])
                summary[f'{band}_z_score'] = z_score
                summary[f'{band}_classification'] = self.classify_abnormality(z_score)
        
        return summary
    
    def create_z_score_tables(self):
        """Create comprehensive z-score tables for all subjects"""
        if len(self.subjects_data) == 0:
            print("⚠️ No subjects processed, returning empty tables")
            return pd.DataFrame(), pd.DataFrame()
            
        print("📋 Creating z-score tables...")
        
        z_score_data = []
        clinical_summaries = []
        
        for subject_id, subject_data in self.subjects_data.items():
            # Determine condition from subject ID
            condition = 'EC' if subject_id.startswith('A') else 'EO'
            
            # Calculate z-scores
            z_scores = self.calculate_z_scores(subject_data, condition)
            if z_scores is None:
                continue
            
            # Create clinical summary
            clinical_summary = self.create_clinical_summary(subject_id, subject_data, z_scores)
            clinical_summaries.append(clinical_summary)
            
            # Create row for each electrode and frequency band
            for i, electrode in enumerate(self.electrodes):
                for band in self.freq_bands.keys():
                    if band in z_scores['z_scores_bands']:
                        z_score_data.append({
                            'subject_id': subject_id,
                            'condition': condition,
                            'age': subject_data['age'],
                            'sex': 'Male' if subject_data['sex'] == 1 else 'Female',
                            'age_group': z_scores['age_group'],
                            'electrode': electrode,
                            'frequency_band': band,
                            'z_score': z_scores['z_scores_bands'][band][i],
                            'classification': self.classify_abnormality(z_scores['z_scores_bands'][band][i]),
                            'raw_power': subject_data['band_powers'][band][i],
                            'normative_mean': self.normative_data[z_scores['age_group']]['band_powers_mean'][band][i],
                            'normative_std': self.normative_data[z_scores['age_group']]['band_powers_std'][band][i]
                        })
        
        # Create DataFrames
        z_score_df = pd.DataFrame(z_score_data)
        clinical_summary_df = pd.DataFrame(clinical_summaries)
        
        print(f"✅ Created z-score table with {len(z_score_df)} rows")
        print(f"✅ Created clinical summary with {len(clinical_summary_df)} subjects")
        
        return z_score_df, clinical_summary_df
    
    def save_database(self, output_path="eeg_paradox_database"):
        """Save all processed data and tables"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        print(f" Saving database to {output_path}...")
        
        # Save z-score table
        z_score_df, clinical_summary_df = self.create_z_score_tables()
        z_score_df.to_csv(output_path / "z_scores_table.csv", index=False)
        z_score_df.to_excel(output_path / "z_scores_table.xlsx", index=False)
        
        # Save clinical summary
        clinical_summary_df.to_csv(output_path / "clinical_summary.csv", index=False)
        clinical_summary_df.to_excel(output_path / "clinical_summary.xlsx", index=False)
        
        # Save normative data
        normative_summary = []
        for age_group, data in self.normative_data.items():
            normative_summary.append({
                'age_group': age_group,
                'n_subjects': data['n_subjects'],
                'age_range': f"{min(data['ages'])}-{max(data['ages'])}",
                'mean_age': np.mean(data['ages']),
                'male_count': sum(1 for s in data['sexes'] if s == 1),
                'female_count': sum(1 for s in data['sexes'] if s == 2)
            })
        
        normative_df = pd.DataFrame(normative_summary)
        normative_df.to_csv(output_path / "normative_summary.csv", index=False)
        normative_df.to_excel(output_path / "normative_summary.xlsx", index=False)
        
        # Save processed data
        np.save(output_path / "normative_data.npy", self.normative_data)
        np.save(output_path / "subjects_data.npy", self.subjects_data)
        
        print("✅ Database saved successfully!")
        
        return z_score_df, clinical_summary_df, normative_df
    
    def create_clinical_visualizations(self, output_path="eeg_paradox_database"):
        """Create clinical QEEG visualizations"""
        if len(self.subjects_data) == 0:
            print("⚠️ No subjects processed, skipping clinical visualizations")
            return
            
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        print("🎨 Creating clinical visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12
        
        # 1. Clinical metrics distribution
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('EEG Paradox Rapid Reporter - Clinical QEEG Metrics', fontsize=20, color='cyan')
        
        # Theta/Beta ratio distribution
        theta_beta_ratios = []
        for data in self.subjects_data.values():
            if 'theta_beta_ratio' in data['clinical_metrics']:
                theta_beta_ratios.extend(data['clinical_metrics']['theta_beta_ratio'])
        
        if theta_beta_ratios:
            axes[0, 0].hist(theta_beta_ratios, bins=30, color='cyan', alpha=0.7, edgecolor='white')
            axes[0, 0].set_title('Theta/Beta Ratio Distribution', color='white', fontsize=16)
            axes[0, 0].set_xlabel('Theta/Beta Ratio', color='white')
            axes[0, 0].set_ylabel('Frequency', color='white')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Peak Alpha Frequency distribution
        paf_values = []
        for data in self.subjects_data.values():
            if 'peak_alpha_freq' in data['clinical_metrics']:
                paf_values.extend(data['clinical_metrics']['peak_alpha_freq'])
        
        if paf_values:
            axes[0, 1].hist(paf_values, bins=30, color='magenta', alpha=0.7, edgecolor='white')
            axes[0, 1].set_title('Peak Alpha Frequency Distribution', color='white', fontsize=16)
            axes[0, 1].set_xlabel('Peak Alpha Frequency (Hz)', color='white')
            axes[0, 1].set_ylabel('Frequency', color='white')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Beta/Alpha ratio distribution
        beta_alpha_ratios = []
        for data in self.subjects_data.values():
            if 'beta_alpha_ratio' in data['clinical_metrics']:
                beta_alpha_ratios.extend(data['clinical_metrics']['beta_alpha_ratio'])
        
        if beta_alpha_ratios:
            axes[1, 0].hist(beta_alpha_ratios, bins=30, color='yellow', alpha=0.7, edgecolor='white')
            axes[1, 0].set_title('Beta/Alpha Ratio Distribution', color='white', fontsize=16)
            axes[1, 0].set_xlabel('Beta/Alpha Ratio', color='white')
            axes[1, 0].set_ylabel('Frequency', color='white')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Age vs Peak Alpha Frequency
        ages = []
        paf_by_age = []
        for data in self.subjects_data.values():
            if 'peak_alpha_freq' in data['clinical_metrics']:
                ages.append(data['age'])
                paf_by_age.append(np.mean(data['clinical_metrics']['peak_alpha_freq']))
        
        if ages and paf_by_age:
            axes[1, 1].scatter(ages, paf_by_age, alpha=0.6, color='green')
            axes[1, 1].set_title('Age vs Peak Alpha Frequency', color='white', fontsize=16)
            axes[1, 1].set_xlabel('Age (years)', color='white')
            axes[1, 1].set_ylabel('Peak Alpha Frequency (Hz)', color='white')
            axes[1, 1].set_ylabel('Peak Alpha Frequency (Hz)', color='white')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(ages, paf_by_age, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(ages, p(ages), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path / "clinical_metrics.png", dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        print("✅ Clinical visualizations created successfully!")
    
    def create_visualizations(self, output_path="eeg_paradox_database"):
        """Create visualizations of the normative data"""
        if len(self.subjects_data) == 0:
            print("⚠️ No subjects processed, skipping visualizations")
            return
            
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        print("🎨 Creating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12
        
        # 1. Age distribution
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('EEG Paradox Rapid Reporter - Cuban Normative Database', fontsize=20, color='cyan')
        
        # Age histogram
        ages = [data['age'] for data in self.subjects_data.values()]
        axes[0, 0].hist(ages, bins=20, color='cyan', alpha=0.7, edgecolor='white')
        axes[0, 0].set_title('Age Distribution', color='white', fontsize=16)
        axes[0, 0].set_xlabel('Age (years)', color='white')
        axes[0, 0].set_ylabel('Number of Subjects', color='white')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sex distribution
        sexes = [data['sex'] for data in self.subjects_data.values()]
        sex_counts = pd.Series(sexes).value_counts()
        
        # Handle case where we might not have both sexes
        if len(sex_counts) > 0:
            labels = ['Male' if i == 1 else 'Female' for i in sex_counts.index]
            axes[0, 1].pie(sex_counts.values, labels=labels, autopct='%1.1f%%', 
                           colors=['#FF6B6B', '#4ECDC4'])
            axes[0, 1].set_title('Sex Distribution', color='white', fontsize=16)
        
        # Condition distribution
        conditions = ['EC' if sid.startswith('A') else 'EO' for sid in self.subjects_data.keys()]
        condition_counts = pd.Series(conditions).value_counts()
        axes[1, 0].bar(condition_counts.index, condition_counts.values, color=['#FFE66D', '#95E1D3'])
        axes[1, 0].set_title('Data Condition Distribution', color='white', fontsize=16)
        axes[1, 0].set_ylabel('Number of Subjects', color='white')
        
        # Age groups
        age_groups = [self._get_age_group(data['age']) for data in self.subjects_data.values()]
        age_group_counts = pd.Series(age_groups).value_counts().sort_index()
        axes[1, 1].bar(range(len(age_group_counts)), age_group_counts.values, color='#A8E6CF')
        axes[1, 1].set_title('Age Group Distribution', color='white', fontsize=16)
        axes[1, 1].set_xlabel('Age Groups', color='white')
        axes[1, 1].set_ylabel('Number of Subjects', color='white')
        axes[1, 1].set_xticks(range(len(age_group_counts)))
        axes[1, 1].set_xticklabels(age_group_counts.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / "database_overview.png", dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        # 2. Normative power spectra visualization
        if len(self.normative_data) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle('EEG Paradox Rapid Reporter - Normative Power Spectra by Age Group', 
                        fontsize=20, color='cyan')
            
            age_groups_sorted = sorted(self.normative_data.keys(), 
                                      key=lambda x: float(x.split('-')[0]))
            
            for i, age_group in enumerate(age_groups_sorted[:4]):  # Show first 4 age groups
                row, col = i // 2, i % 2
                normative = self.normative_data[age_group]
                
                # Plot mean power spectra for all electrodes
                for j, electrode in enumerate(self.electrodes):
                    if j % 3 == 0:  # Show every 3rd electrode to avoid clutter
                        axes[row, col].plot(self.frequencies, 
                                          normative['power_spectra_mean'][j, :],
                                          label=electrode, alpha=0.8, linewidth=1.5)
                
                axes[row, col].set_title(f'Age Group: {age_group} (n={normative["n_subjects"]})', 
                                       color='white', fontsize=14)
                axes[row, col].set_xlabel('Frequency (Hz)', color='white')
                axes[row, col].set_ylabel('Power (μV²)', color='white')
                axes[row, col].set_xlim(0, 20)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path / "normative_power_spectra.png", dpi=300, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            plt.close()
        
        print("✅ Visualizations created successfully!")

def main():
    """Main processing function"""
    print("🚀 EEG Paradox Rapid Reporter - Clinical QEEG Database Processor")
    print("Enhanced for Jay Gunkelman & Mark Jones Clinical Standards")
    print("=" * 80)
    
    # Initialize processor
    processor = CubanEEGProcessor()
    
    # Process all subjects
    processor.process_all_subjects()
    
    # Create normative database
    processor.create_normative_database()
    
    # Save database and create tables
    z_score_df, clinical_summary_df, normative_df = processor.save_database()
    
    # Create visualizations
    processor.create_visualizations()
    processor.create_clinical_visualizations()
    
    # Display summary
    print("\n" + "=" * 80)
    print("📊 CLINICAL QEEG DATABASE SUMMARY")
    print("=" * 80)
    print(f"Total subjects processed: {len(processor.subjects_data)}")
    print(f"Age groups created: {len(processor.normative_data)}")
    print(f"Z-score table rows: {len(z_score_df)}")
    print(f"Clinical summaries: {len(clinical_summary_df)}")
    print(f"Electrodes: {len(processor.electrodes)}")
    print(f"Frequency bands: {list(processor.freq_bands.keys())}")
    print(f"Clinical metrics: Peak Alpha Frequency, Theta/Beta Ratio, Beta/Alpha Ratio")
    
    if len(processor.subjects_data) > 0:
        print("\n🎯 Clinical QEEG Features Added:")
        print("✅ Peak Alpha Frequency (PAF) analysis")
        print("✅ Theta/Beta ratio (ADHD biomarker)")
        print("✅ Beta/Alpha ratio (arousal index)")
        print("✅ Alpha asymmetry calculations")
        print("✅ Clinical abnormality classifications")
        print("✅ Relative power calculations")
        print("✅ Age-appropriate normative data")
        
        print("\n📁 Output Files:")
        print("1. z_scores_table.csv - Complete z-score database")
        print("2. clinical_summary.csv - Clinical QEEG summaries")
        print("3. normative_summary.csv - Age-group statistics")
        print("4. clinical_metrics.png - Clinical QEEG visualizations")
    else:
        print("\n⚠️ No subjects were processed successfully.")
        print("Please check the data loading errors above.")
    
    return processor, z_score_df, clinical_summary_df, normative_df

if __name__ == "__main__":
    processor, z_scores, clinical_summary, normative = main()