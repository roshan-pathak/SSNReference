import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
import mrcfile
from scipy.ndimage import rotate
from scipy.interpolate import griddata
import json
import plotly.express as px
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr
from skimage.feature import hog
import seaborn as sns
from google.colab import files

class SSNRCalculator:
    """Calculates Spectral Signal-to-Noise Ratio for cryo-EM data"""

    def __init__(self):
        self.power_spectrum = None
        self.noise_spectrum = None

    def calculate_power_spectrum(self, image):
        """
        Calculate power spectrum of an image
        """
        ft = fftpack.fft2(image)
        ps = np.abs(ft)**2
        return fftpack.fftshift(ps)

    def radial_average(self, data):
        """
        Calculate radially averaged 1D profile from 2D data
        """
        center = np.array(data.shape) // 2
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)

        # Average over radial bins
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        return radial_profile

    def calculate_ssnr(self, particles, class_average):
        """
        Calculate SSNR for a class

        Parameters:
        -----------
        particles : ndarray
            Stack of particle images (n_particles, height, width)
        class_average : ndarray
            Class average image (height, width)

        Returns:
        --------
        ssnr : ndarray
            1D array of SSNR values as function of spatial frequency
        freq : ndarray
            Corresponding spatial frequencies
        """
        # Calculate average power spectrum
        ps_sum = np.zeros_like(self.calculate_power_spectrum(particles[0]))
        for particle in particles:
            ps = self.calculate_power_spectrum(particle)
            ps_sum += ps
        ps_avg = ps_sum / len(particles)

        # Calculate noise power spectrum
        noise_ps_sum = np.zeros_like(ps_avg)
        for particle in particles:
            noise = particle - class_average
            noise_ps = self.calculate_power_spectrum(noise)
            noise_ps_sum += noise_ps
        noise_ps = noise_ps_sum / len(particles)

        # Calculate radial averages
        ps_radial = self.radial_average(ps_avg)
        noise_ps_radial = self.radial_average(noise_ps)

        # Calculate SSNR
        ssnr = (ps_radial - noise_ps_radial) / noise_ps_radial

        # Generate frequency axis (in 1/pixel units)
        freq = np.fft.fftfreq(len(ssnr))[:len(ssnr)]

        return ssnr, freq

    def get_average_ssnr(self, ssnr, freq, freq_range=(0, 0.5)):
        """
        Calculate average SSNR within a frequency range
        """
        mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        return np.mean(ssnr[mask])

class BackProjector:
    """Handles 3D map back projection for comparison with 2D classes"""

    def __init__(self, volume):
        """
        Initialize with 3D volume

        Parameters:
        -----------
        volume : ndarray
            3D volume to project
        """
        self.volume = volume

    def project_volume(self, euler_angles, output_shape):
        """
        Create 2D projection of 3D volume at specified orientation

        Parameters:
        -----------
        euler_angles : tuple
            (phi, theta, psi) Euler angles in degrees
        output_shape : tuple
            (height, width) of desired output projection

        Returns:
        --------
        ndarray
            2D projection
        """
        phi, theta, psi = euler_angles

        # Rotate volume
        vol_rot = self.rotate_volume(self.volume, phi, theta, psi)

        # Project along Z axis
        projection = np.sum(vol_rot, axis=2)

        # Resize to desired dimensions if needed
        if projection.shape != output_shape:
            y = np.linspace(0, projection.shape[0], output_shape[0])
            x = np.linspace(0, projection.shape[1], output_shape[1])
            xx, yy = np.meshgrid(x, y)

            points = np.column_stack((xx.ravel(), yy.ravel()))
            values = projection.ravel()

            grid_x, grid_y = np.meshgrid(np.arange(output_shape[1]),
                                       np.arange(output_shape[0]))
            projection = griddata(points, values,
                                (grid_x, grid_y),
                                method='linear')

        return projection

    def rotate_volume(self, volume, phi, theta, psi):
        """
        Apply Euler angle rotations to volume
        """
        # Rotate around Z (phi)
        vol_rot = rotate(volume, phi, axes=(1, 0), reshape=False)

        # Rotate around Y (theta)
        vol_rot = rotate(vol_rot, theta, axes=(2, 0), reshape=False)

        # Rotate around Z again (psi)
        vol_rot = rotate(vol_rot, psi, axes=(1, 0), reshape=False)

        return vol_rot

    def optimize_orientation(self, target_image, initial_angles=(0,0,0),
                           angle_range=(-180,180), n_steps=10):
        """
        Find optimal projection orientation to match target image

        Parameters:
        -----------
        target_image : ndarray
            2D image to match
        initial_angles : tuple
            Starting orientation
        angle_range : tuple
            Range of angles to search
        n_steps : int
            Number of steps for each angle

        Returns:
        --------
        tuple
            Optimal (phi, theta, psi) angles
        float
            Correlation at optimal orientation
        """
        best_correlation = -1
        best_angles = initial_angles

        # Grid search over angles
        for phi in np.linspace(angle_range[0], angle_range[1], n_steps):
            for theta in np.linspace(angle_range[0], angle_range[1], n_steps):
                for psi in np.linspace(angle_range[0], angle_range[1], n_steps):
                    projection = self.project_volume((phi, theta, psi),
                                                  target_image.shape)
                    correlation = np.corrcoef(projection.ravel(),
                                           target_image.ravel())[0,1]

                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_angles = (phi, theta, psi)

        return best_angles, best_correlation

class DataParser:
    """Handles parsing of CryoSPARC and RELION data files"""
    def __init__(self):
        self.class_images = None
        self.particle_assignments = None
        self.n_classes = None
        self.class_particles = {}
        self.full_cs_data = None
        self.particle_id_field = None

    def parse_mrc(self, mrc_path):
        """Parse 2D class averages from MRC file"""
        with mrcfile.open(mrc_path) as mrc:
            self.class_images = mrc.data
            if len(mrc.data.shape) == 2:
                self.class_images = mrc.data[np.newaxis, ...]
            self.n_classes = len(self.class_images)
        return self.class_images

    def parse_cs_file(self, cs_path):
        """
        Parse particle assignments from CryoSPARC CS file

        Parameters:
        -----------
        cs_path : str
            Path to the .cs file containing particle data

        Returns:
        --------
        dict
            Dictionary mapping class IDs to lists of particle indices
        """
        # Load CS file as numpy structured array
        self.full_cs_data = np.load(cs_path, allow_pickle=True)

        # Determine particle ID field
        self.particle_id_field = self.get_particle_id_field()

        if self.particle_id_field is None:
            raise ValueError("Particle ID field not found in CS file.")

        # Extract class assignments
        try:
            class_assignments = self.full_cs_data['class']
        except KeyError:
            try:
                class_assignments = self.full_cs_data['class_id']
            except KeyError:
                try:
                    class_assignments = self.full_cs_data['alignments2D/class']
                except KeyError:
                    raise KeyError("Could not find class assignment field in CS file.")

        particle_ids = self.full_cs_data[self.particle_id_field]

        self.particle_assignments = {pid: cid for pid, cid in zip(particle_ids, class_assignments)}

        # Group particles by class
        unique_classes = np.unique(class_assignments)
        self.class_particles = {i: [] for i in unique_classes}

        for pid, cid in self.particle_assignments.items():
            self.class_particles[cid].append(pid)

        return self.class_particles

    def get_particle_id_field(self):
        """Determine the particle ID field in the CS file"""
        possible_fields = ['particle_id', 'uid']
        for field in possible_fields:
            if field in self.full_cs_data.dtype.names:
                return field
        return None

    def get_class_particles(self, class_id):
        """Get list of particles for a specific class"""
        return self.class_particles.get(class_id, [])

    def get_particle_class(self, particle_id):
        """Get class assignment for a specific particle"""
        return self.particle_assignments.get(particle_id, None)

    def print_summary(self):
        """Print summary of parsed data"""
        if self.class_images is not None:
            print(f"Number of class averages: {self.n_classes}")

        if self.class_particles:
            print("\nClass distribution:")
            for class_id, particles in self.class_particles.items():
                print(f"Class {class_id}: {len(particles)} particles")

    def export_to_star(self, original_cs_path, particle_ids, output_star_path):
        """Export selected particles to a .star file"""
        if self.full_cs_data is None:
            self.parse_cs_file(original_cs_path)
        
        # Extract fields
        field_names = self.full_cs_data.dtype.names
        data_to_export = {field: self.full_cs_data[field] for field in field_names}

        # Create mask for selected particles
        mask = np.isin(data_to_export[self.particle_id_field], particle_ids)

        # Filter data
        exported_data = {field: data_to_export[field][mask] for field in field_names}

        # Write to .star file
        with open(output_star_path, 'w') as f:
            f.write("data_\n\n")
            f.write("loop_\n")
            for i, field in enumerate(field_names, start=1):
                f.write(f"_{field} #{i}\n")
            f.write("\n")
            num_entries = len(exported_data[field_names[0]])
            for j in range(num_entries):
                row = ' '.join(str(exported_data[field][j]) for field in field_names)
                f.write(f"{row}\n")

class ScoreCalculator:
    """Calculates various scores for 2D classes"""
    def __init__(self, class_images, reference_map):
        self.class_images = class_images
        self.reference_map = reference_map

    def calculate_ssnr(self, class_idx):
      particles = self.get_particles_for_class(class_idx)
      class_average = self.class_images[class_idx]
      ssnr_calc = SSNRCalculator()
      ssnr, freq = ssnr_calc.calculate_ssnr(particles, class_average)
      return ssnr_calc.get_average_ssnr(ssnr, freq)

    def calculate_pearson_score(self, class_image, projection):
        """Calculate Pearson correlation between class and projection"""
        return pearsonr(class_image.flatten(), projection.flatten())[0]

    def calculate_hog_score(self, class_image, projection):
        """Calculate HOG score between class and projection"""
        class_hog = hog(class_image)
        proj_hog = hog(projection)
        return np.dot(class_hog, proj_hog) / (np.linalg.norm(class_hog) * np.linalg.norm(proj_hog))

    def calculate_all_scores(self):
        """Calculate all scores for each class"""
        scores = []
        for i in range(len(self.class_images)):
            # Get projection of 3D map for this view
            projection = self.get_projection(i)  # This needs to be implemented

            pearson = self.calculate_pearson_score(self.class_images[i], projection)
            hog = self.calculate_hog_score(self.class_images[i], projection)
            ssnr = self.calculate_ssnr(i)  # This needs particle data

            scores.append([pearson, hog, ssnr])

        return np.array(scores)

class ClassSelector:
    """Handles class selection based on various criteria"""
    def __init__(self, scores, class_particles):
        self.scores = scores
        self.class_particles = class_particles
        self.clusters = None

    def select_top_k_percentile(self, k, score_idx=0):
        """Select top K percentile classes by specified score"""
        threshold = np.percentile(self.scores[:, score_idx], 100-k)
        return self.scores[:, score_idx] >= threshold

    def select_top_n(self, n, score_idx=0):
        """Select top N classes by specified score"""
        indices = np.argsort(self.scores[:, score_idx])[-n:]
        selected = np.zeros(len(self.scores), dtype=bool)
        selected[indices] = True
        return selected

    def select_with_ssnr_threshold(self, method, threshold, **kwargs):
        """Select classes using specified method and SSNR threshold"""
        if method == 'top_n':
            base_selection = self.select_top_n(**kwargs)
        else:
            base_selection = self.select_top_k_percentile(**kwargs)

        ssnr_mask = self.scores[:, 2] >= threshold
        return base_selection & ssnr_mask

    def perform_clustering(self, eps=0.3, min_samples=5):
        """Perform DBSCAN clustering on score space"""
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = clustering.fit_predict(self.scores)
        return self.clusters

    def get_particle_groups(self, selection):
        """Split particles into accepted and rejected groups"""
        accepted = []
        rejected = []

        for i, selected in enumerate(selection):
            if selected:
                accepted.extend(self.class_particles[i])
            else:
                rejected.extend(self.class_particles[i])

        return accepted, rejected

class Visualizer:
    """Handles all visualization tasks"""
    def __init__(self, scores, clusters=None):
        self.scores = scores
        self.clusters = clusters

    def plot_3d(self):
        """Create interactive 3D plot"""
        fig = px.scatter_3d(
            x=self.scores[:, 0],
            y=self.scores[:, 1],
            z=self.scores[:, 2],
            color=self.clusters if self.clusters is not None else None,
            labels={'x': 'Pearson Score', 'y': 'HOG Score', 'z': 'SSNR'}
        )
        return fig

    def plot_2d_projections(self):
        """Create 2D projection plots"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Pearson vs HOG
        ax1.scatter(self.scores[:, 0], self.scores[:, 1])
        ax1.set_xlabel('Pearson Score')
        ax1.set_ylabel('HOG Score')

        # Pearson vs SSNR
        ax2.scatter(self.scores[:, 0], self.scores[:, 2])
        ax2.set_xlabel('Pearson Score')
        ax2.set_ylabel('SSNR')

        # HOG vs SSNR
        ax3.scatter(self.scores[:, 1], self.scores[:, 2])
        ax3.set_xlabel('HOG Score')
        ax3.set_ylabel('SSNR')

        plt.tight_layout()
        return fig
