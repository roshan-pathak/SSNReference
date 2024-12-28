# SSNRCalculator
Calculates Spectral Signal-to-Noise Ratio (SSNR) for cryo-EM data.

__init__: Initializes power and noise spectra.
calculate_power_spectrum(image): Computes the power spectrum of an image.
radial_average(data): Calculates the radially averaged 1D profile from 2D data.
calculate_ssnr(particles, class_average): Computes SSNR for a given class by comparing particle power spectra to the class average.
get_average_ssnr(ssnr, freq, freq_range): Calculates the average SSNR within a specified frequency range.
BackProjector
Handles 3D map back projection for comparison with 2D classes.

__init__(volume): Initializes with a 3D volume.
project_volume(euler_angles, output_shape): Creates a 2D projection of the 3D volume at specified Euler angles.
rotate_volume(volume, phi, theta, psi): Applies Euler angle rotations to the volume.
optimize_orientation(target_image, initial_angles, angle_range, n_steps): Finds the optimal projection orientation to match a target image using a grid search.
# DataParser
Parses CryoSPARC and RELION data files.

__init__(): Initializes class images and particle assignments.
parse_mrc(mrc_path): Parses 2D class averages from an MRC file.
parse_cs_file(cs_path): Parses particle assignments from a CryoSPARC CS file.
get_class_particles(class_id): Retrieves particles for a specific class.
get_particle_class(particle_id): Retrieves the class assignment for a specific particle.
print_summary(): Prints a summary of the parsed data.
# ScoreCalculator
Calculates various scores for 2D classes.

__init__(class_images, reference_map): Initializes with class images and a reference map.
calculate_ssnr(class_idx): Computes the SSNR for a specific class.
calculate_pearson_score(class_image, projection): Calculates Pearson correlation between a class image and its projection.
calculate_hog_score(class_image, projection): Computes the Histogram of Oriented Gradients (HOG) score between a class image and its projection.
calculate_all_scores(): Calculates all scores (Pearson, HOG, SSNR) for each class.
# ClassSelector
Handles class selection based on various criteria.

__init__(scores, class_particles): Initializes with scores and class-particle mappings.
select_top_k_percentile(k, score_idx): Selects the top K percentile of classes based on a specified score.
select_top_n(n, score_idx): Selects the top N classes based on a specified score.
select_with_ssnr_threshold(method, threshold, **kwargs): Selects classes using a specified method and SSNR threshold.
perform_clustering(eps, min_samples): Performs DBSCAN clustering on the score space.
get_particle_groups(selection): Splits particles into accepted and rejected groups based on selection.
# Visualizer
Handles all visualization tasks.

__init__(scores, clusters): Initializes with scores and optional cluster information.
plot_3d(): Creates an interactive 3D scatter plot of the scores.
plot_2d_projections(): Generates 2D scatter plots projecting the scores onto different planes (Pearson vs. HOG, Pearson vs. SSNR, HOG vs. SSNR).