import argparse
import json
import numpy as np
from functions import DataParser, ScoreCalculator, ClassSelector, Visualizer
import plotly.io as pio

def main():
    parser = argparse.ArgumentParser(description='CryoSPARC 2D Class Selection Script')
    parser.add_argument('--particles_cs', required=True, help='Path to particles.cs file')
    parser.add_argument('--class_averages', required=True, help='Path to class_averages.md file')
    parser.add_argument('--reference_3d', required=True, help='Path to 3D reference.mrc file')
    parser.add_argument('--method_type', required=True, choices=[
        'K_percentile',
        'top_n',
        'K_percentile_ssnr',
        'top_n_ssnr',
        'clustering'
    ], help='Selection method type')
    args = parser.parse_args()

    # Parse data
    dp = DataParser()
    dp.parse_cs_file(args.particles_cs)
    dp.parse_mrc(args.class_averages)

    # Initialize ScoreCalculator with reference map
    calc = ScoreCalculator(dp.class_images, reference_map=args.reference_3d)
    scores = calc.calculate_all_scores()

    # Initialize ClassSelector
    selector = ClassSelector(scores, dp.class_particles)

    # Selection based on method_type
    if args.method_type == 'K_percentile':
        k = float(input('Enter the top K percentile (e.g., 20 for top 20%): '))
        selection = selector.select_top_k_percentile(k, score_idx=0)  # Assuming Pearson score
    elif args.method_type == 'top_n':
        n = int(input('Enter the top N classes: '))
        selection = selector.select_top_n(n, score_idx=0)  # Assuming Pearson score
    elif args.method_type == 'K_percentile_ssnr':
        k = float(input('Enter the top K percentile (e.g., 20 for top 20%): '))
        ssnr_threshold = float(input('Enter the SSNR threshold (e.g., 0.5): '))
        selection = selector.select_with_ssnr_threshold(
            method='K_percentile',
            threshold=ssnr_threshold,
            k=k,
            score_idx=2  # Assuming SSNR is at index 2
        )
    elif args.method_type == 'top_n_ssnr':
        n = int(input('Enter the top N classes: '))
        ssnr_threshold = float(input('Enter the SSNR threshold (e.g., 0.5): '))
        selection = selector.select_with_ssnr_threshold(
            method='top_n',
            threshold=ssnr_threshold,
            n=n,
            score_idx=2  # Assuming SSNR is at index 2
        )
    elif args.method_type == 'clustering':
        eps = float(input('Enter DBSCAN eps value (e.g., 0.3): '))
        min_samples = int(input('Enter DBSCAN min_samples value (e.g., 5): '))
        clusters = selector.perform_clustering(eps=eps, min_samples=min_samples)
        selection = clusters != -1  # Accept points that are part of a cluster
    else:
        print('Invalid method type')
        return

    # Get accepted and rejected particles
    if args.method_type != 'clustering':
        accepted, rejected = selector.get_particle_groups(selection)
    else:
        accepted = []
        rejected = []
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:
                accepted.extend(selector.class_particles[i])
            else:
                rejected.extend(selector.class_particles[i])

    # Visualize scores
    visualizer = Visualizer(scores, clusters=clusters if args.method_type == 'clustering' else None)
    fig = visualizer.plot_3d()
    
    # Save the 3D plot
    pio.write_html(fig, file='3d_map.html', auto_open=False)

    # Export accepted and rejected particles as .star files
    dp.export_to_star(args.particles_cs, accepted, 'accepted_particles.star')
    dp.export_to_star(args.particles_cs, rejected, 'rejected_particles.star')

    print('3D map saved as 3d_map.html')
    print('Accepted particles saved as accepted_particles.star')
    print('Rejected particles saved as rejected_particles.star')

if __name__ == '__main__':
    main()