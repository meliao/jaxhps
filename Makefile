multi-source-gauss-bumps:
	python multi_source_wave_scattering.py -l 3 -p 16 -k 20 --dirs 20 --debug
	ffmpeg -framerate 12 -i data/multi_source_wave_scattering/gauss_bumps_k_20/multi_source_wave_scattering_%d.png -c:v libx264 -pix_fmt yuv420p data/multi_source_wave_scattering/movie_k_20_gauss_bumps.mp4

multi-source-GBM_1:
	python multi_source_wave_scattering.py -l 3 -p 16 -k 20 --dirs 30 --debug --scattering_potential GBM_1
	ffmpeg -framerate 12 -i data/multi_source_wave_scattering/GBM_1_k_20/multi_source_wave_scattering_%d.png -c:v libx264 -pix_fmt yuv420p data/multi_source_wave_scattering/movie_k_20_GBM_1.mp4