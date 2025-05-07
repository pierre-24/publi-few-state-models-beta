all: main SI

main:
	python3 plot_exci.py -o ../Figure3.pdf
	python3 plot_sym_beta_SHS.py -o ../Figure5.pdf
	python3 plot_sym_beta_Jisq.py -o ../Figure7.pdf
	python3 plot_3s_heatmap.py -o ../Figure8a.pdf -T 0.5
	python3 plot_3s_heatmap.py -o ../Figure8b.pdf -T 0.1
	python3 plot_hs_all.py -o ../Figure9.pdf
	python3 plot_hs_theta_all.py -o ../Figure10.pdf
	python3 plot_3s_max.py -o ../Figure11.pdf

SI:
	python3 plot_xs_components.py -S 2 -C "zzz" -o ../FigureS1.pdf
	python3 plot_xs_components.py -S 3 -C "zzz,zxx" -o ../FigureS2a.pdf --theta 30
	python3 plot_xs_components.py -S 3 -C "zzz,zxx" -o ../FigureS2b.pdf --theta 60
	python3 plot_xs_components.py -S 4 -C "yyy,zzz,zyy" -o ../FigureS3a.pdf --theta 30
	python3 plot_xs_components.py -S 4 -C "yyy,zzz,zyy" -o ../FigureS3b.pdf --theta 60
	python3 plot_xs_components.py -S 4 -C "yyy" -o ../FigureS3c.pdf --theta 90
	python3 plot_xs_components.py -S 5 -C "xyz" -o ../FigureS4.pdf --theta 90
	python3 plot_all.py -o ../FigureS5.pdf

clean:
	rm -rf ../FigureS*.pdf ../Figure*.pdf
