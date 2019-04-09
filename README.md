# Evaluating adaptive decomposition methods for EEG signal seizure detection and classification
Vinícius R. Carvalho*,  Márcio F.D. Moraes, Antônio P. Braga, Eduardo M.A.M. Mendes
Programa de Pós-Graduação em Engenharia Elétrica – Universidade Federal de Minas Gerais – Av. Antônio Carlos 6627, 31270-901, Belo Horizonte, MG, Brasil.
Núcleo de Neurociências, Departamento de Fisiologia e Biofísica, Instituto de Ciências Biológicas, Universidade Federal de Minas Gerais, Belo Horizonte, Brazil.

*vrcarva@ufmg.br

Scripts to decompose EEG signals from the Bonn University datbase (http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3), according to three methods: Empirical Mode Decomposition (EMD),
Empirical Wavelet Transform (EWT) and Variational Mode Decomposition. 
Several features are then extracted from each decomposed mode and the resulting matrixes are written in .csv files
Files are loaded by main_classify, which splits data into training/testing sets into 5-folds for cross-validation. 
Several classifiers are evaluated and mean performance results are presented after 10 iterations.

The following packages are required: numpy, scipy, scikit-learn, pandas, matplotlib, EMD-signal, ewtpy, vmdpy.
