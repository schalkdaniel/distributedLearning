setwd('/opt')
devtools::load_all()

out_dir <- '/model'
files <- c('/data/iris1.csv', '/data/iris2.csv', '/data/iris3.csv')

registry_dir = initializeDistributedModel(formula = formula(Sepal.Length ~ Petal.Length + Sepal.Width),
					  model = "LinearModel",
					  optimizer = "gradientDescent",
					  out_dir = out_dir,
					  files = files,
					  epochs = 30000L,
					  learning_rate = 0.01,
					  mse_eps = 1e-10,
					  file_reader = read.csv,
					  overwrite = TRUE)

