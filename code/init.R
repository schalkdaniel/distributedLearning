# !!! This script is executed as part of `docker build` !!!
# !!! It will not exist inside the image after built !!!

# Get the environment variables, they are defined in the Dockerfile
model_dir <- Sys.getenv("MODEL_DIR")
code_dir <- Sys.getenv("CODE_DIR")
data_dir <- Sys.getenv("DATA_DIR")

# Change to the distributed learning dir and load stufg
setwd(code_dir)
devtools::load_all()

# The output of the distributed learning will be written to the model dir
out_dir <- model_dir

# TODO Refactor
files <- c(paste(data_dir, "iris0.csv", sep="/"),
           paste(data_dir, "iris1.csv", sep="/"),
           paste(data_dir, "iris2.csv", sep="/"))

initializeDistributedModel(formula = formula(Sepal.Length ~ Petal.Length + Sepal.Width),
			   model = "LinearModel",
			   optimizer = "gradientDescent",
			   out_dir = out_dir,
			   files = files,
			   epochs = 30000L,
			   learning_rate = 0.01,
			   mse_eps = 1e-10,
			   file_reader = read.csv,
			   overwrite = TRUE)

