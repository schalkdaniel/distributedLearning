# Get the environment variables
model_dir <- Sys.getenv("MODEL_DIR")
code_dir <- Sys.getenv("CODE_DIR")
data_dir <- Sys.getenv("DATA_DIR")

# Change to the distributed learning dir and load stuff
setwd(code_dir)
devtools::load_all()

# Setup file path
model_file <- paste(model_dir, "train_files/model.rds", sep="/")
regis_dir <- paste(model_dir, "train_files", sep="/")

# Load the model
load(model_file)

# Print the model as a summary of the state of the algorithm
show(model)

