# Get the environment variables
code_dir <- Sys.getenv("CODE_DIR")
model_dir <- Sys.getenv("MODEL_DIR")

# Change to the distributed learning dir and load stufg
setwd(code_dir)
devtools::load_all()

# Perform the next training
model_file <- paste(model_dir, "train_files/model.rds", sep="/")
regis_dir <- paste(model_dir, "train_files", sep="/")

load(model_file)
trainDistributedModel(regis = regis_dir, epochs_at_once = 10L)
