# Get the environment variables
code_dir <- Sys.getenv("CODE_DIR")
model_dir <- Sys.getenv("MODEL_DIR")
registry_file <- Sys.getenv("REGISTRY_FILE")

# Change to the distributed learning dir and load stufg
setwd(code_dir)
devtools::load_all()

# Perform the next training
load(registry_file)
load(paste(model_dir, "train_files/model.rds", sep="/"))
trainDistributedModel(regis = registry_dir, epochs_at_once = 10L)
